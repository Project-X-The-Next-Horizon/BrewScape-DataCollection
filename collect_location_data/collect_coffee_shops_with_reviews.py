#!/usr/bin/env python3
"""
Collect coffee shop data and reviews from Google Places API (New).

Input:
    lat_lng_radius.json (array of objects with lat, lng, radius, collected)

Output:
    coffee_shops_with_reviews.csv

Resumable execution flow:
1) Load location seeds from lat_lng_radius.json and skip circles already marked
   as collected.
2) Bootstrap the existing CSV into an in-memory place_id ledger so repeat runs
   do not append duplicate places.
3) Run Nearby Search (New) around each pending seed circle using several
   coffee-related place types and multiple ranking modes.
4) Run Text Search (New) over the seed's bounding rectangle for several
   coffee-related query/type pairs, recursively subdividing saturated regions.
5) After every successful API response, append only brand-new places to the CSV
   and flush immediately so progress survives interrupted sessions.
6) Retry failed API requests up to 3 total attempts with short exponential
   backoff before giving up on that request.
7) Mark a circle as collected only after all requests for that circle finish
   without API failures, then checkpoint the JSON file atomically.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
from urllib import error, request


NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
TEXT_URL = "https://places.googleapis.com/v1/places:searchText"

SLEEP_SECONDS = 0.2
MAX_REQUEST_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1.0, 2.0)
EARTH_RADIUS_METERS = 6_371_008.8
NEARBY_MAX_RESULTS = 20
TEXT_PAGE_SIZE = 20
TEXT_MAX_RESULTS = 60
MAX_TEXT_SUBDIVISION_DEPTH = 4
MIN_TEXT_RECT_SIDE_METERS = 80.0

NEARBY_INCLUDED_TYPES = [
    "cafe",
    "coffee_shop",
    "coffee_stand",
    "coffee_roastery",
]
NEARBY_RANK_PREFERENCES = ["DISTANCE", "POPULARITY"]

NEARBY_FIELD_MASK = (
    "places.id,"
    "places.displayName,"
    "places.location,"
    "places.rating,"
    "places.userRatingCount,"
    "places.reviews"
)
TEXT_FIELD_MASK = (
    "places.id,"
    "places.displayName,"
    "places.location,"
    "places.rating,"
    "places.userRatingCount,"
    "places.reviews,"
    "nextPageToken"
)

CSV_COLUMNS = [
    "place_id",
    "name",
    "lat",
    "lon",
    "average_rating",
    "total_review_count",
    "earliest_available_review_date",
    "review_1_text",
    "review_2_text",
    "review_3_text",
    "review_4_text",
    "review_5_text",
]

REPO_ROOT = Path(__file__).resolve().parent
INPUT_PATH = REPO_ROOT / "lat_lng_radius.json"
OUTPUT_PATH = REPO_ROOT / "coffee_shops_with_reviews.csv"
ENV_PATH = REPO_ROOT / ".env"
API_KEY_ENV_VAR = "GOOGLE_PLACES_API_KEY"


def _load_dotenv(path: Path) -> None:
    """Populate process environment variables from a simple .env file."""
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
        return

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("export "):
            stripped = stripped[7:].strip()

        if "=" not in stripped:
            print(
                f"[WARN] Ignoring invalid .env line {line_number} in {path}.",
                file=sys.stderr,
            )
            continue

        key, value = stripped.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


_load_dotenv(ENV_PATH)
API_KEY = os.environ.get(API_KEY_ENV_VAR, "").strip()


@dataclass(frozen=True)
class Location:
    """Validated location seed loaded from input JSON."""

    lat: float
    lng: float
    radius: float
    input_index: int
    collected: bool
    circle_id: str
    raw_row: dict[str, Any]


@dataclass(frozen=True)
class TextSearchStrategy:
    """One categorical Text Search strategy used to widen recall."""

    label: str
    text_query: str
    included_type: str


@dataclass(frozen=True)
class SearchRectangle:
    """Axis-aligned rectangle used for recursive Text Search coverage."""

    low_lat: float
    low_lng: float
    high_lat: float
    high_lng: float
    depth: int = 0


@dataclass
class SearchOutcome:
    """Results of one paginated Text Search over one rectangle."""

    place_ids: list[str]
    saturated: bool
    request_failed: bool = False


@dataclass
class NearbySearchOutcome:
    """Results of one Nearby Search request."""

    place_ids: list[str]
    request_failed: bool = False


@dataclass
class CsvLedger:
    """Append-only CSV state used to resume work across sessions."""

    path: Path
    handle: TextIO
    writer: Any
    seen_place_ids: set[str]


@dataclass
class Stats:
    """Runtime counters for progress reporting and debugging."""

    total_locations_read: int = 0
    valid_locations_processed: int = 0
    invalid_locations_skipped: int = 0
    existing_csv_rows_loaded: int = 0
    existing_place_ids_loaded: int = 0
    locations_skipped_already_collected: int = 0
    nearby_requests: int = 0
    text_requests: int = 0
    text_regions_searched: int = 0
    rectangles_split: int = 0
    raw_places_seen: int = 0
    total_candidate_place_ids: int = 0
    duplicate_place_ids_filtered: int = 0
    api_errors: int = 0
    rows_appended_this_run: int = 0
    circles_left_pending_api_errors: int = 0
    json_checkpoint_writes: int = 0


TEXT_SEARCH_STRATEGIES = (
    TextSearchStrategy(label="cafe", text_query="cafe", included_type="cafe"),
    TextSearchStrategy(
        label="coffee_shop",
        text_query="coffee shop",
        included_type="coffee_shop",
    ),
    TextSearchStrategy(
        label="coffee_stand",
        text_query="coffee stand",
        included_type="coffee_stand",
    ),
    TextSearchStrategy(
        label="coffee_roastery",
        text_query="coffee roastery",
        included_type="coffee_roastery",
    ),
)


def _is_placeholder_key(value: str) -> bool:
    """Return True when API key has not been replaced with a real key."""
    return not value.strip() or value.strip() == "YOUR_API_KEY_HERE"


def _to_float(value: Any) -> float | None:
    """Best-effort numeric conversion used for permissive input parsing."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _normalize_place_id(value: Any) -> str:
    """Normalize one place ID to a stripped string, or '' if invalid."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def filter_redundant_place_ids(place_ids: Iterable[Any]) -> list[str]:
    """Return place IDs once, preserving first-seen order."""
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()

    for raw_place_id in place_ids:
        place_id = _normalize_place_id(raw_place_id)
        if not place_id or place_id in seen_ids:
            continue
        seen_ids.add(place_id)
        ordered_ids.append(place_id)

    return ordered_ids


def _load_locations(path: Path, stats: Stats) -> tuple[list[Location], list[Any]]:
    """Load input location seeds and preserve the original JSON payload."""
    if not path.exists():
        print(f"Error: input file not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not isinstance(payload, list):
        print(f"Error: expected a JSON array in {path}.", file=sys.stderr)
        raise SystemExit(1)

    stats.total_locations_read = len(payload)
    locations: list[Location] = []

    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            stats.invalid_locations_skipped += 1
            print(f"[WARN] Skipping row {index}: expected object.", file=sys.stderr)
            continue

        lat = _to_float(row.get("lat"))
        lng = _to_float(row.get("lng"))
        radius = _to_float(row.get("radius"))

        if lat is None or lng is None or radius is None:
            stats.invalid_locations_skipped += 1
            print(
                f"[WARN] Skipping row {index}: missing/invalid lat,lng,radius.",
                file=sys.stderr,
            )
            continue

        if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
            stats.invalid_locations_skipped += 1
            print(f"[WARN] Skipping row {index}: lat/lng out of range.", file=sys.stderr)
            continue

        if not (0.0 < radius <= 50_000.0):
            stats.invalid_locations_skipped += 1
            print(
                f"[WARN] Skipping row {index}: radius must be in (0, 50000].",
                file=sys.stderr,
            )
            continue

        collected = row.get("collected") is True
        row["collected"] = collected

        circle_id = row.get("circle_id")
        if not isinstance(circle_id, str):
            circle_id = ""

        locations.append(
            Location(
                lat=lat,
                lng=lng,
                radius=radius,
                input_index=index,
                collected=collected,
                circle_id=circle_id.strip(),
                raw_row=row,
            )
        )

    return locations, payload


def _load_existing_place_ids(path: Path, stats: Stats) -> set[str]:
    """Load the existing CSV file and seed the cross-session place_id ledger."""
    if not path.exists():
        return set()

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "place_id" not in reader.fieldnames:
                print(
                    f"Error: existing CSV is missing required 'place_id' header: {path}",
                    file=sys.stderr,
                )
                raise SystemExit(1)

            seen_place_ids: set[str] = set()
            rows_loaded = 0

            for row in reader:
                rows_loaded += 1
                place_id = _normalize_place_id(row.get("place_id"))
                if place_id:
                    seen_place_ids.add(place_id)
    except OSError as exc:
        print(f"Error: could not read existing CSV {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except csv.Error as exc:
        print(f"Error: invalid CSV in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    stats.existing_csv_rows_loaded = rows_loaded
    stats.existing_place_ids_loaded = len(seen_place_ids)
    return seen_place_ids


def _open_csv_ledger(path: Path, seen_place_ids: set[str]) -> CsvLedger:
    """Open the append-only CSV ledger and create the file/header when missing."""
    file_missing = not path.exists()
    try:
        handle = path.open("a", encoding="utf-8", newline="")
    except OSError as exc:
        print(f"Error: could not open output CSV {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
    if file_missing:
        writer.writeheader()
        handle.flush()

    return CsvLedger(path=path, handle=handle, writer=writer, seen_place_ids=seen_place_ids)


def _close_csv_ledger(ledger: CsvLedger) -> None:
    """Close the CSV ledger handle safely."""
    try:
        ledger.handle.close()
    except OSError as exc:
        print(f"[WARN] Could not close output CSV {ledger.path}: {exc}", file=sys.stderr)


def _append_new_rows(
    ledger: CsvLedger,
    places: Iterable[dict[str, Any]],
    stats: Stats,
) -> int:
    """Append brand-new place rows to the CSV and flush immediately."""
    rows_to_append: list[dict[str, Any]] = []

    for place in places:
        row = _extract_place_row(place)
        if row is None:
            continue

        stats.total_candidate_place_ids += 1
        place_id = row["place_id"]
        if place_id in ledger.seen_place_ids:
            stats.duplicate_place_ids_filtered += 1
            continue

        ledger.seen_place_ids.add(place_id)
        rows_to_append.append(row)

    if not rows_to_append:
        return 0

    try:
        ledger.writer.writerows(rows_to_append)
        ledger.handle.flush()
    except OSError as exc:
        print(f"Error: could not append to output CSV {ledger.path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    stats.rows_appended_this_run += len(rows_to_append)
    return len(rows_to_append)


def _write_location_checkpoint(path: Path, payload: list[Any], stats: Stats) -> None:
    """Atomically rewrite the location JSON file after a completed circle."""
    temp_path = path.with_name(f"{path.name}.tmp")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    try:
        temp_path.write_text(serialized, encoding="utf-8")
        os.replace(temp_path, path)
    except OSError as exc:
        print(f"Error: could not write location checkpoint {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    stats.json_checkpoint_writes += 1


def _request_json(
    *,
    method: str,
    url: str,
    field_mask: str,
    stats: Stats,
    context: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Perform one API request and decode JSON safely."""
    headers = {
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": field_mask,
    }
    data: bytes | None = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    req = request.Request(url=url, data=data, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=40) as response:
            raw = response.read()
            if not raw:
                return {}
            try:
                decoded = raw.decode("utf-8")
            except UnicodeDecodeError:
                decoded = raw.decode("utf-8", errors="replace")
            return json.loads(decoded)
    except error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if len(body) > 500:
            body = body[:500] + "...(truncated)"
        print(
            f"[API ERROR] {context} -> HTTP {exc.code}: {body or exc.reason}",
            file=sys.stderr,
        )
        stats.api_errors += 1
    except error.URLError as exc:
        print(f"[API ERROR] {context} -> URL error: {exc.reason}", file=sys.stderr)
        stats.api_errors += 1
    except TimeoutError:
        print(f"[API ERROR] {context} -> request timed out.", file=sys.stderr)
        stats.api_errors += 1
    except json.JSONDecodeError as exc:
        print(f"[API ERROR] {context} -> invalid JSON response: {exc}", file=sys.stderr)
        stats.api_errors += 1
    finally:
        time.sleep(SLEEP_SECONDS)

    return None


def _haversine_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two lat/lng points in meters."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lng2 - lng1)

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def _meters_to_latitude_delta(radius_meters: float) -> float:
    """Convert a north/south distance in meters to latitude degrees."""
    return math.degrees(radius_meters / EARTH_RADIUS_METERS)


def _meters_to_longitude_delta(radius_meters: float, latitude: float) -> float:
    """Convert an east/west distance in meters to longitude degrees."""
    cos_lat = math.cos(math.radians(latitude))
    if abs(cos_lat) < 1e-12:
        return 180.0
    return math.degrees(radius_meters / (EARTH_RADIUS_METERS * cos_lat))


def _circle_to_bounding_rectangle(location: Location) -> SearchRectangle:
    """Create the smallest axis-aligned rectangle that encloses one input circle."""
    lat_delta = _meters_to_latitude_delta(location.radius)
    lng_delta = _meters_to_longitude_delta(location.radius, location.lat)
    low_lat = max(-90.0, location.lat - lat_delta)
    high_lat = min(90.0, location.lat + lat_delta)
    low_lng = max(-180.0, location.lng - lng_delta)
    high_lng = min(180.0, location.lng + lng_delta)
    return SearchRectangle(
        low_lat=low_lat,
        low_lng=low_lng,
        high_lat=high_lat,
        high_lng=high_lng,
        depth=0,
    )


def _point_in_rectangle(lat: float, lng: float, rectangle: SearchRectangle) -> bool:
    """Return True when a point falls inside the rectangle bounds."""
    return (
        rectangle.low_lat <= lat <= rectangle.high_lat
        and rectangle.low_lng <= lng <= rectangle.high_lng
    )


def _rectangle_intersects_circle(
    rectangle: SearchRectangle,
    center_lat: float,
    center_lng: float,
    radius: float,
) -> bool:
    """Return True when a rectangle intersects the input search circle."""
    closest_lat = min(max(center_lat, rectangle.low_lat), rectangle.high_lat)
    closest_lng = min(max(center_lng, rectangle.low_lng), rectangle.high_lng)
    distance = _haversine_meters(center_lat, center_lng, closest_lat, closest_lng)
    return distance <= radius


def _rectangle_long_side_meters(rectangle: SearchRectangle) -> float:
    """Return the longer side length of a rectangle in meters."""
    mid_lat = (rectangle.low_lat + rectangle.high_lat) / 2.0
    mid_lng = (rectangle.low_lng + rectangle.high_lng) / 2.0
    width = _haversine_meters(mid_lat, rectangle.low_lng, mid_lat, rectangle.high_lng)
    height = _haversine_meters(rectangle.low_lat, mid_lng, rectangle.high_lat, mid_lng)
    return max(width, height)


def _can_split_rectangle(rectangle: SearchRectangle) -> bool:
    """Return True when the rectangle is still large enough to subdivide."""
    if rectangle.depth >= MAX_TEXT_SUBDIVISION_DEPTH:
        return False
    if rectangle.high_lat <= rectangle.low_lat or rectangle.high_lng <= rectangle.low_lng:
        return False
    return _rectangle_long_side_meters(rectangle) > MIN_TEXT_RECT_SIDE_METERS


def _subdivide_rectangle(rectangle: SearchRectangle) -> list[SearchRectangle]:
    """Split one rectangle into four child rectangles."""
    mid_lat = (rectangle.low_lat + rectangle.high_lat) / 2.0
    mid_lng = (rectangle.low_lng + rectangle.high_lng) / 2.0
    next_depth = rectangle.depth + 1

    children = [
        SearchRectangle(rectangle.low_lat, rectangle.low_lng, mid_lat, mid_lng, next_depth),
        SearchRectangle(rectangle.low_lat, mid_lng, mid_lat, rectangle.high_lng, next_depth),
        SearchRectangle(mid_lat, rectangle.low_lng, rectangle.high_lat, mid_lng, next_depth),
        SearchRectangle(mid_lat, mid_lng, rectangle.high_lat, rectangle.high_lng, next_depth),
    ]

    valid_children: list[SearchRectangle] = []
    for child in children:
        if child.high_lat > child.low_lat and child.high_lng > child.low_lng:
            valid_children.append(child)
    return valid_children


def _extract_localized_text(value: Any) -> str:
    """Extract text from Google LocalizedText objects or raw string values."""
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text.strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_review_text(review: dict[str, Any]) -> str:
    """Prefer translated/localized review text and fallback to original text."""
    text = _extract_localized_text(review.get("text"))
    if text:
        return text
    return _extract_localized_text(review.get("originalText"))


def _parse_publish_time(value: str) -> datetime | None:
    """Parse RFC3339-like timestamps returned by Places reviews."""
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _extract_earliest_available_review_date(reviews: list[dict[str, Any]]) -> str:
    """Compute oldest review date from the up-to-5 review payload."""
    parsed_times: list[datetime] = []
    fallback_raw_times: list[str] = []

    for review in reviews:
        publish_time = review.get("publishTime")
        if not isinstance(publish_time, str):
            continue

        publish_time = publish_time.strip()
        if not publish_time:
            continue

        fallback_raw_times.append(publish_time)
        parsed = _parse_publish_time(publish_time)
        if parsed is not None:
            parsed_times.append(parsed)

    if parsed_times:
        return min(parsed_times).date().isoformat()

    if fallback_raw_times:
        raw = min(fallback_raw_times)
        if "T" in raw:
            return raw.split("T", maxsplit=1)[0]
        return raw[:10]

    return ""


def _extract_place_row(place: dict[str, Any]) -> dict[str, Any] | None:
    """Transform one search result place object into one CSV row."""
    place_id = _normalize_place_id(place.get("id"))
    if not place_id:
        return None

    display_name = _extract_localized_text(place.get("displayName"))
    place_location = place.get("location")
    if not isinstance(place_location, dict):
        return None

    lat_value = _to_float(place_location.get("latitude"))
    lon_value = _to_float(place_location.get("longitude"))
    if lat_value is None or lon_value is None:
        return None

    rating = _to_float(place.get("rating"))
    user_rating_count = place.get("userRatingCount")
    if isinstance(user_rating_count, bool):
        user_rating_count = ""
    elif not isinstance(user_rating_count, int):
        if isinstance(user_rating_count, float):
            user_rating_count = int(user_rating_count)
        elif isinstance(user_rating_count, str) and user_rating_count.strip().isdigit():
            user_rating_count = int(user_rating_count.strip())
        else:
            user_rating_count = ""

    reviews_raw = place.get("reviews")
    reviews: list[dict[str, Any]] = []
    if isinstance(reviews_raw, list):
        for review in reviews_raw[:5]:
            if isinstance(review, dict):
                reviews.append(review)

    review_texts = [_extract_review_text(review) for review in reviews]
    while len(review_texts) < 5:
        review_texts.append("")

    earliest_date = _extract_earliest_available_review_date(reviews)

    return {
        "place_id": place_id,
        "name": display_name,
        "lat": lat_value,
        "lon": lon_value,
        "average_rating": rating if rating is not None else "",
        "total_review_count": user_rating_count,
        "earliest_available_review_date": earliest_date,
        "review_1_text": review_texts[0],
        "review_2_text": review_texts[1],
        "review_3_text": review_texts[2],
        "review_4_text": review_texts[3],
        "review_5_text": review_texts[4],
    }


def _place_matches_seed(
    place: dict[str, Any],
    location: Location,
    rectangle: SearchRectangle | None = None,
) -> bool:
    """Return True when a search result belongs inside the intended seed area."""
    place_location = place.get("location")
    if not isinstance(place_location, dict):
        return False

    place_lat = _to_float(place_location.get("latitude"))
    place_lng = _to_float(place_location.get("longitude"))
    if place_lat is None or place_lng is None:
        return False

    if rectangle is not None and not _point_in_rectangle(place_lat, place_lng, rectangle):
        return False

    distance = _haversine_meters(location.lat, location.lng, place_lat, place_lng)
    return distance <= location.radius


def _dedupe_place_payloads(
    places: Iterable[dict[str, Any]],
    stats: Stats,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collapse repeated place payloads by place ID while preserving order."""
    ordered_places: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []

    for place in places:
        place_id = _normalize_place_id(place.get("id"))
        if not place_id:
            continue
        if place_id in ordered_places:
            stats.duplicate_place_ids_filtered += 1
            continue
        ordered_places[place_id] = place
        ordered_ids.append(place_id)

    return list(ordered_places.values()), ordered_ids


def _search_nearby_for_location(
    location: Location,
    *,
    rank_preference: str,
    location_seq: int,
    total_locations: int,
    ledger: CsvLedger,
    stats: Stats,
) -> NearbySearchOutcome:
    """Run one Nearby Search request for a seed circle."""
    payload = {
        "includedTypes": NEARBY_INCLUDED_TYPES,
        "maxResultCount": NEARBY_MAX_RESULTS,
        "rankPreference": rank_preference,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": location.lat, "longitude": location.lng},
                "radius": location.radius,
            }
        },
    }

    stats.nearby_requests += 1
    response = _request_json(
        method="POST",
        url=NEARBY_URL,
        field_mask=NEARBY_FIELD_MASK,
        stats=stats,
        context=(
            "searchNearby "
            f"(location {location_seq}/{total_locations}, input row {location.input_index}, "
            f"rank={rank_preference})"
        ),
        payload=payload,
    )
    if response is None:
        return NearbySearchOutcome(place_ids=[], request_failed=True)

    raw_places = response.get("places")
    if not isinstance(raw_places, list):
        return NearbySearchOutcome(place_ids=[], request_failed=False)

    stats.raw_places_seen += len(raw_places)
    matching_places = [
        place
        for place in raw_places
        if isinstance(place, dict) and _place_matches_seed(place, location)
    ]
    deduped_places, place_ids = _dedupe_place_payloads(matching_places, stats)
    _append_new_rows(ledger, deduped_places, stats)
    return NearbySearchOutcome(place_ids=place_ids, request_failed=False)


def _search_text_rectangle(
    location: Location,
    rectangle: SearchRectangle,
    strategy: TextSearchStrategy,
    *,
    location_seq: int,
    total_locations: int,
    ledger: CsvLedger,
    stats: Stats,
) -> SearchOutcome:
    """Run one paginated Text Search over one rectangle."""
    stats.text_regions_searched += 1

    ordered_ids: list[str] = []
    next_page_token: str | None = None
    seen_tokens: set[str] = set()
    raw_result_count = 0

    while True:
        payload: dict[str, Any] = {
            "textQuery": strategy.text_query,
            "includedType": strategy.included_type,
            "strictTypeFiltering": True,
            "pageSize": TEXT_PAGE_SIZE,
            "rankPreference": "DISTANCE",
            "locationRestriction": {
                "rectangle": {
                    "low": {
                        "latitude": rectangle.low_lat,
                        "longitude": rectangle.low_lng,
                    },
                    "high": {
                        "latitude": rectangle.high_lat,
                        "longitude": rectangle.high_lng,
                    },
                }
            },
        }
        if next_page_token:
            payload["pageToken"] = next_page_token

        stats.text_requests += 1
        response = _request_json(
            method="POST",
            url=TEXT_URL,
            field_mask=TEXT_FIELD_MASK,
            stats=stats,
            context=(
                "searchText "
                f"(location {location_seq}/{total_locations}, input row {location.input_index}, "
                f"strategy={strategy.label}, depth={rectangle.depth}, "
                f"page_token={'yes' if next_page_token else 'no'})"
            ),
            payload=payload,
        )
        if response is None:
            return SearchOutcome(
                place_ids=ordered_ids,
                saturated=raw_result_count >= TEXT_MAX_RESULTS,
                request_failed=True,
            )

        places = response.get("places")
        if isinstance(places, list):
            stats.raw_places_seen += len(places)
            raw_result_count += len(places)

            matching_places = [
                place
                for place in places
                if isinstance(place, dict) and _place_matches_seed(place, location, rectangle)
            ]
            deduped_places, deduped_ids = _dedupe_place_payloads(matching_places, stats)
            ordered_ids.extend(deduped_ids)
            _append_new_rows(ledger, deduped_places, stats)

        if raw_result_count >= TEXT_MAX_RESULTS:
            break

        token = response.get("nextPageToken")
        if not isinstance(token, str):
            break

        token = token.strip()
        if not token or token in seen_tokens:
            break

        seen_tokens.add(token)
        next_page_token = token

    return SearchOutcome(
        place_ids=ordered_ids,
        saturated=raw_result_count >= TEXT_MAX_RESULTS,
        request_failed=False,
    )


def _collect_places_for_location(
    location: Location,
    *,
    location_seq: int,
    total_locations: int,
    ledger: CsvLedger,
    stats: Stats,
) -> tuple[list[str], bool]:
    """Collect coffee shops for one seed using nearby + adaptive text search."""
    location_candidate_ids: list[str] = []

    for rank_preference in NEARBY_RANK_PREFERENCES:
        nearby_outcome = _search_nearby_for_location(
            location,
            rank_preference=rank_preference,
            location_seq=location_seq,
            total_locations=total_locations,
            ledger=ledger,
            stats=stats,
        )
        location_candidate_ids.extend(nearby_outcome.place_ids)
        if nearby_outcome.request_failed:
            return filter_redundant_place_ids(location_candidate_ids), False

    root_rectangle = _circle_to_bounding_rectangle(location)
    search_queue: deque[tuple[SearchRectangle, TextSearchStrategy]] = deque(
        (root_rectangle, strategy) for strategy in TEXT_SEARCH_STRATEGIES
    )

    while search_queue:
        rectangle, strategy = search_queue.popleft()

        outcome = _search_text_rectangle(
            location,
            rectangle,
            strategy,
            location_seq=location_seq,
            total_locations=total_locations,
            ledger=ledger,
            stats=stats,
        )

        location_candidate_ids.extend(outcome.place_ids)
        if outcome.request_failed:
            return filter_redundant_place_ids(location_candidate_ids), False

        if not outcome.saturated or not _can_split_rectangle(rectangle):
            continue

        children = [
            child
            for child in _subdivide_rectangle(rectangle)
            if _rectangle_intersects_circle(child, location.lat, location.lng, location.radius)
        ]
        if not children:
            continue

        stats.rectangles_split += 1
        for child in children:
            search_queue.append((child, strategy))

    return filter_redundant_place_ids(location_candidate_ids), True


def main() -> int:
    """Run the full coffee-shop collection pipeline."""
    if _is_placeholder_key(API_KEY):
        print(
            f"Error: set {API_KEY_ENV_VAR} in {ENV_PATH} or your environment before running.",
            file=sys.stderr,
        )
        return 1

    stats = Stats()
    locations, payload = _load_locations(INPUT_PATH, stats)
    if not locations:
        print("Error: no valid locations to process.", file=sys.stderr)
        return 1

    seen_place_ids = _load_existing_place_ids(OUTPUT_PATH, stats)
    ledger = _open_csv_ledger(OUTPUT_PATH, seen_place_ids)
    total_locations = len(locations)
    try:
        for sequence, location in enumerate(locations, start=1):
            if location.collected:
                stats.locations_skipped_already_collected += 1
                print(
                    f"[INFO] Location {sequence}/{total_locations} "
                    f"(input row {location.input_index}) already collected; skipping."
                )
                continue

            rows_before = stats.rows_appended_this_run
            location_place_ids, completed_without_error = _collect_places_for_location(
                location,
                location_seq=sequence,
                total_locations=total_locations,
                ledger=ledger,
                stats=stats,
            )
            stats.valid_locations_processed += 1

            new_rows_for_location = stats.rows_appended_this_run - rows_before
            if completed_without_error:
                location.raw_row["collected"] = True
                _write_location_checkpoint(INPUT_PATH, payload, stats)
            else:
                stats.circles_left_pending_api_errors += 1

            status_text = "completed" if completed_without_error else "left pending"
            print(
                f"[INFO] Location {sequence}/{total_locations} (input row {location.input_index}) "
                f"surfaced {len(location_place_ids)} unique place IDs, "
                f"appended {new_rows_for_location} new rows, {status_text}."
            )
    finally:
        _close_csv_ledger(ledger)

    print("")
    print("Collection complete.")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Total locations read: {stats.total_locations_read}")
    print(f"Existing CSV rows loaded: {stats.existing_csv_rows_loaded}")
    print(f"Existing unique place IDs loaded: {stats.existing_place_ids_loaded}")
    print(f"Valid locations processed: {stats.valid_locations_processed}")
    print(
        "Locations already collected and skipped: "
        f"{stats.locations_skipped_already_collected}"
    )
    print(f"Invalid locations skipped: {stats.invalid_locations_skipped}")
    print(f"Nearby requests sent: {stats.nearby_requests}")
    print(f"Text requests sent: {stats.text_requests}")
    print(f"Text regions searched: {stats.text_regions_searched}")
    print(f"Rectangles split: {stats.rectangles_split}")
    print(f"Raw search results seen: {stats.raw_places_seen}")
    print(f"Candidate place rows processed: {stats.total_candidate_place_ids}")
    print(f"Rows appended this run: {stats.rows_appended_this_run}")
    print(f"Duplicate place IDs filtered: {stats.duplicate_place_ids_filtered}")
    print(
        "Circles left pending because of API errors: "
        f"{stats.circles_left_pending_api_errors}"
    )
    print(f"JSON checkpoints written: {stats.json_checkpoint_writes}")
    print(f"API errors: {stats.api_errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
