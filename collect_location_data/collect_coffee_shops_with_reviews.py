#!/usr/bin/env python3
"""
Collect coffee shop data and reviews from Google Places API (New).

Input:
    lat_lng_radius.json (array of objects with lat, lng, radius)

Output:
    coffee_shops_with_reviews.csv
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request


API_KEY = "AIzaSyBkpabPLS7ejjyE7ckd65w540bA8_CkgtU"

NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
TEXT_URL = "https://places.googleapis.com/v1/places:searchText"
DETAILS_URL_TEMPLATE = "https://places.googleapis.com/v1/places/{place_id}"

SLEEP_SECONDS = 0.2
PER_LOCATION_CAP = 60
EARTH_RADIUS_METERS = 6_371_008.8

NEARBY_FIELD_MASK = "places.id"
TEXT_FIELD_MASK = "places.id,places.location,nextPageToken"
DETAILS_FIELD_MASK = "id,displayName,location,rating,userRatingCount,reviews"

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


@dataclass
class Location:
    lat: float
    lng: float
    radius: float
    input_index: int


@dataclass
class Stats:
    total_locations_read: int = 0
    valid_locations_processed: int = 0
    invalid_locations_skipped: int = 0
    total_ids_discovered: int = 0
    unique_places_written: int = 0
    duplicate_place_ids_skipped: int = 0
    api_errors: int = 0
    details_errors: int = 0


def _is_placeholder_key(value: str) -> bool:
    return not value.strip() or value.strip() == "YOUR_API_KEY_HERE"


def _to_float(value: Any) -> float | None:
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


def _load_locations(path: Path, stats: Stats) -> list[Location]:
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

        locations.append(Location(lat=lat, lng=lng, radius=radius, input_index=index))

    return locations


def _request_json(
    *,
    method: str,
    url: str,
    field_mask: str,
    stats: Stats,
    context: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
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


def _add_place_id(place_id: Any, seen: set[str], ordered_ids: list[str]) -> None:
    if not isinstance(place_id, str):
        return
    candidate = place_id.strip()
    if not candidate or candidate in seen:
        return
    seen.add(candidate)
    ordered_ids.append(candidate)


def _collect_place_ids_for_location(
    location: Location,
    location_seq: int,
    total_locations: int,
    stats: Stats,
) -> list[str]:
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()

    nearby_payload = {
        "includedTypes": ["coffee_shop"],
        "maxResultCount": 20,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": location.lat, "longitude": location.lng},
                "radius": location.radius,
            }
        },
    }
    nearby_response = _request_json(
        method="POST",
        url=NEARBY_URL,
        field_mask=NEARBY_FIELD_MASK,
        stats=stats,
        context=(
            "searchNearby "
            f"(location {location_seq}/{total_locations}, input row {location.input_index})"
        ),
        payload=nearby_payload,
    )
    if nearby_response and isinstance(nearby_response.get("places"), list):
        for place in nearby_response["places"]:
            if isinstance(place, dict):
                _add_place_id(place.get("id"), seen_ids, ordered_ids)

    next_page_token: str | None = None
    while len(ordered_ids) < PER_LOCATION_CAP:
        text_payload: dict[str, Any] = {
            "textQuery": "coffee shop",
            "includedType": "coffee_shop",
            "strictTypeFiltering": True,
            "pageSize": 20,
            "locationBias": {
                "circle": {
                    "center": {"latitude": location.lat, "longitude": location.lng},
                    "radius": location.radius,
                }
            },
        }
        if next_page_token:
            text_payload["pageToken"] = next_page_token

        text_response = _request_json(
            method="POST",
            url=TEXT_URL,
            field_mask=TEXT_FIELD_MASK,
            stats=stats,
            context=(
                "searchText "
                f"(location {location_seq}/{total_locations}, input row {location.input_index}, "
                f"page_token={'yes' if next_page_token else 'no'})"
            ),
            payload=text_payload,
        )
        if text_response is None:
            break

        places = text_response.get("places")
        if isinstance(places, list):
            for place in places:
                if not isinstance(place, dict):
                    continue

                place_id = place.get("id")
                place_location = place.get("location")
                if not isinstance(place_location, dict):
                    continue

                place_lat = _to_float(place_location.get("latitude"))
                place_lng = _to_float(place_location.get("longitude"))
                if place_lat is None or place_lng is None:
                    continue

                distance = _haversine_meters(
                    location.lat, location.lng, place_lat, place_lng
                )
                if distance <= location.radius:
                    _add_place_id(place_id, seen_ids, ordered_ids)
                if len(ordered_ids) >= PER_LOCATION_CAP:
                    break

        if len(ordered_ids) >= PER_LOCATION_CAP:
            break

        token = text_response.get("nextPageToken")
        if not isinstance(token, str) or not token.strip():
            break

        token = token.strip()
        if token == next_page_token:
            break
        next_page_token = token

    print(
        f"[INFO] Location {location_seq}/{total_locations} (input row {location.input_index}) "
        f"collected {len(ordered_ids)} unique place IDs."
    )
    return ordered_ids


def _extract_localized_text(value: Any) -> str:
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text.strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_review_text(review: dict[str, Any]) -> str:
    text = _extract_localized_text(review.get("text"))
    if text:
        return text
    return _extract_localized_text(review.get("originalText"))


def _parse_publish_time(value: str) -> datetime | None:
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


def _extract_place_row(details: dict[str, Any], fallback_place_id: str) -> dict[str, Any]:
    place_id = details.get("id")
    if not isinstance(place_id, str) or not place_id.strip():
        place_id = fallback_place_id

    display_name = _extract_localized_text(details.get("displayName"))
    place_location = details.get("location")
    lat: float | str = ""
    lon: float | str = ""
    if isinstance(place_location, dict):
        lat_value = _to_float(place_location.get("latitude"))
        lon_value = _to_float(place_location.get("longitude"))
        if lat_value is not None:
            lat = lat_value
        if lon_value is not None:
            lon = lon_value

    rating = _to_float(details.get("rating"))
    user_rating_count = details.get("userRatingCount")
    if isinstance(user_rating_count, bool):
        user_rating_count = ""
    elif not isinstance(user_rating_count, int):
        if isinstance(user_rating_count, float):
            user_rating_count = int(user_rating_count)
        elif isinstance(user_rating_count, str) and user_rating_count.strip().isdigit():
            user_rating_count = int(user_rating_count.strip())
        else:
            user_rating_count = ""

    reviews_raw = details.get("reviews")
    reviews: list[dict[str, Any]] = []
    if isinstance(reviews_raw, list):
        for review in reviews_raw[:5]:
            if isinstance(review, dict):
                reviews.append(review)

    review_texts = [_extract_review_text(review) for review in reviews]
    while len(review_texts) < 5:
        review_texts.append("")

    earliest_date = _extract_earliest_available_review_date(reviews)

    row: dict[str, Any] = {
        "place_id": place_id,
        "name": display_name,
        "lat": lat,
        "lon": lon,
        "average_rating": rating if rating is not None else "",
        "total_review_count": user_rating_count,
        "earliest_available_review_date": earliest_date,
        "review_1_text": review_texts[0],
        "review_2_text": review_texts[1],
        "review_3_text": review_texts[2],
        "review_4_text": review_texts[3],
        "review_5_text": review_texts[4],
    }
    return row


def _fetch_place_details(place_id: str, stats: Stats) -> dict[str, Any] | None:
    url = DETAILS_URL_TEMPLATE.format(place_id=parse.quote(place_id, safe=""))
    details = _request_json(
        method="GET",
        url=url,
        field_mask=DETAILS_FIELD_MASK,
        stats=stats,
        context=f"placeDetails (place_id={place_id})",
    )
    if details is None:
        stats.details_errors += 1
        return None
    if not isinstance(details, dict):
        stats.details_errors += 1
        print(
            f"[API ERROR] placeDetails (place_id={place_id}) -> unexpected response type.",
            file=sys.stderr,
        )
        return None
    return _extract_place_row(details, place_id)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if _is_placeholder_key(API_KEY):
        print(
            "Error: set API_KEY in this script before running "
            "(current value is placeholder).",
            file=sys.stderr,
        )
        return 1

    stats = Stats()
    locations = _load_locations(INPUT_PATH, stats)
    if not locations:
        print("Error: no valid locations to process.", file=sys.stderr)
        return 1

    total_locations = len(locations)
    processed_place_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    for sequence, location in enumerate(locations, start=1):
        place_ids = _collect_place_ids_for_location(
            location=location,
            location_seq=sequence,
            total_locations=total_locations,
            stats=stats,
        )
        stats.valid_locations_processed += 1
        stats.total_ids_discovered += len(place_ids)

        for place_id in place_ids:
            if place_id in processed_place_ids:
                stats.duplicate_place_ids_skipped += 1
                continue

            row = _fetch_place_details(place_id, stats)
            if row is None:
                continue

            rows.append(row)
            processed_place_ids.add(place_id)
            stats.unique_places_written += 1

    _write_csv(OUTPUT_PATH, rows)

    print("")
    print("Collection complete.")
    print(f"Input file: {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Total locations read: {stats.total_locations_read}")
    print(f"Valid locations processed: {stats.valid_locations_processed}")
    print(f"Invalid locations skipped: {stats.invalid_locations_skipped}")
    print(f"Total IDs discovered (per-location sum): {stats.total_ids_discovered}")
    print(f"Total unique places written: {stats.unique_places_written}")
    print(f"Skipped duplicate place IDs: {stats.duplicate_place_ids_skipped}")
    print(f"API errors: {stats.api_errors}")
    print(f"Place details errors: {stats.details_errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
