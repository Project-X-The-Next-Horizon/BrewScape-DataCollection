#!/usr/bin/env python3
"""
Generate optimized Chiang Mai coverage circles for lat_lng_radius.json.

Dependencies:
    pip install shapely
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

try:
    from shapely.geometry import Point, box, shape
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import transform, unary_union
except ImportError:  # pragma: no cover
    print(
        "Error: 'shapely' is not installed. Install it with: pip install shapely",
        file=sys.stderr,
    )
    raise SystemExit(1)


REPO_ROOT = Path(__file__).resolve().parent
MERGED_BORDER_PATH = REPO_ROOT / "chiang_mai_main_area_merged_border.geojson"
OUTPUT_PATH = REPO_ROOT / "lat_lng_radius.json"

NOMINATIM_USER_AGENT = "BrewScape-DataCollection/1.0 (+circle-generator)"
RELATION_IDS = ("R18271830", "R19975357", "R19033670")
CITY_RELATION_ID = "R18271830"
SUTHEP_RELATION_ID = "R19975357"
DISTRICT_RELATION_ID = "R19033670"
SUPPLEMENT_ANCHOR = (98.9674, 18.8108)  # lng, lat

# West/left side of Chang Phueak supplement uses 1000m; east/right side stays 500m.
CHANG_PHUEAK_WEST_1000_MAX_LNG = 98.952

RADIUS_500 = 500.0
RADIUS_1000 = 1000.0
COVERAGE_STEP = 100.0
EXACT_BUFFER_RESOLUTION = 64
EXACT_COVERAGE_TOLERANCE_M2 = 1.0

SHIFT_STEP = 100.0
SHIFT_MAX_DISTANCE = 2000.0
SHIFT_ANGLES = 16
MAX_AUTO_FIX_ADDITIONS = 2000
MAX_EXACT_PATCH = 20
EPS = 1e-9

COARSE_SPACING500_MIN = 700
COARSE_SPACING500_MAX = 900
COARSE_SPACING500_STEP = 25
COARSE_SPACING1000_MIN = 1400
COARSE_SPACING1000_MAX = 1800
COARSE_SPACING1000_STEP = 25

SPACING500_MIN = 650
SPACING500_MAX = 950
SPACING1000_MIN = 1300
SPACING1000_MAX = 1900

FINE_SPACING500_DELTA = 25
FINE_SPACING500_STEP = 5
FINE_SPACING1000_DELTA = 50
FINE_SPACING1000_STEP = 25
FAST_LOCAL_500_DELTA = 25
FAST_LOCAL_1000_DELTA = 25


def _fatal(message: str) -> "None":
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


@dataclass(frozen=True)
class LocalProjection:
    lng0: float
    lat0: float
    cos_lat0: float

    def to_xy(self, lng: float, lat: float) -> tuple[float, float]:
        x = (lng - self.lng0) * self.cos_lat0 * 111320.0
        y = (lat - self.lat0) * 111320.0
        return x, y

    def to_lng_lat(self, x: float, y: float) -> tuple[float, float]:
        lng = (x / (self.cos_lat0 * 111320.0)) + self.lng0
        lat = (y / 111320.0) + self.lat0
        return lng, lat

    def _forward(self, x: Any, y: Any, z: Any = None):
        if hasattr(x, "__iter__"):
            xs = [((xi - self.lng0) * self.cos_lat0 * 111320.0) for xi in x]
            ys = [((yi - self.lat0) * 111320.0) for yi in y]
            return xs, ys
        return (x - self.lng0) * self.cos_lat0 * 111320.0, (y - self.lat0) * 111320.0

    def _inverse(self, x: Any, y: Any, z: Any = None):
        if hasattr(x, "__iter__"):
            xs = [((xi / (self.cos_lat0 * 111320.0)) + self.lng0) for xi in x]
            ys = [((yi / 111320.0) + self.lat0) for yi in y]
            return xs, ys
        return (x / (self.cos_lat0 * 111320.0)) + self.lng0, (y / 111320.0) + self.lat0

    def geom_to_xy(self, geom: BaseGeometry) -> BaseGeometry:
        return transform(self._forward, geom)

    def geom_to_lng_lat(self, geom: BaseGeometry) -> BaseGeometry:
        return transform(self._inverse, geom)


@dataclass
class CandidateResult:
    spacing500: int
    spacing1000: int
    records: list[dict[str, Any]]
    count500: int
    count1000: int
    sample_auto_added: int
    exact_patch_added: int
    exact_missing_area: float


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate lat_lng_radius.json. Default uses a fast hex-spacing "
            "approximation; use --auto for full auto-tuning."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use mathematically derived spacing (sqrt(3) * radius). "
            "This is already the default."
        ),
    )
    mode_group.add_argument(
        "--auto",
        action="store_true",
        help="Use full coarse+fine auto-tuning search (slower).",
    )
    parser.add_argument(
        "--fast-local",
        action="store_true",
        help=(
            "In fast mode, evaluate a small local neighborhood (up to 9 "
            "combinations) around derived spacing."
        ),
    )
    args = parser.parse_args(argv)
    if args.fast_local and args.auto:
        parser.error("--fast-local cannot be combined with --auto")
    return args


def _load_border_geometry(path: Path) -> BaseGeometry:
    if not path.exists():
        _fatal(f"merged border file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fatal(f"invalid merged border JSON in {path}: {exc}")

    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        _fatal(f"expected FeatureCollection in {path}")

    features = data.get("features")
    if not isinstance(features, list) or not features:
        _fatal(f"expected at least one feature in {path}")

    feature0 = features[0]
    if not isinstance(feature0, dict):
        _fatal(f"invalid feature in {path}")

    geometry = feature0.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        _fatal(f"expected polygon geometry in {path}")

    geom = shape(geometry)
    if geom.is_empty:
        _fatal("merged border geometry is empty")
    return geom


def _fetch_relation_geometries() -> dict[str, BaseGeometry]:
    query = parse.urlencode(
        {
            "format": "jsonv2",
            "osm_ids": ",".join(RELATION_IDS),
            "polygon_geojson": 1,
            "namedetails": 1,
        }
    )
    url = f"https://nominatim.openstreetmap.org/lookup?{query}"
    req = request.Request(url, headers={"User-Agent": NOMINATIM_USER_AGENT})

    try:
        with request.urlopen(req, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except (error.URLError, TimeoutError) as exc:
        _fatal(f"unable to fetch relation geometries from Nominatim: {exc}")

    try:
        rows = json.loads(payload)
    except json.JSONDecodeError as exc:
        _fatal(f"invalid Nominatim response JSON: {exc}")

    if not isinstance(rows, list) or not rows:
        _fatal("Nominatim lookup returned no rows")

    by_relation: dict[str, BaseGeometry] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("osm_type") != "relation":
            continue
        osm_id = row.get("osm_id")
        if not isinstance(osm_id, int):
            continue
        relation_id = f"R{osm_id}"
        geojson = row.get("geojson")
        if not isinstance(geojson, dict):
            continue
        if geojson.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        geom = shape(geojson)
        if not geom.is_empty:
            by_relation[relation_id] = geom

    missing = [relation for relation in RELATION_IDS if relation not in by_relation]
    if missing:
        _fatal(f"missing required relation geometries: {', '.join(missing)}")
    return by_relation


def _iter_polygon_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        result: list[BaseGeometry] = []
        for sub in geom.geoms:
            result.extend(_iter_polygon_parts(sub))
        return result
    return []


def _build_zone500_with_supplement(
    geometries: dict[str, BaseGeometry],
) -> tuple[BaseGeometry, BaseGeometry]:
    city = geometries[CITY_RELATION_ID]
    suthep = geometries[SUTHEP_RELATION_ID]
    district = geometries[DISTRICT_RELATION_ID]

    city_plus_suthep = unary_union([city, suthep])
    supplement_candidate = district.difference(city_plus_suthep)
    anchor_point = Point(SUPPLEMENT_ANCHOR[0], SUPPLEMENT_ANCHOR[1])

    selected_parts = [
        part for part in _iter_polygon_parts(supplement_candidate) if part.covers(anchor_point)
    ]
    if not selected_parts:
        _fatal("unable to find Chang Phueak supplement piece using anchor point")

    supplement_piece = unary_union(selected_parts)
    zone500 = unary_union([city, supplement_piece])
    return zone500, supplement_piece


def _build_projection(geom_lng_lat: BaseGeometry) -> LocalProjection:
    centroid = geom_lng_lat.centroid
    lat0 = float(centroid.y)
    return LocalProjection(
        lng0=float(centroid.x),
        lat0=lat0,
        cos_lat0=math.cos(math.radians(lat0)),
    )


def _iter_staggered_grid(bounds: tuple[float, float, float, float], spacing: float):
    min_x, min_y, max_x, max_y = bounds
    row_step = spacing * math.sqrt(3.0) / 2.0
    y = min_y
    row_index = 0

    while y <= max_y + EPS:
        x = min_x + (spacing / 2.0 if row_index % 2 else 0.0)
        while x <= max_x + EPS:
            yield x, y
            x += spacing
        y += row_step
        row_index += 1


def _generate_zone_centers(
    zone_xy: BaseGeometry,
    spacing: float,
    radius: float,
) -> list[tuple[float, float]]:
    if zone_xy.is_empty:
        return []

    zone_buffer = zone_xy.buffer(radius)
    min_x, min_y, max_x, max_y = zone_xy.bounds
    expanded = (min_x - radius, min_y - radius, max_x + radius, max_y + radius)

    centers: list[tuple[float, float]] = []
    for x, y in _iter_staggered_grid(expanded, spacing):
        point = Point(x, y)
        if zone_buffer.covers(point):
            centers.append((x, y))
    return centers


def _distance_squared(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _is_valid_1000_center(
    center: tuple[float, float],
    zone1000_buffer_xy: BaseGeometry,
    zone500_xy: BaseGeometry,
    centers500: list[tuple[float, float]],
) -> bool:
    point = Point(center[0], center[1])
    if not zone1000_buffer_xy.covers(point):
        return False
    if zone500_xy.covers(point):
        return False

    limit_sq = RADIUS_500 * RADIUS_500
    for center500 in centers500:
        if _distance_squared(center, center500) <= limit_sq + EPS:
            return False
    return True


def _filter_1000_centers(
    candidates: list[tuple[float, float]],
    zone1000_buffer_xy: BaseGeometry,
    zone500_xy: BaseGeometry,
    centers500: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return [
        center
        for center in candidates
        if _is_valid_1000_center(center, zone1000_buffer_xy, zone500_xy, centers500)
    ]


def _build_coverage_samples(
    area_xy: BaseGeometry,
    step: float,
) -> list[tuple[float, float]]:
    min_x, min_y, max_x, max_y = area_xy.bounds
    start_x = math.floor(min_x / step) * step
    start_y = math.floor(min_y / step) * step
    end_x = math.ceil(max_x / step) * step
    end_y = math.ceil(max_y / step) * step

    samples: list[tuple[float, float]] = []
    y = start_y
    while y <= end_y + EPS:
        x = start_x
        while x <= end_x + EPS:
            point = Point(x, y)
            if area_xy.covers(point):
                samples.append((x, y))
            x += step
        y += step
    return samples


def _point_is_covered(point: tuple[float, float], circles: list[tuple[float, float, float]]) -> bool:
    x, y = point
    for center_x, center_y, radius in circles:
        dx = x - center_x
        dy = y - center_y
        if dx * dx + dy * dy <= (radius * radius) + EPS:
            return True
    return False


def _find_uncovered_samples(
    samples: list[tuple[float, float]],
    circles: list[tuple[float, float, float]],
) -> list[tuple[float, float]]:
    return [sample for sample in samples if not _point_is_covered(sample, circles)]


def _try_shifted_1000_center(
    origin: tuple[float, float],
    zone1000_buffer_xy: BaseGeometry,
    zone500_xy: BaseGeometry,
    centers500: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if _is_valid_1000_center(origin, zone1000_buffer_xy, zone500_xy, centers500):
        return origin

    angles = [(2.0 * math.pi * idx) / SHIFT_ANGLES for idx in range(SHIFT_ANGLES)]
    distance = SHIFT_STEP
    while distance <= SHIFT_MAX_DISTANCE + EPS:
        for angle in angles:
            candidate = (
                origin[0] + (distance * math.cos(angle)),
                origin[1] + (distance * math.sin(angle)),
            )
            if _is_valid_1000_center(candidate, zone1000_buffer_xy, zone500_xy, centers500):
                return candidate
        distance += SHIFT_STEP

    return None


def _center_key_xy(center: tuple[float, float], radius: float) -> tuple[float, float, int]:
    return (round(center[0], 3), round(center[1], 3), int(radius))


def _circles_from_centers(
    centers500: list[tuple[float, float]],
    centers1000: list[tuple[float, float]],
) -> list[tuple[float, float, float]]:
    circles: list[tuple[float, float, float]] = [(*center, RADIUS_500) for center in centers500]
    circles.extend([(*center, RADIUS_1000) for center in centers1000])
    return circles


def _circles_to_coverage_geometry(circles: list[tuple[float, float, float]]) -> BaseGeometry:
    if not circles:
        return Point(0.0, 0.0).buffer(0.0)
    disk_geometries = [
        Point(center_x, center_y).buffer(radius, resolution=EXACT_BUFFER_RESOLUTION)
        for center_x, center_y, radius in circles
    ]
    return unary_union(disk_geometries)


def _exact_missing_geometry(
    merged_border_xy: BaseGeometry,
    circles: list[tuple[float, float, float]],
) -> BaseGeometry:
    coverage = _circles_to_coverage_geometry(circles)
    return merged_border_xy.difference(coverage)


def _auto_fix_coverage(
    merged_border_xy: BaseGeometry,
    zone500_xy: BaseGeometry,
    zone1000_xy: BaseGeometry,
    centers500: list[tuple[float, float]],
    centers1000: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int]:
    zone500_buffer_xy = zone500_xy.buffer(RADIUS_500)
    zone1000_buffer_xy = zone1000_xy.buffer(RADIUS_1000)
    samples = _build_coverage_samples(merged_border_xy, COVERAGE_STEP)

    circles: list[tuple[float, float, float]] = [(*center, RADIUS_500) for center in centers500]
    circles.extend([(*center, RADIUS_1000) for center in centers1000])

    keys = {_center_key_xy(center, RADIUS_500) for center in centers500}
    keys.update({_center_key_xy(center, RADIUS_1000) for center in centers1000})

    added_total = 0
    rounds = 0
    while True:
        rounds += 1
        if rounds > 15:
            _fatal("coverage auto-fix exceeded maximum rounds")

        uncovered = _find_uncovered_samples(samples, circles)
        if not uncovered:
            return centers500, centers1000, added_total

        added_this_round = 0
        for sample in uncovered:
            if _point_is_covered(sample, circles):
                continue

            sample_point = Point(sample[0], sample[1])
            if zone500_buffer_xy.covers(sample_point):
                center = sample
                key = _center_key_xy(center, RADIUS_500)
                if key in keys:
                    continue
                centers500.append(center)
                circles.append((center[0], center[1], RADIUS_500))
                keys.add(key)
                added_this_round += 1
                added_total += 1
            else:
                shifted = _try_shifted_1000_center(
                    sample,
                    zone1000_buffer_xy=zone1000_buffer_xy,
                    zone500_xy=zone500_xy,
                    centers500=centers500,
                )
                if shifted is not None:
                    key = _center_key_xy(shifted, RADIUS_1000)
                    if key in keys:
                        continue
                    centers1000.append(shifted)
                    circles.append((shifted[0], shifted[1], RADIUS_1000))
                    keys.add(key)
                    added_this_round += 1
                    added_total += 1
                else:
                    key = _center_key_xy(sample, RADIUS_500)
                    if key in keys:
                        continue
                    centers500.append(sample)
                    circles.append((sample[0], sample[1], RADIUS_500))
                    keys.add(key)
                    added_this_round += 1
                    added_total += 1

            if added_total > MAX_AUTO_FIX_ADDITIONS:
                _fatal("coverage auto-fix exceeded hard cap on additions")

        if added_this_round == 0:
            _fatal("coverage auto-fix found gaps but could not add circles")


def _auto_fix_exact_coverage(
    merged_border_xy: BaseGeometry,
    zone500_xy: BaseGeometry,
    zone1000_xy: BaseGeometry,
    centers500: list[tuple[float, float]],
    centers1000: list[tuple[float, float]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int, float]:
    zone500_buffer_xy = zone500_xy.buffer(RADIUS_500)
    zone1000_buffer_xy = zone1000_xy.buffer(RADIUS_1000)
    keys = {_center_key_xy(center, RADIUS_500) for center in centers500}
    keys.update({_center_key_xy(center, RADIUS_1000) for center in centers1000})

    added_total = 0
    for _ in range(MAX_EXACT_PATCH + 1):
        circles = _circles_from_centers(centers500, centers1000)
        missing = _exact_missing_geometry(merged_border_xy, circles)
        missing_area = float(missing.area)
        if missing_area <= EXACT_COVERAGE_TOLERANCE_M2 + EPS:
            return centers500, centers1000, added_total, missing_area

        if added_total >= MAX_EXACT_PATCH:
            break

        parts = [part for part in _iter_polygon_parts(missing) if part.area > EPS]
        if not parts:
            break

        parts.sort(key=lambda geom: geom.area, reverse=True)
        added_this_round = False
        for part in parts:
            representative = part.representative_point()
            center = (float(representative.x), float(representative.y))
            if zone500_buffer_xy.covers(representative):
                key500 = _center_key_xy(center, RADIUS_500)
                if key500 in keys:
                    continue
                centers500.append(center)
                keys.add(key500)
                added_total += 1
                added_this_round = True
                break

            shifted = _try_shifted_1000_center(
                center,
                zone1000_buffer_xy=zone1000_buffer_xy,
                zone500_xy=zone500_xy,
                centers500=centers500,
            )
            if shifted is not None:
                key1000 = _center_key_xy(shifted, RADIUS_1000)
                if key1000 not in keys:
                    centers1000.append(shifted)
                    keys.add(key1000)
                    added_total += 1
                    added_this_round = True
                    break

            key500 = _center_key_xy(center, RADIUS_500)
            if key500 in keys:
                continue
            centers500.append(center)
            keys.add(key500)
            added_total += 1
            added_this_round = True
            break

        if not added_this_round:
            break

    circles = _circles_from_centers(centers500, centers1000)
    missing_area = float(_exact_missing_geometry(merged_border_xy, circles).area)
    return centers500, centers1000, added_total, missing_area


def _records_from_centers(
    projection: LocalProjection,
    centers500: list[tuple[float, float]],
    centers1000: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dedupe: set[tuple[float, float, int]] = set()

    def add_row(center: tuple[float, float], radius: int):
        lng, lat = projection.to_lng_lat(center[0], center[1])
        lat_r = round(lat, 6)
        lng_r = round(lng, 6)
        key = (lat_r, lng_r, radius)
        if key in dedupe:
            return
        dedupe.add(key)
        rows.append(
            {
                "lat": lat_r,
                "lng": lng_r,
                "radius": radius,
                "collected": False,
            }
        )

    for center in centers500:
        add_row(center, 500)
    for center in centers1000:
        add_row(center, 1000)

    rows.sort(key=lambda row: (row["radius"], row["lat"], row["lng"]))
    return rows


def _validate_no_full_swallow(
    records: list[dict[str, Any]],
    projection: LocalProjection,
) -> int:
    centers500: list[tuple[float, float]] = []
    centers1000: list[tuple[float, float]] = []

    for row in records:
        center = projection.to_xy(float(row["lng"]), float(row["lat"]))
        if int(row["radius"]) == 500:
            centers500.append(center)
        elif int(row["radius"]) == 1000:
            centers1000.append(center)

    violations = 0
    threshold_sq = RADIUS_500 * RADIUS_500
    for center500 in centers500:
        for center1000 in centers1000:
            if _distance_squared(center500, center1000) <= threshold_sq + EPS:
                violations += 1
                break
    return violations


def _validate_coverage_strict(
    records: list[dict[str, Any]],
    merged_border_xy: BaseGeometry,
    projection: LocalProjection,
) -> int:
    circles = _circles_from_records(records, projection)

    samples = _build_coverage_samples(merged_border_xy, COVERAGE_STEP)
    uncovered = _find_uncovered_samples(samples, circles)
    return len(uncovered)


def _circles_from_records(
    records: list[dict[str, Any]],
    projection: LocalProjection,
) -> list[tuple[float, float, float]]:
    circles: list[tuple[float, float, float]] = []
    for row in records:
        x, y = projection.to_xy(float(row["lng"]), float(row["lat"]))
        circles.append((x, y, float(row["radius"])))
    return circles


def _validate_exact_coverage(
    records: list[dict[str, Any]],
    merged_border_xy: BaseGeometry,
    projection: LocalProjection,
) -> float:
    circles = _circles_from_records(records, projection)
    missing = _exact_missing_geometry(merged_border_xy, circles)
    return float(missing.area)


def _iter_spacing_values(minimum: int, maximum: int, step: int) -> list[int]:
    return list(range(minimum, maximum + 1, step))


def _derive_hex_spacing(radius: float, minimum: int, maximum: int, step: int) -> int:
    if step <= 0:
        _fatal("spacing step must be positive")
    target = math.sqrt(3.0) * radius
    snapped = int(round(target / step) * step)
    return max(minimum, min(maximum, snapped))


def _fast_local_values(base: int, minimum: int, maximum: int, delta: int) -> list[int]:
    values = [base]
    if delta > 0:
        values.extend([base - delta, base + delta])
    return sorted({value for value in values if minimum <= value <= maximum})


def _candidate_sort_key(candidate: CandidateResult) -> tuple[Any, ...]:
    return (
        len(candidate.records),
        candidate.exact_patch_added,
        candidate.count500,
        -candidate.spacing500,
        -candidate.spacing1000,
        candidate.spacing500,
        candidate.spacing1000,
    )


def _is_better_candidate(candidate: CandidateResult, best: CandidateResult | None) -> bool:
    if best is None:
        return True
    return _candidate_sort_key(candidate) < _candidate_sort_key(best)


def _evaluate_spacing_candidate(
    spacing500: int,
    spacing1000: int,
    merged_border_xy: BaseGeometry,
    zone500_xy: BaseGeometry,
    zone1000_xy: BaseGeometry,
    projection: LocalProjection,
) -> CandidateResult | None:
    centers500 = _generate_zone_centers(
        zone500_xy,
        spacing=float(spacing500),
        radius=RADIUS_500,
    )
    centers1000_candidates = _generate_zone_centers(
        zone1000_xy,
        spacing=float(spacing1000),
        radius=RADIUS_1000,
    )
    centers1000 = _filter_1000_centers(
        centers1000_candidates,
        zone1000_buffer_xy=zone1000_xy.buffer(RADIUS_1000),
        zone500_xy=zone500_xy,
        centers500=centers500,
    )

    centers500, centers1000, sample_auto_added = _auto_fix_coverage(
        merged_border_xy=merged_border_xy,
        zone500_xy=zone500_xy,
        zone1000_xy=zone1000_xy,
        centers500=centers500,
        centers1000=centers1000,
    )

    centers500, centers1000, exact_patch_added, exact_missing_area = _auto_fix_exact_coverage(
        merged_border_xy=merged_border_xy,
        zone500_xy=zone500_xy,
        zone1000_xy=zone1000_xy,
        centers500=centers500,
        centers1000=centers1000,
    )
    if exact_missing_area > EXACT_COVERAGE_TOLERANCE_M2 + EPS:
        return None

    records = _records_from_centers(projection, centers500, centers1000)
    count500 = sum(1 for row in records if int(row["radius"]) == 500)
    count1000 = sum(1 for row in records if int(row["radius"]) == 1000)

    violations = _validate_no_full_swallow(records, projection)
    if violations:
        return None

    uncovered_count = _validate_coverage_strict(records, merged_border_xy, projection)
    if uncovered_count:
        return None

    exact_missing_after_rounding = _validate_exact_coverage(records, merged_border_xy, projection)
    if exact_missing_after_rounding > EXACT_COVERAGE_TOLERANCE_M2 + EPS:
        return None

    return CandidateResult(
        spacing500=spacing500,
        spacing1000=spacing1000,
        records=records,
        count500=count500,
        count1000=count1000,
        sample_auto_added=sample_auto_added,
        exact_patch_added=exact_patch_added,
        exact_missing_area=exact_missing_after_rounding,
    )


def _search_best_candidate(
    spacing500_values: list[int],
    spacing1000_values: list[int],
    merged_border_xy: BaseGeometry,
    zone500_xy: BaseGeometry,
    zone1000_xy: BaseGeometry,
    projection: LocalProjection,
) -> tuple[CandidateResult | None, int, int]:
    best: CandidateResult | None = None
    tested = 0
    valid = 0

    for spacing500 in spacing500_values:
        for spacing1000 in spacing1000_values:
            tested += 1
            candidate = _evaluate_spacing_candidate(
                spacing500=spacing500,
                spacing1000=spacing1000,
                merged_border_xy=merged_border_xy,
                zone500_xy=zone500_xy,
                zone1000_xy=zone1000_xy,
                projection=projection,
            )
            if candidate is None:
                continue
            valid += 1
            if _is_better_candidate(candidate, best):
                best = candidate

    return best, tested, valid


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    use_fast = not args.auto
    merged_border_lng_lat = _load_border_geometry(MERGED_BORDER_PATH)
    relation_geometries = _fetch_relation_geometries()

    zone500_base_lng_lat, chang_phueak_supplement_lng_lat = _build_zone500_with_supplement(
        relation_geometries
    )
    zone500_lng_lat = zone500_base_lng_lat.intersection(merged_border_lng_lat)
    if zone500_lng_lat.is_empty:
        _fatal("zone500 is empty after clipping with merged border")

    # Move only west/left side of Chang Phueak supplement to 1000m zone.
    west_mask = box(-180.0, -90.0, CHANG_PHUEAK_WEST_1000_MAX_LNG, 90.0)
    west_chang_phueak_override = (
        chang_phueak_supplement_lng_lat.intersection(west_mask).intersection(merged_border_lng_lat)
    )
    if not west_chang_phueak_override.is_empty:
        zone500_lng_lat = zone500_lng_lat.difference(west_chang_phueak_override)
        if zone500_lng_lat.is_empty:
            _fatal("zone500 became empty after applying west Chang Phueak 1000m override")

    zone1000_lng_lat = merged_border_lng_lat.difference(zone500_lng_lat)
    if zone1000_lng_lat.is_empty:
        _fatal("zone1000 is empty after subtracting zone500 from merged border")

    projection = _build_projection(merged_border_lng_lat)
    merged_border_xy = projection.geom_to_xy(merged_border_lng_lat)
    zone500_xy = projection.geom_to_xy(zone500_lng_lat)
    zone1000_xy = projection.geom_to_xy(zone1000_lng_lat)

    search_mode = "fast"
    coarse_tested = 0
    coarse_valid = 0
    fine_tested = 0
    fine_valid = 0

    if use_fast:
        search_mode = "fast-local" if args.fast_local else "fast"
        fast500 = _derive_hex_spacing(
            radius=RADIUS_500,
            minimum=SPACING500_MIN,
            maximum=SPACING500_MAX,
            step=FINE_SPACING500_STEP,
        )
        fast1000 = _derive_hex_spacing(
            radius=RADIUS_1000,
            minimum=SPACING1000_MIN,
            maximum=SPACING1000_MAX,
            step=FINE_SPACING1000_STEP,
        )

        if args.fast_local:
            fast500_values = _fast_local_values(
                base=fast500,
                minimum=SPACING500_MIN,
                maximum=SPACING500_MAX,
                delta=FAST_LOCAL_500_DELTA,
            )
            fast1000_values = _fast_local_values(
                base=fast1000,
                minimum=SPACING1000_MIN,
                maximum=SPACING1000_MAX,
                delta=FAST_LOCAL_1000_DELTA,
            )
        else:
            fast500_values = [fast500]
            fast1000_values = [fast1000]

        best_candidate, coarse_tested, coarse_valid = _search_best_candidate(
            spacing500_values=fast500_values,
            spacing1000_values=fast1000_values,
            merged_border_xy=merged_border_xy,
            zone500_xy=zone500_xy,
            zone1000_xy=zone1000_xy,
            projection=projection,
        )
        if best_candidate is None:
            _fatal(
                "auto-tuning failed: no valid candidate in fast search "
                f"({coarse_tested} tested)"
            )
    else:
        search_mode = "auto"
        coarse500_values = _iter_spacing_values(
            COARSE_SPACING500_MIN,
            COARSE_SPACING500_MAX,
            COARSE_SPACING500_STEP,
        )
        coarse1000_values = _iter_spacing_values(
            COARSE_SPACING1000_MIN,
            COARSE_SPACING1000_MAX,
            COARSE_SPACING1000_STEP,
        )

        coarse_best, coarse_tested, coarse_valid = _search_best_candidate(
            spacing500_values=coarse500_values,
            spacing1000_values=coarse1000_values,
            merged_border_xy=merged_border_xy,
            zone500_xy=zone500_xy,
            zone1000_xy=zone1000_xy,
            projection=projection,
        )
        if coarse_best is None:
            _fatal(
                "auto-tuning failed: no valid candidate in coarse search "
                f"({coarse_tested} tested)"
            )

        fine500_min = max(SPACING500_MIN, coarse_best.spacing500 - FINE_SPACING500_DELTA)
        fine500_max = min(SPACING500_MAX, coarse_best.spacing500 + FINE_SPACING500_DELTA)
        fine1000_min = max(SPACING1000_MIN, coarse_best.spacing1000 - FINE_SPACING1000_DELTA)
        fine1000_max = min(SPACING1000_MAX, coarse_best.spacing1000 + FINE_SPACING1000_DELTA)

        fine500_values = _iter_spacing_values(fine500_min, fine500_max, FINE_SPACING500_STEP)
        fine1000_values = _iter_spacing_values(fine1000_min, fine1000_max, FINE_SPACING1000_STEP)

        fine_best, fine_tested, fine_valid = _search_best_candidate(
            spacing500_values=fine500_values,
            spacing1000_values=fine1000_values,
            merged_border_xy=merged_border_xy,
            zone500_xy=zone500_xy,
            zone1000_xy=zone1000_xy,
            projection=projection,
        )

        best_candidate = coarse_best
        if fine_best is not None and _is_better_candidate(fine_best, best_candidate):
            best_candidate = fine_best

    records = best_candidate.records
    violations = _validate_no_full_swallow(records, projection)
    if violations:
        _fatal(f"no-full-swallow validation failed: {violations} violating 500m centers")

    uncovered_count = _validate_coverage_strict(records, merged_border_xy, projection)
    if uncovered_count:
        _fatal(f"coverage validation failed: {uncovered_count} uncovered strict-grid samples")

    exact_missing_area = _validate_exact_coverage(records, merged_border_xy, projection)
    if exact_missing_area > EXACT_COVERAGE_TOLERANCE_M2 + EPS:
        _fatal(
            "exact coverage validation failed: "
            f"missing area {exact_missing_area:.6f} m^2 exceeds tolerance "
            f"{EXACT_COVERAGE_TOLERANCE_M2:.6f} m^2"
        )

    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    count500 = best_candidate.count500
    count1000 = best_candidate.count1000
    print(
        f"Generated {len(records)} circles | "
        f"mode: {search_mode} | "
        f"500m: {count500} | 1000m: {count1000} | "
        f"spacing500: {best_candidate.spacing500} | "
        f"spacing1000: {best_candidate.spacing1000} | "
        f"sample-auto-added: {best_candidate.sample_auto_added} | "
        f"exact-patch-added: {best_candidate.exact_patch_added} | "
        f"exact-missing-area-m2: {best_candidate.exact_missing_area:.6f} | "
        f"coarse-tested: {coarse_tested} ({coarse_valid} valid) | "
        f"fine-tested: {fine_tested} ({fine_valid} valid) | "
        f"output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
