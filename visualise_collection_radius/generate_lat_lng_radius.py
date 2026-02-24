#!/usr/bin/env python3
"""
Generate optimized Chiang Mai coverage circles for lat_lng_radius.json.

Dependencies:
    pip install shapely
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, parse, request

try:
    from shapely.geometry import Point, shape
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

RADIUS_500 = 500.0
RADIUS_1000 = 1000.0
SPACING_500 = 600.0
SPACING_1000 = 1200.0
COVERAGE_STEP = 100.0

SHIFT_STEP = 100.0
SHIFT_MAX_DISTANCE = 2000.0
SHIFT_ANGLES = 16
MAX_AUTO_FIX_ADDITIONS = 2000
EPS = 1e-9


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


def _build_zone500(geometries: dict[str, BaseGeometry]) -> BaseGeometry:
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

    return unary_union([city, *selected_parts])


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
    circles: list[tuple[float, float, float]] = []
    for row in records:
        x, y = projection.to_xy(float(row["lng"]), float(row["lat"]))
        circles.append((x, y, float(row["radius"])))

    samples = _build_coverage_samples(merged_border_xy, COVERAGE_STEP)
    uncovered = _find_uncovered_samples(samples, circles)
    return len(uncovered)


def main() -> int:
    merged_border_lng_lat = _load_border_geometry(MERGED_BORDER_PATH)
    relation_geometries = _fetch_relation_geometries()

    zone500_lng_lat = _build_zone500(relation_geometries).intersection(merged_border_lng_lat)
    if zone500_lng_lat.is_empty:
        _fatal("zone500 is empty after clipping with merged border")

    zone1000_lng_lat = merged_border_lng_lat.difference(zone500_lng_lat)
    if zone1000_lng_lat.is_empty:
        _fatal("zone1000 is empty after subtracting zone500 from merged border")

    projection = _build_projection(merged_border_lng_lat)
    merged_border_xy = projection.geom_to_xy(merged_border_lng_lat)
    zone500_xy = projection.geom_to_xy(zone500_lng_lat)
    zone1000_xy = projection.geom_to_xy(zone1000_lng_lat)

    centers500 = _generate_zone_centers(zone500_xy, spacing=SPACING_500, radius=RADIUS_500)
    centers1000_candidates = _generate_zone_centers(
        zone1000_xy,
        spacing=SPACING_1000,
        radius=RADIUS_1000,
    )
    centers1000 = _filter_1000_centers(
        centers1000_candidates,
        zone1000_buffer_xy=zone1000_xy.buffer(RADIUS_1000),
        zone500_xy=zone500_xy,
        centers500=centers500,
    )

    centers500, centers1000, auto_added = _auto_fix_coverage(
        merged_border_xy=merged_border_xy,
        zone500_xy=zone500_xy,
        zone1000_xy=zone1000_xy,
        centers500=centers500,
        centers1000=centers1000,
    )

    records = _records_from_centers(projection, centers500, centers1000)
    violations = _validate_no_full_swallow(records, projection)
    if violations:
        _fatal(f"no-full-swallow validation failed: {violations} violating 500m centers")

    uncovered_count = _validate_coverage_strict(records, merged_border_xy, projection)
    if uncovered_count:
        _fatal(f"coverage validation failed: {uncovered_count} uncovered strict-grid samples")

    OUTPUT_PATH.write_text(json.dumps(records, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    count500 = sum(1 for row in records if int(row["radius"]) == 500)
    count1000 = sum(1 for row in records if int(row["radius"]) == 1000)
    print(
        f"Generated {len(records)} circles | "
        f"500m: {count500} | 1000m: {count1000} | "
        f"auto-added: {auto_added} | output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
