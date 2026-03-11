#!/usr/bin/env python3
"""
Generate 1000m square-grid collection circles for Chiang Mai.

Pipeline:
1) Load the local Chiang Mai merged border GeoJSON.
2) Project the border into local meter space.
3) Build an axis-aligned 1000m x 1000m square grid over the border bounds.
4) Keep cells whose square polygon intersects the Chiang Mai border.
5) Emit one center record per kept cell to lat_lng_radius.json.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
BORDER_PATH = REPO_ROOT / "chiang_mai_main_area_merged_border.geojson"
OUTPUT_PATH = REPO_ROOT / "lat_lng_radius.json"

GRID_SIZE_M = 500.0
OUTPUT_RADIUS_M = 500.0
EPS = 1e-9


def _fatal(message: str) -> "None":
    """Abort execution with a consistent error format."""
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


@dataclass(frozen=True)
class PolygonData:
    """Projected polygon with optional holes and a cached bounding box."""

    outer: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]]
    bbox: tuple[float, float, float, float]


def _normalize_ring(raw_ring: Any) -> list[tuple[float, float]]:
    """Convert a GeoJSON ring into an open list of numeric coordinate pairs."""
    if not isinstance(raw_ring, list):
        _fatal("polygon ring must be a coordinate array")
    ring: list[tuple[float, float]] = []
    for pair in raw_ring:
        if not (
            isinstance(pair, list)
            and len(pair) >= 2
            and isinstance(pair[0], (int, float))
            and isinstance(pair[1], (int, float))
        ):
            _fatal("polygon ring contains invalid coordinate pairs")
        ring.append((float(pair[0]), float(pair[1])))
    if len(ring) < 4:
        _fatal("polygon ring must contain at least four coordinates")
    if ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) < 3:
        _fatal("polygon ring must contain at least three distinct coordinates")
    return ring


def _bbox_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """Compute axis-aligned bounds for a sequence of points."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _load_border_polygons(path: Path) -> list[PolygonData]:
    """Load and validate polygon coordinates from the local border GeoJSON."""
    if not path.exists():
        _fatal(f"border file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fatal(f"invalid border JSON in {path}: {exc}")

    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        _fatal(f"expected FeatureCollection in {path}")

    features = data.get("features")
    if not isinstance(features, list) or not features:
        _fatal(f"expected at least one feature in {path}")

    polygons: list[PolygonData] = []
    for feature in features:
        if not isinstance(feature, dict):
            _fatal(f"invalid feature structure in {path}")
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            _fatal(f"feature geometry missing in {path}")

        geometry_type = geometry.get("type")
        coordinates = geometry.get("coordinates")
        if geometry_type == "Polygon":
            coordinate_sets = [coordinates]
        elif geometry_type == "MultiPolygon":
            coordinate_sets = coordinates
        else:
            _fatal(f"expected polygon geometry in {path}")

        if not isinstance(coordinate_sets, list) or not coordinate_sets:
            _fatal(f"polygon geometry has no coordinates in {path}")

        for raw_polygon in coordinate_sets:
            if not isinstance(raw_polygon, list) or not raw_polygon:
                _fatal(f"polygon is missing rings in {path}")
            outer = _normalize_ring(raw_polygon[0])
            holes = [_normalize_ring(raw_ring) for raw_ring in raw_polygon[1:]]
            polygons.append(
                PolygonData(
                    outer=outer,
                    holes=holes,
                    bbox=_bbox_from_points(outer),
                )
            )

    if not polygons:
        _fatal("border geometry is empty")
    return polygons


@dataclass(frozen=True)
class LocalProjection:
    """Local equirectangular-like projection centered on the border centroid."""

    lng0: float
    lat0: float
    cos_lat0: float

    def to_xy(self, lng: float, lat: float) -> tuple[float, float]:
        """Convert geographic lon/lat into local meter coordinates."""
        x = (lng - self.lng0) * self.cos_lat0 * 111320.0
        y = (lat - self.lat0) * 111320.0
        return x, y

    def to_lng_lat(self, x: float, y: float) -> tuple[float, float]:
        """Convert local meter coordinates back to geographic lon/lat."""
        lng = (x / (self.cos_lat0 * 111320.0)) + self.lng0
        lat = (y / 111320.0) + self.lat0
        return lng, lat


def _signed_ring_area(ring: list[tuple[float, float]]) -> float:
    """Compute the signed area of an open polygon ring."""
    area = 0.0
    for idx, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(idx + 1) % len(ring)]
        area += (x1 * y2) - (x2 * y1)
    return area / 2.0


def _projection_anchor(polygons: list[PolygonData]) -> tuple[float, float]:
    """Choose a stable projection anchor from polygon outer rings."""
    weighted_x = 0.0
    weighted_y = 0.0
    weight_total = 0.0
    all_points: list[tuple[float, float]] = []

    for polygon in polygons:
        all_points.extend(polygon.outer)
        area = _signed_ring_area(polygon.outer)
        if abs(area) <= EPS:
            continue
        centroid_factor = 0.0
        centroid_x = 0.0
        centroid_y = 0.0
        for idx, (x1, y1) in enumerate(polygon.outer):
            x2, y2 = polygon.outer[(idx + 1) % len(polygon.outer)]
            cross = (x1 * y2) - (x2 * y1)
            centroid_factor += cross
            centroid_x += (x1 + x2) * cross
            centroid_y += (y1 + y2) * cross
        if abs(centroid_factor) <= EPS:
            continue
        centroid_x /= 3.0 * centroid_factor
        centroid_y /= 3.0 * centroid_factor
        weight = abs(area)
        weighted_x += centroid_x * weight
        weighted_y += centroid_y * weight
        weight_total += weight

    if weight_total > EPS:
        return weighted_x / weight_total, weighted_y / weight_total

    min_x, min_y, max_x, max_y = _bbox_from_points(all_points)
    return (min_x + max_x) / 2.0, (min_y + max_y) / 2.0


def _build_projection(border_polygons_lng_lat: list[PolygonData]) -> LocalProjection:
    """Build a local meter-space projection around the border centroid."""
    lng0, lat0 = _projection_anchor(border_polygons_lng_lat)
    return LocalProjection(
        lng0=lng0,
        lat0=lat0,
        cos_lat0=math.cos(math.radians(lat0)),
    )


def _project_polygons(
    polygons_lng_lat: list[PolygonData],
    projection: LocalProjection,
) -> list[PolygonData]:
    """Project polygon rings from lon/lat into local meter coordinates."""
    projected: list[PolygonData] = []
    for polygon in polygons_lng_lat:
        outer = [projection.to_xy(lng, lat) for lng, lat in polygon.outer]
        holes = [
            [projection.to_xy(lng, lat) for lng, lat in hole]
            for hole in polygon.holes
        ]
        projected.append(PolygonData(outer=outer, holes=holes, bbox=_bbox_from_points(outer)))
    return projected


def _point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> bool:
    """Return True when a point lies on a line segment."""
    cross = ((bx - ax) * (py - ay)) - ((by - ay) * (px - ax))
    if abs(cross) > EPS:
        return False
    return (
        min(ax, bx) - EPS <= px <= max(ax, bx) + EPS
        and min(ay, by) - EPS <= py <= max(ay, by) + EPS
    )


def _orientation(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> float:
    """Return the signed triangle area used for segment intersection tests."""
    return ((bx - ax) * (cy - ay)) - ((by - ay) * (cx - ax))


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    """Return True when two closed segments intersect."""
    ax1, ay1 = a1
    ax2, ay2 = a2
    bx1, by1 = b1
    bx2, by2 = b2

    o1 = _orientation(ax1, ay1, ax2, ay2, bx1, by1)
    o2 = _orientation(ax1, ay1, ax2, ay2, bx2, by2)
    o3 = _orientation(bx1, by1, bx2, by2, ax1, ay1)
    o4 = _orientation(bx1, by1, bx2, by2, ax2, ay2)

    if abs(o1) <= EPS and _point_on_segment(bx1, by1, ax1, ay1, ax2, ay2):
        return True
    if abs(o2) <= EPS and _point_on_segment(bx2, by2, ax1, ay1, ax2, ay2):
        return True
    if abs(o3) <= EPS and _point_on_segment(ax1, ay1, bx1, by1, bx2, by2):
        return True
    if abs(o4) <= EPS and _point_on_segment(ax2, ay2, bx1, by1, bx2, by2):
        return True

    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))


def _ring_segments(ring: list[tuple[float, float]]):
    """Iterate closed segments for an open polygon ring."""
    for idx, point1 in enumerate(ring):
        yield point1, ring[(idx + 1) % len(ring)]


def _point_in_ring(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    """Return True when the point is inside or on the boundary of a ring."""
    px, py = point
    inside = False
    for (x1, y1), (x2, y2) in _ring_segments(ring):
        if _point_on_segment(px, py, x1, y1, x2, y2):
            return True
        crosses = ((y1 > py) != (y2 > py))
        if not crosses:
            continue
        x_at_y = ((x2 - x1) * (py - y1) / (y2 - y1)) + x1
        if x_at_y >= px - EPS:
            inside = not inside
    return inside


def _point_in_polygon(point: tuple[float, float], polygon: PolygonData) -> bool:
    """Return True when point lies in the filled polygon area or on its boundary."""
    if not _point_in_ring(point, polygon.outer):
        return False
    for hole in polygon.holes:
        if _point_in_ring(point, hole):
            return False
    return True


def _bbox_intersects(
    bbox1: tuple[float, float, float, float],
    bbox2: tuple[float, float, float, float],
) -> bool:
    """Cheap rectangle overlap test."""
    min_x1, min_y1, max_x1, max_y1 = bbox1
    min_x2, min_y2, max_x2, max_y2 = bbox2
    return not (
        max_x1 < min_x2 - EPS
        or max_x2 < min_x1 - EPS
        or max_y1 < min_y2 - EPS
        or max_y2 < min_y1 - EPS
    )


def _cell_intersects_polygon(
    cell_bbox: tuple[float, float, float, float],
    polygon: PolygonData,
) -> bool:
    """Return True when an axis-aligned cell intersects the polygon geometry."""
    if not _bbox_intersects(cell_bbox, polygon.bbox):
        return False

    min_x, min_y, max_x, max_y = cell_bbox
    cell_corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]
    if any(_point_in_polygon(corner, polygon) for corner in cell_corners):
        return True

    if any(
        min_x - EPS <= vx <= max_x + EPS and min_y - EPS <= vy <= max_y + EPS
        for vx, vy in polygon.outer
    ):
        return True
    for hole in polygon.holes:
        if any(
            min_x - EPS <= vx <= max_x + EPS and min_y - EPS <= vy <= max_y + EPS
            for vx, vy in hole
        ):
            return True

    cell_edges = [
        ((min_x, min_y), (max_x, min_y)),
        ((max_x, min_y), (max_x, max_y)),
        ((max_x, max_y), (min_x, max_y)),
        ((min_x, max_y), (min_x, min_y)),
    ]
    for ring in [polygon.outer, *polygon.holes]:
        for seg_start, seg_end in _ring_segments(ring):
            for edge_start, edge_end in cell_edges:
                if _segments_intersect(seg_start, seg_end, edge_start, edge_end):
                    return True

    return False


def _iter_intersecting_grid_centers(border_xy_polygons: list[PolygonData]):
    """Yield center points for 1000m cells that intersect the border geometry."""
    border_bbox = _bbox_from_points([
        point
        for polygon in border_xy_polygons
        for point in polygon.outer
    ])
    min_x, min_y, max_x, max_y = border_bbox
    start_x = math.floor(min_x / GRID_SIZE_M) * GRID_SIZE_M
    start_y = math.floor(min_y / GRID_SIZE_M) * GRID_SIZE_M
    end_x = math.ceil(max_x / GRID_SIZE_M) * GRID_SIZE_M
    end_y = math.ceil(max_y / GRID_SIZE_M) * GRID_SIZE_M

    row = 0
    y = start_y
    while y < end_y - EPS:
        col = 0
        x = start_x
        while x < end_x - EPS:
            cell_bbox = (x, y, x + GRID_SIZE_M, y + GRID_SIZE_M)
            if any(_cell_intersects_polygon(cell_bbox, polygon) for polygon in border_xy_polygons):
                yield row, col, x + (GRID_SIZE_M / 2.0), y + (GRID_SIZE_M / 2.0)
            x += GRID_SIZE_M
            col += 1
        y += GRID_SIZE_M
        row += 1


def _build_output_rows(border_polygons_lng_lat: list[PolygonData]) -> tuple[list[dict[str, Any]], int]:
    """Create sibling-compatible rows from the intersecting square grid."""
    projection = _build_projection(border_polygons_lng_lat)
    border_xy_polygons = _project_polygons(border_polygons_lng_lat, projection)

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, float]] = set()
    cell_count = 0
    for _row, _col, center_x, center_y in _iter_intersecting_grid_centers(border_xy_polygons):
        lng, lat = projection.to_lng_lat(center_x, center_y)
        key = (round(lat, 6), round(lng, 6))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(
            {
                "lat": key[0],
                "lng": key[1],
                "radius": int(OUTPUT_RADIUS_M),
                "collected": False,
                "population_density": None,
            }
        )
        cell_count += 1

    rows.sort(key=lambda item: (item["lat"], item["lng"]))
    return rows, cell_count


def main() -> int:
    """Program entrypoint for generating lat_lng_radius.json."""
    border_polygons_lng_lat = _load_border_polygons(BORDER_PATH)
    rows, cell_count = _build_output_rows(border_polygons_lng_lat)

    if not rows:
        _fatal("grid generation produced zero intersecting cells")

    OUTPUT_PATH.write_text(
        json.dumps(rows, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Grid size m: {GRID_SIZE_M:.0f} | "
        f"Intersecting cells: {cell_count} | "
        f"Output rows: {len(rows)} | "
        f"Radius m: {OUTPUT_RADIUS_M:.0f} | "
        f"Output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
