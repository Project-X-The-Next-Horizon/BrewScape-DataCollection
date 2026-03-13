#!/usr/bin/env python3
"""
Generate staged Terascope collection circles for Chiang Mai.

Flow:
1) Cover dominant Built-up cells with only 500 m circles.
2) Cover the remaining area with 1500/2000 m circles.
3) Use square grids with center spacing equal to radius.
4) Enforce stage-specific exact coverage and preserve collected flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from shapely.geometry import GeometryCollection, Point, box, shape
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import polygonize, transform, unary_union
    from shapely.prepared import prep
    from shapely.strtree import STRtree
except ImportError:  # pragma: no cover
    import sys

    print("Error: install shapely with: pip install shapely", file=sys.stderr)
    raise SystemExit(1)


REPO_ROOT = Path(__file__).resolve().parent
INPUT_TERASCOPE_PATH = REPO_ROOT.parent / "terascope" / "chiang_mai_terascope_100m_cells.csv"
BORDER_PATH = REPO_ROOT.parent / "population_density" / "chiang_mai_main_area_merged_border.geojson"
OUTPUT_PATH = REPO_ROOT / "lat_lng_radius.json"
DEFAULT_CALIBRATION_CSV_PATH = (
    REPO_ROOT.parent.parent / "collect_location_data" / "prev_data" / "coffee_shops_with_reviews3.csv"
)

EPS = 1e-9
EXACT_BUFFER_RESOLUTION = 64
TERASCOPE_GRID_SIZE_M = 100.0
OPEN_NON_BUILTUP_LABELS = {"Cropland", "Grassland"}
PRIMARY_SQUARE_PHASES = ((0.0, 0.0), (0.5, 0.5))
REPAIR_SQUARE_PHASES = ((0.5, 0.0), (0.0, 0.5))
DEFAULT_MIN_COVER_COUNT = 2
DEFAULT_SOFT_MAX_COVER_COUNT = 4
DEFAULT_SHOP_SOFT_CAP = 45
DEFAULT_SHOP_HARD_CAP = 60
DEFAULT_BUILD_UP_PATCH_LIMIT = 800
DEFAULT_NON_BUILT_UP_PATCH_LIMIT = 1200
DEFAULT_PRUNE_MAX_PASSES = 0


def _fatal(message: str) -> "None":
    import sys

    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


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


def _to_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _load_border_geometry(path: Path) -> BaseGeometry:
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
    geometry = features[0].get("geometry") if isinstance(features[0], dict) else None
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        _fatal(f"expected polygon geometry in {path}")
    geom = shape(geometry)
    if geom.is_empty:
        _fatal("border geometry is empty")
    return geom


def _radius_rule_for_land_cover(
    land_cover_label: str,
    built_up_share_pct: float,
) -> tuple[str, int]:
    _ = built_up_share_pct
    if land_cover_label == "Built-up":
        return "built_up_dominant", 500
    if land_cover_label in OPEN_NON_BUILTUP_LABELS:
        return "open_non_builtup", 1500
    return "sparse_non_builtup", 2000


def _load_terascope_rows(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        _fatal(f"input CSV not found: {path}")
    rows: list[dict[str, float | int | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "grid_row",
            "grid_col",
            "lon_center",
            "lat_center",
            "dominant_land_cover_label",
            "land_cover_50_share_pct",
        }
        if reader.fieldnames is None:
            _fatal(f"CSV has no header row: {path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            _fatal(f"CSV missing columns: {', '.join(sorted(missing))}")
        for raw in reader:
            grid_row = _to_float(raw.get("grid_row"))
            grid_col = _to_float(raw.get("grid_col"))
            lon = _to_float(raw.get("lon_center"))
            lat = _to_float(raw.get("lat_center"))
            built_up_share_pct = _to_float(raw.get("land_cover_50_share_pct"))
            land_cover_label = raw.get("dominant_land_cover_label")
            if (
                grid_row is None
                or grid_col is None
                or lon is None
                or lat is None
                or built_up_share_pct is None
                or not isinstance(land_cover_label, str)
                or not math.isfinite(lon)
                or not math.isfinite(lat)
                or not math.isfinite(built_up_share_pct)
            ):
                continue
            land_cover_label = land_cover_label.strip() or "No land cover data"
            if (
                built_up_share_pct < 0
                or built_up_share_pct > 100
                or not (-180 <= lon <= 180 and -90 <= lat <= 90)
            ):
                continue
            radius_rule, radius = _radius_rule_for_land_cover(land_cover_label, built_up_share_pct)
            rows.append(
                {
                    "grid_row": int(grid_row),
                    "grid_col": int(grid_col),
                    "lng": lon,
                    "lat": lat,
                    "land_cover_label": land_cover_label,
                    "built_up_share_pct": built_up_share_pct,
                    "radius_rule": radius_rule,
                    "radius": radius,
                }
            )
    if not rows:
        _fatal("no valid rows found in Terascope CSV")
    return rows


def _load_calibration_rows(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        _fatal(f"calibration CSV not found: {path}")
    rows: list[dict[str, float | str]] = []
    seen_place_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"place_id", "lat", "lon"}
        if reader.fieldnames is None:
            _fatal(f"calibration CSV has no header row: {path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            _fatal(f"calibration CSV missing columns: {', '.join(sorted(missing))}")
        for raw in reader:
            place_id = raw.get("place_id")
            lat = _to_float(raw.get("lat"))
            lng = _to_float(raw.get("lon"))
            if not isinstance(place_id, str):
                continue
            place_id = place_id.strip()
            if (
                not place_id
                or place_id in seen_place_ids
                or lat is None
                or lng is None
                or not (-90 <= lat <= 90)
                or not (-180 <= lng <= 180)
            ):
                continue
            seen_place_ids.add(place_id)
            rows.append({"place_id": place_id, "lat": lat, "lng": lng})
    if not rows:
        _fatal(f"calibration CSV has zero usable rows: {path}")
    return rows


@dataclass(frozen=True)
class LocalProjection:
    lng0: float
    lat0: float
    cos_lat0: float

    def to_xy(self, lng: float, lat: float) -> tuple[float, float]:
        return (lng - self.lng0) * self.cos_lat0 * 111320.0, (lat - self.lat0) * 111320.0

    def to_lng_lat(self, x: float, y: float) -> tuple[float, float]:
        return (x / (self.cos_lat0 * 111320.0)) + self.lng0, (y / 111320.0) + self.lat0

    def _forward(self, x: Any, y: Any, z: Any = None):
        if hasattr(x, "__iter__"):
            return (
                [((xi - self.lng0) * self.cos_lat0 * 111320.0) for xi in x],
                [((yi - self.lat0) * 111320.0) for yi in y],
            )
        return (x - self.lng0) * self.cos_lat0 * 111320.0, (y - self.lat0) * 111320.0

    def geom_to_xy(self, geom: BaseGeometry) -> BaseGeometry:
        return transform(self._forward, geom)


@dataclass(frozen=True)
class TerascopeCell:
    grid_row: int
    grid_col: int
    x: float
    y: float
    land_cover_label: str
    built_up_share_pct: float
    radius_rule: str
    radius: int


@dataclass(frozen=True)
class TerascopeGridModel:
    origin_x: float
    origin_y: float
    grid_size_m: float
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    cells_by_key: dict[tuple[int, int], TerascopeCell]
    cells: list[TerascopeCell]


@dataclass
class CirclePlacement:
    x: float
    y: float
    radius: int
    land_cover_label: str
    built_up_share_pct: float
    radius_rule: str
    stage: str
    grid_phase: str
    estimated_shop_load: int = 0


@dataclass
class ExactCoverageState:
    undercovered_geom: BaseGeometry
    undercovered_area: float
    deficit_area: float
    largest_undercovered_part: BaseGeometry | None
    geoms_by_count: dict[int, BaseGeometry]


@dataclass(frozen=True)
class CalibrationShop:
    place_id: str
    x: float
    y: float


@dataclass
class CalibrationModel:
    shops: list[CalibrationShop]
    shop_soft_cap: int
    shop_hard_cap: int
    distance_sq_cache: dict[tuple[float, float], list[float]] = field(default_factory=dict)
    shop_load_cache: dict[tuple[float, float, int], int] = field(default_factory=dict)


@dataclass
class CalibrationCoverageState:
    cover_by_circle: list[set[int]]
    shop_cover_counts: list[int]
    shop_non_overloaded_counts: list[int]
    uncovered_shops: int
    only_overloaded_shops: int
    soft_overloaded_circles: int
    hard_overloaded_circles: int
    max_estimated_shop_load: int
    mean_estimated_shop_load: float
    target_shop_indices: list[int]


@dataclass(frozen=True)
class StageGenerationResult:
    name: str
    circles: list[CirclePlacement]
    seeded_count: int
    patch_added: int
    exact_state: ExactCoverageState


def _build_projection(border_geom_lng_lat: BaseGeometry) -> LocalProjection:
    centroid = border_geom_lng_lat.centroid
    lat0 = float(centroid.y)
    return LocalProjection(
        lng0=float(centroid.x),
        lat0=lat0,
        cos_lat0=math.cos(math.radians(lat0)),
    )


def _build_terascope_model(
    terascope_rows: list[dict[str, float | int | str]],
    projection: LocalProjection,
) -> TerascopeGridModel:
    cells_by_key: dict[tuple[int, int], TerascopeCell] = {}
    origin_x_candidates: list[float] = []
    origin_y_candidates: list[float] = []
    for row in terascope_rows:
        grid_row = int(row["grid_row"])
        grid_col = int(row["grid_col"])
        x, y = projection.to_xy(float(row["lng"]), float(row["lat"]))
        origin_x_candidates.append(x - ((grid_col + 0.5) * TERASCOPE_GRID_SIZE_M))
        origin_y_candidates.append(y - ((grid_row + 0.5) * TERASCOPE_GRID_SIZE_M))
        cells_by_key[(grid_row, grid_col)] = TerascopeCell(
            grid_row=grid_row,
            grid_col=grid_col,
            x=x,
            y=y,
            land_cover_label=str(row["land_cover_label"]),
            built_up_share_pct=float(row["built_up_share_pct"]),
            radius_rule=str(row["radius_rule"]),
            radius=int(row["radius"]),
        )
    if not cells_by_key:
        _fatal("Terascope grid model is empty")
    origin_x = sum(origin_x_candidates) / len(origin_x_candidates)
    origin_y = sum(origin_y_candidates) / len(origin_y_candidates)
    for candidate in origin_x_candidates:
        if abs(candidate - origin_x) > 1e-3:
            _fatal("inconsistent Terascope grid origin on x-axis")
    for candidate in origin_y_candidates:
        if abs(candidate - origin_y) > 1e-3:
            _fatal("inconsistent Terascope grid origin on y-axis")
    rows = [key[0] for key in cells_by_key]
    cols = [key[1] for key in cells_by_key]
    return TerascopeGridModel(
        origin_x=origin_x,
        origin_y=origin_y,
        grid_size_m=TERASCOPE_GRID_SIZE_M,
        min_row=min(rows),
        max_row=max(rows),
        min_col=min(cols),
        max_col=max(cols),
        cells_by_key=cells_by_key,
        cells=list(cells_by_key.values()),
    )


def _build_calibration_model(
    calibration_rows: list[dict[str, float | str]],
    projection: LocalProjection,
    shop_soft_cap: int,
    shop_hard_cap: int,
) -> CalibrationModel:
    shops: list[CalibrationShop] = []
    for row in calibration_rows:
        x, y = projection.to_xy(float(row["lng"]), float(row["lat"]))
        shops.append(CalibrationShop(place_id=str(row["place_id"]), x=x, y=y))
    return CalibrationModel(shops=shops, shop_soft_cap=shop_soft_cap, shop_hard_cap=shop_hard_cap)


def _center_key_xy(x: float, y: float) -> tuple[float, float]:
    return round(x, 3), round(y, 3)


def _union_geometries(parts: list[BaseGeometry]) -> BaseGeometry:
    if not parts:
        return GeometryCollection()
    if len(parts) == 1:
        return parts[0]
    return unary_union(parts)


def _sorted_shop_distances_sq(
    x: float,
    y: float,
    calibration_model: CalibrationModel | None,
) -> list[float]:
    if calibration_model is None or not calibration_model.shops:
        return []
    key = _center_key_xy(x, y)
    cached = calibration_model.distance_sq_cache.get(key)
    if cached is not None:
        return cached
    distances_sq = sorted(
        ((shop.x - x) * (shop.x - x)) + ((shop.y - y) * (shop.y - y))
        for shop in calibration_model.shops
    )
    calibration_model.distance_sq_cache[key] = distances_sq
    return distances_sq


def _estimate_shop_load(
    x: float,
    y: float,
    radius: int,
    calibration_model: CalibrationModel | None,
) -> int:
    if calibration_model is None or not calibration_model.shops:
        return 0
    key = (*_center_key_xy(x, y), int(radius))
    cached = calibration_model.shop_load_cache.get(key)
    if cached is not None:
        return cached
    distances_sq = _sorted_shop_distances_sq(x, y, calibration_model)
    threshold = float(radius * radius) + EPS
    count = bisect_right(distances_sq, threshold)
    calibration_model.shop_load_cache[key] = count
    return count


def _empty_calibration_coverage_state(circle_count: int) -> CalibrationCoverageState:
    return CalibrationCoverageState(
        cover_by_circle=[set() for _ in range(circle_count)],
        shop_cover_counts=[],
        shop_non_overloaded_counts=[],
        uncovered_shops=0,
        only_overloaded_shops=0,
        soft_overloaded_circles=0,
        hard_overloaded_circles=0,
        max_estimated_shop_load=0,
        mean_estimated_shop_load=0.0,
        target_shop_indices=[],
    )


def _build_calibration_coverage_state(
    circles: list[CirclePlacement],
    calibration_model: CalibrationModel | None,
    active: list[bool] | None = None,
) -> CalibrationCoverageState:
    if calibration_model is None or not calibration_model.shops:
        return _empty_calibration_coverage_state(len(circles))
    cover_by_circle = [set() for _ in circles]
    shop_cover_counts = [0] * len(calibration_model.shops)
    shop_non_overloaded_counts = [0] * len(calibration_model.shops)
    active_circle_loads: list[int] = []
    for ci, circle in enumerate(circles):
        if active is not None and not active[ci]:
            continue
        active_circle_loads.append(circle.estimated_shop_load)
        r2 = float(circle.radius * circle.radius)
        is_non_overloaded = circle.estimated_shop_load <= calibration_model.shop_soft_cap
        for si, shop in enumerate(calibration_model.shops):
            dx = shop.x - circle.x
            dy = shop.y - circle.y
            if (dx * dx) + (dy * dy) <= r2 + EPS:
                cover_by_circle[ci].add(si)
                shop_cover_counts[si] += 1
                if is_non_overloaded:
                    shop_non_overloaded_counts[si] += 1
    uncovered_shops = sum(1 for count in shop_cover_counts if count == 0)
    only_overloaded_shops = sum(
        1
        for cover_count, non_overloaded_count in zip(shop_cover_counts, shop_non_overloaded_counts)
        if cover_count > 0 and non_overloaded_count == 0
    )
    target_shop_indices = [
        idx
        for idx, (cover_count, non_overloaded_count) in enumerate(
            zip(shop_cover_counts, shop_non_overloaded_counts)
        )
        if cover_count == 0 or (cover_count > 0 and non_overloaded_count == 0)
    ]
    soft_overloaded_circles = sum(
        1
        for ci, circle in enumerate(circles)
        if (active is None or active[ci]) and circle.estimated_shop_load > calibration_model.shop_soft_cap
    )
    hard_overloaded_circles = sum(
        1
        for ci, circle in enumerate(circles)
        if (active is None or active[ci]) and circle.estimated_shop_load > calibration_model.shop_hard_cap
    )
    max_estimated_shop_load = max(active_circle_loads, default=0)
    mean_estimated_shop_load = (
        sum(active_circle_loads) / len(active_circle_loads) if active_circle_loads else 0.0
    )
    return CalibrationCoverageState(
        cover_by_circle=cover_by_circle,
        shop_cover_counts=shop_cover_counts,
        shop_non_overloaded_counts=shop_non_overloaded_counts,
        uncovered_shops=uncovered_shops,
        only_overloaded_shops=only_overloaded_shops,
        soft_overloaded_circles=soft_overloaded_circles,
        hard_overloaded_circles=hard_overloaded_circles,
        max_estimated_shop_load=max_estimated_shop_load,
        mean_estimated_shop_load=mean_estimated_shop_load,
        target_shop_indices=target_shop_indices,
    )


def _iter_grid_ring_keys(row: int, col: int, ring: int):
    if ring <= 0:
        yield row, col
        return
    for current_col in range(col - ring, col + ring + 1):
        yield row - ring, current_col
        yield row + ring, current_col
    for current_row in range(row - ring + 1, row + ring):
        yield current_row, col - ring
        yield current_row, col + ring


def _query_density(x: float, y: float, density_points: TerascopeGridModel) -> TerascopeCell:
    row = int(math.floor(((y - density_points.origin_y) / density_points.grid_size_m) + EPS))
    col = int(math.floor(((x - density_points.origin_x) / density_points.grid_size_m) + EPS))
    exact = density_points.cells_by_key.get((row, col))
    if exact is not None:
        return exact
    max_ring = max(
        abs(row - density_points.min_row),
        abs(row - density_points.max_row),
        abs(col - density_points.min_col),
        abs(col - density_points.max_col),
    ) + 2
    best_cell: TerascopeCell | None = None
    best_dist_sq = float("inf")
    for ring in range(1, max_ring + 1):
        found_this_ring = False
        for ring_row, ring_col in _iter_grid_ring_keys(row, col, ring):
            cell = density_points.cells_by_key.get((ring_row, ring_col))
            if cell is None:
                continue
            found_this_ring = True
            dx = x - cell.x
            dy = y - cell.y
            dist_sq = (dx * dx) + (dy * dy)
            if (
                best_cell is None
                or dist_sq < best_dist_sq
                or (
                    math.isclose(dist_sq, best_dist_sq)
                    and (cell.grid_row, cell.grid_col) < (best_cell.grid_row, best_cell.grid_col)
                )
            ):
                best_cell = cell
                best_dist_sq = dist_sq
        if found_this_ring and best_cell is not None:
            return best_cell
    if not density_points.cells:
        _fatal("Terascope grid model is empty")
    return min(
        density_points.cells,
        key=lambda cell: (
            ((x - cell.x) * (x - cell.x)) + ((y - cell.y) * (y - cell.y)),
            cell.grid_row,
            cell.grid_col,
        ),
    )


def _cell_polygon(cell: TerascopeCell, clip_to: BaseGeometry | None = None) -> BaseGeometry:
    half = TERASCOPE_GRID_SIZE_M / 2.0
    geom = box(cell.x - half, cell.y - half, cell.x + half, cell.y + half)
    if clip_to is not None:
        geom = geom.intersection(clip_to)
    return geom


def _build_stage_masks(
    density_points: TerascopeGridModel,
    border_xy: BaseGeometry,
) -> tuple[BaseGeometry, BaseGeometry, BaseGeometry, BaseGeometry]:
    built_parts: list[BaseGeometry] = []
    non_built_parts: list[BaseGeometry] = []
    for cell in density_points.cells:
        clipped = _cell_polygon(cell, clip_to=border_xy)
        if clipped.is_empty or clipped.area <= EPS:
            continue
        if cell.land_cover_label == "Built-up":
            built_parts.append(clipped)
        else:
            non_built_parts.append(clipped)
    built_center_mask = _union_geometries(built_parts)
    non_built_center_mask = _union_geometries(non_built_parts)
    built_target_mask = built_center_mask
    non_built_target_mask = border_xy.difference(built_target_mask).intersection(non_built_center_mask)
    return built_target_mask, non_built_target_mask, built_center_mask, non_built_center_mask


def _iter_polygon_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        parts: list[BaseGeometry] = []
        for sub in geom.geoms:
            parts.extend(_iter_polygon_parts(sub))
        return parts
    return []


def _circle_geometries(
    circles: list[CirclePlacement],
    clip_to: BaseGeometry | None = None,
) -> list[BaseGeometry]:
    geometries: list[BaseGeometry] = []
    for circle in circles:
        geom = Point(circle.x, circle.y).buffer(float(circle.radius), quad_segs=EXACT_BUFFER_RESOLUTION)
        if clip_to is not None:
            geom = geom.intersection(clip_to)
        geometries.append(geom)
    return geometries


def _empty_exact_coverage_state(target_mask: BaseGeometry) -> ExactCoverageState:
    if target_mask.is_empty or target_mask.area <= EPS:
        return ExactCoverageState(
            undercovered_geom=GeometryCollection(),
            undercovered_area=0.0,
            deficit_area=0.0,
            largest_undercovered_part=None,
            geoms_by_count={},
        )
    return ExactCoverageState(
        undercovered_geom=target_mask,
        undercovered_area=float(target_mask.area),
        deficit_area=float(target_mask.area),
        largest_undercovered_part=target_mask,
        geoms_by_count={0: target_mask},
    )


def _exact_coverage_state(
    target_mask: BaseGeometry,
    circle_geometries: list[BaseGeometry],
    min_cover_count: int,
) -> ExactCoverageState:
    if target_mask.is_empty or target_mask.area <= EPS:
        return _empty_exact_coverage_state(GeometryCollection())
    if min_cover_count <= 0:
        _fatal("min_cover_count must be positive")
    valid_geometries = [geom for geom in circle_geometries if not geom.is_empty and geom.area > EPS]
    if not valid_geometries:
        return ExactCoverageState(
            undercovered_geom=target_mask,
            undercovered_area=float(target_mask.area),
            deficit_area=float(target_mask.area * min_cover_count),
            largest_undercovered_part=target_mask,
            geoms_by_count={0: target_mask},
        )
    if min_cover_count == 2:
        covered_once_geom = _union_geometries(valid_geometries)
        tree = STRtree(valid_geometries)
        overlap_parts: list[BaseGeometry] = []
        for idx, geom in enumerate(valid_geometries):
            for other_idx in tree.query(geom):
                other_idx = int(other_idx)
                if other_idx <= idx:
                    continue
                overlap = geom.intersection(valid_geometries[other_idx])
                if not overlap.is_empty and overlap.area > EPS:
                    overlap_parts.append(overlap)
        covered_twice_or_more = _union_geometries(overlap_parts)
        count0_geom = target_mask.difference(covered_once_geom)
        count1_geom = covered_once_geom.difference(covered_twice_or_more)
        undercovered_geom = target_mask.difference(covered_twice_or_more)
        undercovered_parts = [part for part in _iter_polygon_parts(undercovered_geom) if part.area > EPS]
        largest_part = max(undercovered_parts, key=lambda part: part.area, default=None)
        return ExactCoverageState(
            undercovered_geom=undercovered_geom,
            undercovered_area=float(undercovered_geom.area),
            deficit_area=float((count0_geom.area * 2.0) + count1_geom.area),
            largest_undercovered_part=largest_part,
            geoms_by_count={0: count0_geom, 1: count1_geom, 2: covered_twice_or_more},
        )
    boundaries: list[BaseGeometry] = [target_mask.boundary]
    for geom in valid_geometries:
        if geom.is_empty or geom.area <= EPS:
            continue
        boundaries.append(geom.boundary)
    partition_lines = unary_union(boundaries)
    target_prepared = prep(target_mask)
    prepared_circles = [prep(geom) for geom in valid_geometries]
    count_parts: dict[int, list[BaseGeometry]] = {}
    undercovered_parts: list[BaseGeometry] = []
    undercovered_area = 0.0
    deficit_area = 0.0
    for face in polygonize(partition_lines):
        if face.is_empty or face.area <= EPS:
            continue
        rep = face.representative_point()
        if not target_prepared.covers(rep):
            continue
        cover_count = 0
        for prepared_circle in prepared_circles:
            if prepared_circle.covers(rep):
                cover_count += 1
        if cover_count <= min_cover_count:
            count_parts.setdefault(cover_count, []).append(face)
        if cover_count < min_cover_count:
            undercovered_parts.append(face)
            undercovered_area += float(face.area)
            deficit_area += float(face.area) * float(min_cover_count - cover_count)
    geoms_by_count = {count: _union_geometries(parts) for count, parts in count_parts.items()}
    largest_part = max(undercovered_parts, key=lambda part: part.area, default=None)
    return ExactCoverageState(
        undercovered_geom=_union_geometries(undercovered_parts),
        undercovered_area=undercovered_area,
        deficit_area=deficit_area,
        largest_undercovered_part=largest_part,
        geoms_by_count=geoms_by_count,
    )


def _stage_cell_matches(cell: TerascopeCell, stage_name: str, radius: int) -> bool:
    if stage_name == "built_up":
        return radius == 500 and cell.land_cover_label == "Built-up" and cell.radius_rule == "built_up_dominant"
    return (
        cell.land_cover_label != "Built-up"
        and cell.radius == radius
        and cell.radius_rule in {"open_non_builtup", "sparse_non_builtup"}
    )


def _grid_base_origin(min_value: float, radius: int) -> float:
    return math.floor(min_value / float(radius)) * float(radius)


def _build_square_grid_origins(
    bounds: tuple[float, float, float, float],
    radius: int,
    phases: tuple[tuple[float, float], ...],
) -> list[tuple[str, float, float]]:
    min_x, min_y, _max_x, _max_y = bounds
    base_x = _grid_base_origin(min_x, radius)
    base_y = _grid_base_origin(min_y, radius)
    origins: list[tuple[str, float, float]] = []
    for phase_x, phase_y in phases:
        origins.append(
            (
                f"{phase_x:.1f},{phase_y:.1f}",
                base_x + (phase_x * float(radius)),
                base_y + (phase_y * float(radius)),
            )
        )
    return origins


def _build_repair_grid_origins(
    bounds: tuple[float, float, float, float],
    radius: int,
    phases: tuple[tuple[float, float], ...],
    target_x: float,
    target_y: float,
) -> list[tuple[str, float, float]]:
    """Build global square grids plus a local target-aligned repair grid."""
    origins = _build_square_grid_origins(bounds, radius, phases)
    origins.append(("target,target", target_x, target_y))
    return origins


def _iter_square_nodes_in_bounds(
    bounds: tuple[float, float, float, float],
    radius: int,
    origin_x: float,
    origin_y: float,
):
    min_x, min_y, max_x, max_y = bounds
    row_min = int(math.floor((min_y - origin_y) / float(radius))) - 1
    row_max = int(math.ceil((max_y - origin_y) / float(radius))) + 1
    col_min = int(math.floor((min_x - origin_x) / float(radius))) - 1
    col_max = int(math.ceil((max_x - origin_x) / float(radius))) + 1
    for row in range(row_min, row_max + 1):
        y = origin_y + (row * float(radius))
        for col in range(col_min, col_max + 1):
            x = origin_x + (col * float(radius))
            yield row, col, x, y


def _iter_square_nodes_near_target(
    radius: int,
    origin_x: float,
    origin_y: float,
    target_x: float,
    target_y: float,
    neighbor_ring: int,
) -> list[tuple[int, int, float, float, float]]:
    row0 = int(round((target_y - origin_y) / float(radius)))
    col0 = int(round((target_x - origin_x) / float(radius)))
    candidates: list[tuple[int, int, float, float, float]] = []
    for row in range(row0 - neighbor_ring, row0 + neighbor_ring + 1):
        y = origin_y + (row * float(radius))
        for col in range(col0 - neighbor_ring, col0 + neighbor_ring + 1):
            x = origin_x + (col * float(radius))
            candidates.append((row, col, x, y, math.hypot(x - target_x, y - target_y)))
    candidates.sort(key=lambda item: item[4])
    return candidates


def _make_circle_placement(
    x: float,
    y: float,
    density_points: TerascopeGridModel,
    calibration_model: CalibrationModel | None,
    stage_name: str,
    grid_phase: str,
) -> CirclePlacement:
    cell = _query_density(x, y, density_points)
    return CirclePlacement(
        x=x,
        y=y,
        radius=cell.radius,
        land_cover_label=cell.land_cover_label,
        built_up_share_pct=cell.built_up_share_pct,
        radius_rule=cell.radius_rule,
        stage=stage_name,
        grid_phase=grid_phase,
        estimated_shop_load=_estimate_shop_load(x, y, cell.radius, calibration_model),
    )


def _seed_stage_circles(
    stage_name: str,
    target_mask: BaseGeometry,
    center_mask: BaseGeometry,
    density_points: TerascopeGridModel,
    calibration_model: CalibrationModel | None,
    radius_order: tuple[int, ...],
    phases: tuple[tuple[float, float], ...],
) -> list[CirclePlacement]:
    if target_mask.is_empty or target_mask.area <= EPS or center_mask.is_empty or center_mask.area <= EPS:
        return []
    circles: list[CirclePlacement] = []
    seen_centers: set[tuple[float, float]] = set()
    center_bounds = center_mask.bounds
    for radius in radius_order:
        for phase_label, origin_x, origin_y in _build_square_grid_origins(center_bounds, radius, phases):
            for _row, _col, x, y in _iter_square_nodes_in_bounds(center_bounds, radius, origin_x, origin_y):
                point = Point(x, y)
                if not center_mask.covers(point):
                    continue
                key = _center_key_xy(x, y)
                if key in seen_centers:
                    continue
                cell = _query_density(x, y, density_points)
                if not _stage_cell_matches(cell, stage_name, radius):
                    continue
                candidate_geom = point.buffer(float(radius), quad_segs=EXACT_BUFFER_RESOLUTION).intersection(
                    target_mask
                )
                if candidate_geom.is_empty or candidate_geom.area <= EPS:
                    continue
                seen_centers.add(key)
                circles.append(
                    _make_circle_placement(
                        x=x,
                        y=y,
                        density_points=density_points,
                        calibration_model=calibration_model,
                        stage_name=stage_name,
                        grid_phase=phase_label,
                    )
                )
    return circles


def _patch_stage_exact_coverage(
    stage_name: str,
    target_mask: BaseGeometry,
    center_mask: BaseGeometry,
    density_points: TerascopeGridModel,
    calibration_model: CalibrationModel | None,
    circles: list[CirclePlacement],
    support_circles: list[CirclePlacement],
    radius_order: tuple[int, ...],
    phases: tuple[tuple[float, float], ...],
    min_cover_count: int,
    exact_tolerance_m2: float,
    exact_patch_limit: int,
    deadline: float,
) -> tuple[int, ExactCoverageState] | None:
    if target_mask.is_empty or target_mask.area <= EPS:
        return 0, _empty_exact_coverage_state(GeometryCollection())
    added = 0
    seen_centers = {_center_key_xy(circle.x, circle.y) for circle in circles}
    center_bounds = center_mask.bounds if not center_mask.is_empty else target_mask.bounds
    while True:
        if time.perf_counter() > deadline:
            return None
        exact_state = _exact_coverage_state(
            target_mask,
            _circle_geometries(support_circles + circles, clip_to=target_mask),
            min_cover_count=min_cover_count,
        )
        if exact_state.undercovered_area <= exact_tolerance_m2 + EPS:
            return added, exact_state
        if added >= exact_patch_limit:
            return None
        part = exact_state.largest_undercovered_part
        if part is None or part.area <= EPS:
            return None
        rep = part.representative_point()
        target_x = float(rep.x)
        target_y = float(rep.y)
        best_circle: CirclePlacement | None = None
        best_key: tuple[float, float] | None = None
        best_score: tuple[float, int, float] | None = None
        for radius in radius_order:
            for phase_label, origin_x, origin_y in _build_repair_grid_origins(
                center_bounds,
                radius,
                phases,
                target_x,
                target_y,
            ):
                for _row, _col, x, y, dist in _iter_square_nodes_near_target(
                    radius=radius,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    target_x=target_x,
                    target_y=target_y,
                    neighbor_ring=3,
                ):
                    point = Point(x, y)
                    key = _center_key_xy(x, y)
                    if key in seen_centers or not center_mask.covers(point):
                        continue
                    cell = _query_density(x, y, density_points)
                    if not _stage_cell_matches(cell, stage_name, radius):
                        continue
                    candidate_geom = point.buffer(
                        float(radius),
                        quad_segs=EXACT_BUFFER_RESOLUTION,
                    ).intersection(target_mask)
                    gain_area = float(candidate_geom.intersection(exact_state.undercovered_geom).area)
                    if gain_area <= EPS:
                        continue
                    score = (-gain_area, radius, dist)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_key = key
                        best_circle = _make_circle_placement(
                            x=x,
                            y=y,
                            density_points=density_points,
                            calibration_model=calibration_model,
                            stage_name=stage_name,
                            grid_phase=phase_label,
                        )
        if best_circle is None or best_key is None:
            search_geom = exact_state.largest_undercovered_part.buffer(float(max(radius_order)))
            for cell in density_points.cells:
                key = _center_key_xy(cell.x, cell.y)
                if key in seen_centers:
                    continue
                if not _stage_cell_matches(cell, stage_name, cell.radius):
                    continue
                point = Point(cell.x, cell.y)
                if not center_mask.covers(point):
                    continue
                if not _cell_polygon(cell).intersects(search_geom):
                    continue
                candidate_geom = point.buffer(
                    float(cell.radius),
                    quad_segs=EXACT_BUFFER_RESOLUTION,
                ).intersection(target_mask)
                gain_area = float(candidate_geom.intersection(exact_state.undercovered_geom).area)
                if gain_area <= EPS:
                    continue
                score = (-gain_area, cell.radius, point.distance(rep))
                if best_score is None or score < best_score:
                    best_score = score
                    best_key = key
                    best_circle = _make_circle_placement(
                        x=cell.x,
                        y=cell.y,
                        density_points=density_points,
                        calibration_model=calibration_model,
                        stage_name=stage_name,
                        grid_phase="cell-center",
                    )
        if best_circle is None or best_key is None:
            return None
        circles.append(best_circle)
        seen_centers.add(best_key)
        added += 1


def _generate_stage(
    stage_name: str,
    target_mask: BaseGeometry,
    center_mask: BaseGeometry,
    density_points: TerascopeGridModel,
    calibration_model: CalibrationModel | None,
    support_circles: list[CirclePlacement],
    radius_order: tuple[int, ...],
    min_cover_count: int,
    exact_tolerance_m2: float,
    exact_patch_limit: int,
    deadline: float,
) -> StageGenerationResult | None:
    if target_mask.is_empty or target_mask.area <= EPS:
        return StageGenerationResult(
            name=stage_name,
            circles=[],
            seeded_count=0,
            patch_added=0,
            exact_state=_empty_exact_coverage_state(GeometryCollection()),
        )
    target_cells = _stage_target_cells(density_points, stage_name)
    circles = _seed_stage_circles(
        stage_name=stage_name,
        target_mask=target_mask,
        center_mask=center_mask,
        density_points=density_points,
        calibration_model=calibration_model,
        radius_order=radius_order,
        phases=PRIMARY_SQUARE_PHASES,
    )
    seeded_count = len(circles)
    patch_added = _patch_stage_cell_coverage(
        stage_name=stage_name,
        target_cells=target_cells,
        target_mask=target_mask,
        center_mask=center_mask,
        density_points=density_points,
        calibration_model=calibration_model,
        circles=circles,
        support_circles=support_circles,
        radius_order=radius_order,
        phases=PRIMARY_SQUARE_PHASES + REPAIR_SQUARE_PHASES,
        min_cover_count=min_cover_count,
        patch_limit=exact_patch_limit,
        deadline=deadline,
    )
    if patch_added is None:
        return None
    exact_state = _exact_coverage_state(
        target_mask,
        _circle_geometries(support_circles + circles, clip_to=target_mask),
        min_cover_count=min_cover_count,
    )
    return StageGenerationResult(
        name=stage_name,
        circles=circles,
        seeded_count=seeded_count,
        patch_added=patch_added,
        exact_state=exact_state,
    )


def _coverage_counts_for_cells(
    cells: list[TerascopeCell],
    circles: list[CirclePlacement],
) -> list[int]:
    counts: list[int] = []
    for cell in cells:
        cover_count = 0
        for circle in circles:
            dx = cell.x - circle.x
            dy = cell.y - circle.y
            if (dx * dx) + (dy * dy) <= float(circle.radius * circle.radius) + EPS:
                cover_count += 1
        counts.append(cover_count)
    return counts


def _stage_target_cells(
    density_points: TerascopeGridModel,
    stage_name: str,
) -> list[TerascopeCell]:
    if stage_name == "built_up":
        return [cell for cell in density_points.cells if cell.land_cover_label == "Built-up"]
    return [cell for cell in density_points.cells if cell.land_cover_label != "Built-up"]


def _patch_stage_cell_coverage(
    stage_name: str,
    target_cells: list[TerascopeCell],
    target_mask: BaseGeometry,
    center_mask: BaseGeometry,
    density_points: TerascopeGridModel,
    calibration_model: CalibrationModel | None,
    circles: list[CirclePlacement],
    support_circles: list[CirclePlacement],
    radius_order: tuple[int, ...],
    phases: tuple[tuple[float, float], ...],
    min_cover_count: int,
    patch_limit: int,
    deadline: float,
) -> int | None:
    added = 0
    seen_centers = {_center_key_xy(circle.x, circle.y) for circle in circles}
    center_bounds = center_mask.bounds if not center_mask.is_empty else target_mask.bounds
    while True:
        if time.perf_counter() > deadline:
            return None
        all_circles = support_circles + circles
        counts = _coverage_counts_for_cells(target_cells, all_circles)
        undercovered = [
            (cell, count)
            for cell, count in zip(target_cells, counts)
            if count < min_cover_count
        ]
        if not undercovered:
            return added
        if added >= patch_limit:
            return None
        undercovered.sort(key=lambda item: (item[1], item[0].grid_row, item[0].grid_col))
        target_cell, _count = undercovered[0]
        best_circle: CirclePlacement | None = None
        best_key: tuple[float, float] | None = None
        best_score: tuple[int, int, float] | None = None

        for radius in radius_order:
            for phase_label, origin_x, origin_y in _build_repair_grid_origins(
                center_bounds,
                radius,
                phases,
                target_cell.x,
                target_cell.y,
            ):
                for _row, _col, x, y, dist in _iter_square_nodes_near_target(
                    radius=radius,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    target_x=target_cell.x,
                    target_y=target_cell.y,
                    neighbor_ring=3,
                ):
                    point = Point(x, y)
                    key = _center_key_xy(x, y)
                    if key in seen_centers or not center_mask.covers(point):
                        continue
                    cell = _query_density(x, y, density_points)
                    if not _stage_cell_matches(cell, stage_name, radius):
                        continue
                    if point.buffer(float(radius), quad_segs=EXACT_BUFFER_RESOLUTION).intersection(target_mask).is_empty:
                        continue
                    gain = 0
                    for candidate_cell, candidate_count in undercovered:
                        dx = candidate_cell.x - x
                        dy = candidate_cell.y - y
                        if (dx * dx) + (dy * dy) <= float(radius * radius) + EPS:
                            gain += min_cover_count - candidate_count
                    if gain <= 0:
                        continue
                    score = (-gain, radius, dist)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_key = key
                        best_circle = _make_circle_placement(
                            x=x,
                            y=y,
                            density_points=density_points,
                            calibration_model=calibration_model,
                            stage_name=stage_name,
                            grid_phase=phase_label,
                        )

        if best_circle is None or best_key is None:
            for cell, candidate_count in undercovered:
                offset = TERASCOPE_GRID_SIZE_M / 4.0
                candidate_positions = (
                    (cell.x, cell.y, "cell-center"),
                    (cell.x - offset, cell.y - offset, "cell-q1"),
                    (cell.x - offset, cell.y + offset, "cell-q2"),
                    (cell.x + offset, cell.y - offset, "cell-q3"),
                    (cell.x + offset, cell.y + offset, "cell-q4"),
                )
                for px, py, phase_label in candidate_positions:
                    key = _center_key_xy(px, py)
                    if key in seen_centers:
                        continue
                    point = Point(px, py)
                    if not center_mask.covers(point):
                        continue
                    if point.buffer(float(cell.radius), quad_segs=EXACT_BUFFER_RESOLUTION).intersection(target_mask).is_empty:
                        continue
                    gain = 0
                    for other_cell, other_count in undercovered:
                        dx = other_cell.x - px
                        dy = other_cell.y - py
                        if (dx * dx) + (dy * dy) <= float(cell.radius * cell.radius) + EPS:
                            gain += min_cover_count - other_count
                    if gain <= 0:
                        continue
                    score = (-gain, cell.radius, 0.0)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_key = key
                        best_circle = _make_circle_placement(
                            x=px,
                            y=py,
                            density_points=density_points,
                            calibration_model=calibration_model,
                            stage_name=stage_name,
                            grid_phase=phase_label,
                        )
        if best_circle is None or best_key is None:
            return None
        circles.append(best_circle)
        seen_centers.add(best_key)
        added += 1


def _coverage_metrics(
    coverage_counts: list[int],
    min_cover_count: int,
    soft_max_cover_count: int,
) -> tuple[int, int, int]:
    if not coverage_counts:
        return 0, 0, 0
    return (
        min(coverage_counts),
        sum(1 for count in coverage_counts if count < min_cover_count),
        sum(1 for count in coverage_counts if count > soft_max_cover_count),
    )


def _prune_circles(
    circles: list[CirclePlacement],
    built_target_mask: BaseGeometry,
    non_built_target_mask: BaseGeometry,
    calibration_model: CalibrationModel | None,
    min_cover_count: int,
    exact_tolerance_m2: float,
    prune_max_passes: int,
    deadline: float,
) -> tuple[list[CirclePlacement], int, ExactCoverageState, ExactCoverageState, CalibrationCoverageState] | None:
    if not circles:
        calibration_state = _build_calibration_coverage_state(circles, calibration_model)
        return (
            [],
            0,
            _empty_exact_coverage_state(GeometryCollection()),
            _empty_exact_coverage_state(GeometryCollection()),
            calibration_state,
        )
    active = [True] * len(circles)
    removed_total = 0
    baseline_calibration = _build_calibration_coverage_state(circles, calibration_model, active)
    for _pass_idx in range(prune_max_passes):
        removed_this_pass = 0
        order = sorted(
            range(len(circles)),
            key=lambda idx: (
                -circles[idx].radius,
                circles[idx].estimated_shop_load,
                circles[idx].stage,
                circles[idx].x,
                circles[idx].y,
            ),
        )
        for idx in order:
            if time.perf_counter() > deadline:
                return None
            if not active[idx]:
                continue
            active[idx] = False
            built_geoms = _circle_geometries(
                [circle for ci, circle in enumerate(circles) if active[ci] and circle.stage == "built_up"],
                clip_to=built_target_mask,
            )
            non_built_geoms = _circle_geometries(
                [circle for ci, circle in enumerate(circles) if active[ci]],
                clip_to=non_built_target_mask,
            )
            built_state = _exact_coverage_state(built_target_mask, built_geoms, min_cover_count)
            non_built_state = _exact_coverage_state(non_built_target_mask, non_built_geoms, min_cover_count)
            if (
                built_state.undercovered_area > exact_tolerance_m2 + EPS
                or non_built_state.undercovered_area > exact_tolerance_m2 + EPS
            ):
                active[idx] = True
                continue
            calibration_state = _build_calibration_coverage_state(circles, calibration_model, active)
            if (
                calibration_state.uncovered_shops > baseline_calibration.uncovered_shops
                or calibration_state.only_overloaded_shops > baseline_calibration.only_overloaded_shops
            ):
                active[idx] = True
                continue
            removed_total += 1
            removed_this_pass += 1
        if removed_this_pass == 0:
            break
    pruned = [circle for idx, circle in enumerate(circles) if active[idx]]
    built_state = _exact_coverage_state(
        built_target_mask,
        _circle_geometries([circle for circle in pruned if circle.stage == "built_up"], clip_to=built_target_mask),
        min_cover_count,
    )
    non_built_state = _exact_coverage_state(
        non_built_target_mask,
        _circle_geometries(pruned, clip_to=non_built_target_mask),
        min_cover_count,
    )
    calibration_state = _build_calibration_coverage_state(pruned, calibration_model)
    return pruned, removed_total, built_state, non_built_state, calibration_state


def _circles_to_rows(circles: list[CirclePlacement], projection: LocalProjection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for circle in circles:
        lng, lat = projection.to_lng_lat(circle.x, circle.y)
        rows.append(
            {
                "lat": round(lat, 6),
                "lng": round(lng, 6),
                "radius": int(circle.radius),
                "collected": False,
                "land_cover_label": circle.land_cover_label,
                "built_up_share_pct": round(circle.built_up_share_pct, 6),
                "radius_rule": circle.radius_rule,
            }
        )
    rows.sort(key=lambda item: (item["lat"], item["lng"], item["radius"]))
    return rows


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _apply_collected_preservation(
    new_rows: list[dict[str, Any]],
    old_rows: list[dict[str, Any]],
    projection: LocalProjection,
    preserve_distance: float,
) -> tuple[int, int]:
    old_index: dict[tuple[float, float], bool] = {}
    old_unique: list[tuple[float, float, bool]] = []
    for row in old_rows:
        lat = _to_float(row.get("lat"))
        lng = _to_float(row.get("lng"))
        collected = _to_bool(row.get("collected"))
        if lat is None or lng is None or collected is None:
            continue
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            continue
        key = (round(lat, 6), round(lng, 6))
        if key in old_index:
            old_index[key] = old_index[key] or collected
            continue
        old_index[key] = collected
        old_unique.append((float(lat), float(lng), collected))
    exact_matches = 0
    matched_new: set[int] = set()
    matched_old: set[int] = set()
    for idx, row in enumerate(new_rows):
        key = (round(float(row["lat"]), 6), round(float(row["lng"]), 6))
        if key not in old_index:
            continue
        row["collected"] = bool(old_index[key])
        exact_matches += 1
        matched_new.add(idx)
        for old_idx, (olat, olng, _ocol) in enumerate(old_unique):
            if (round(olat, 6), round(olng, 6)) == key:
                matched_old.add(old_idx)
    if preserve_distance <= 0:
        return exact_matches, 0
    new_xy: dict[int, tuple[float, float]] = {}
    for idx, row in enumerate(new_rows):
        if idx in matched_new:
            continue
        new_xy[idx] = projection.to_xy(float(row["lng"]), float(row["lat"]))
    old_xy: dict[int, tuple[float, float, bool]] = {}
    for old_idx, (lat, lng, collected) in enumerate(old_unique):
        if old_idx in matched_old:
            continue
        x, y = projection.to_xy(lng, lat)
        old_xy[old_idx] = (x, y, collected)
    limit_sq = preserve_distance * preserve_distance
    pairs: list[tuple[float, int, int]] = []
    for new_idx, (nx, ny) in new_xy.items():
        for old_idx, (ox, oy, _collected) in old_xy.items():
            dx = nx - ox
            dy = ny - oy
            dist_sq = (dx * dx) + (dy * dy)
            if dist_sq <= limit_sq + EPS:
                pairs.append((dist_sq, new_idx, old_idx))
    pairs.sort(key=lambda item: item[0])
    used_new: set[int] = set()
    used_old: set[int] = set()
    nearest_matches = 0
    for _d2, new_idx, old_idx in pairs:
        if new_idx in used_new or old_idx in used_old:
            continue
        used_new.add(new_idx)
        used_old.add(old_idx)
        new_rows[new_idx]["collected"] = bool(old_xy[old_idx][2])
        nearest_matches += 1
    return exact_matches, nearest_matches


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT.parent.parent / path).resolve()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate staged square-grid Terascope circles for Chiang Mai."
    )
    parser.add_argument("--calibration-csv", type=str, default=str(DEFAULT_CALIBRATION_CSV_PATH))
    parser.add_argument("--disable-calibration", action="store_true")
    parser.add_argument("--shop-soft-cap", type=int, default=DEFAULT_SHOP_SOFT_CAP)
    parser.add_argument("--shop-hard-cap", type=int, default=DEFAULT_SHOP_HARD_CAP)
    parser.add_argument("--preserve-distance", type=float, default=300.0)
    parser.add_argument("--opt-max-seconds", type=float, default=600.0)
    parser.add_argument("--exact-tolerance-m2", type=float, default=1.0)
    parser.add_argument("--min-cover-count", type=int, default=DEFAULT_MIN_COVER_COUNT)
    parser.add_argument("--soft-max-cover-count", type=int, default=DEFAULT_SOFT_MAX_COVER_COUNT)
    parser.add_argument("--built-up-patch-limit", type=int, default=DEFAULT_BUILD_UP_PATCH_LIMIT)
    parser.add_argument("--non-built-up-patch-limit", type=int, default=DEFAULT_NON_BUILT_UP_PATCH_LIMIT)
    parser.add_argument("--prune-max-passes", type=int, default=DEFAULT_PRUNE_MAX_PASSES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.shop_soft_cap <= 0:
        _fatal("--shop-soft-cap must be > 0")
    if args.shop_hard_cap < args.shop_soft_cap:
        _fatal("--shop-hard-cap must be >= --shop-soft-cap")
    if args.preserve_distance < 0:
        _fatal("--preserve-distance must be >= 0")
    if args.opt_max_seconds <= 0:
        _fatal("--opt-max-seconds must be > 0")
    if args.exact_tolerance_m2 < 0:
        _fatal("--exact-tolerance-m2 must be >= 0")
    if args.min_cover_count <= 0:
        _fatal("--min-cover-count must be > 0")
    if args.soft_max_cover_count < args.min_cover_count:
        _fatal("--soft-max-cover-count must be >= --min-cover-count")
    if args.built_up_patch_limit <= 0:
        _fatal("--built-up-patch-limit must be > 0")
    if args.non_built_up_patch_limit <= 0:
        _fatal("--non-built-up-patch-limit must be > 0")
    if args.prune_max_passes < 0:
        _fatal("--prune-max-passes must be >= 0")

    border_lng_lat = _load_border_geometry(BORDER_PATH)
    terascope_rows = _load_terascope_rows(INPUT_TERASCOPE_PATH)
    projection = _build_projection(border_lng_lat)
    border_xy = projection.geom_to_xy(border_lng_lat)
    density_points = _build_terascope_model(terascope_rows, projection)
    built_target_mask, non_built_target_mask, built_center_mask, non_built_center_mask = _build_stage_masks(
        density_points,
        border_xy,
    )

    calibration_model: CalibrationModel | None = None
    if not args.disable_calibration:
        calibration_rows = _load_calibration_rows(_resolve_path(args.calibration_csv))
        calibration_model = _build_calibration_model(
            calibration_rows=calibration_rows,
            projection=projection,
            shop_soft_cap=args.shop_soft_cap,
            shop_hard_cap=args.shop_hard_cap,
        )

    start = time.perf_counter()
    deadline = start + args.opt_max_seconds
    built_result = _generate_stage(
        stage_name="built_up",
        target_mask=built_target_mask,
        center_mask=built_center_mask,
        density_points=density_points,
        calibration_model=calibration_model,
        support_circles=[],
        radius_order=(500,),
        min_cover_count=args.min_cover_count,
        exact_tolerance_m2=args.exact_tolerance_m2,
        exact_patch_limit=args.built_up_patch_limit,
        deadline=deadline,
    )
    if built_result is None:
        _fatal("failed to satisfy exact built-up coverage with 500 m circles")
    non_built_result = _generate_stage(
        stage_name="non_built_up",
        target_mask=non_built_target_mask,
        center_mask=non_built_center_mask,
        density_points=density_points,
        calibration_model=calibration_model,
        support_circles=built_result.circles,
        radius_order=(1500, 2000),
        min_cover_count=args.min_cover_count,
        exact_tolerance_m2=args.exact_tolerance_m2,
        exact_patch_limit=args.non_built_up_patch_limit,
        deadline=deadline,
    )
    if non_built_result is None:
        _fatal("failed to satisfy exact non-built-up coverage with 1500/2000 m circles")

    pruned = _prune_circles(
        circles=built_result.circles + non_built_result.circles,
        built_target_mask=built_target_mask,
        non_built_target_mask=non_built_target_mask,
        calibration_model=calibration_model,
        min_cover_count=args.min_cover_count,
        exact_tolerance_m2=args.exact_tolerance_m2,
        prune_max_passes=args.prune_max_passes,
        deadline=deadline,
    )
    if pruned is None:
        _fatal("circle pruning exceeded the configured runtime budget")
    final_circles, prune_removed, built_exact_state, non_built_exact_state, calibration_state = pruned

    output_rows = _circles_to_rows(final_circles, projection)
    old_rows = _load_existing_rows(OUTPUT_PATH)
    exact_preserved, nearest_preserved = _apply_collected_preservation(
        new_rows=output_rows,
        old_rows=old_rows,
        projection=projection,
        preserve_distance=args.preserve_distance,
    )
    OUTPUT_PATH.write_text(json.dumps(output_rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    built_cells = [cell for cell in density_points.cells if cell.land_cover_label == "Built-up"]
    non_built_cells = [cell for cell in density_points.cells if cell.land_cover_label != "Built-up"]
    built_counts = _coverage_counts_for_cells(
        built_cells,
        [circle for circle in final_circles if circle.stage == "built_up"],
    )
    non_built_counts = _coverage_counts_for_cells(
        non_built_cells,
        final_circles,
    )
    built_min_cover, built_below_min, built_above_soft = _coverage_metrics(
        built_counts,
        min_cover_count=args.min_cover_count,
        soft_max_cover_count=args.soft_max_cover_count,
    )
    non_built_min_cover, non_built_below_min, non_built_above_soft = _coverage_metrics(
        non_built_counts,
        min_cover_count=args.min_cover_count,
        soft_max_cover_count=args.soft_max_cover_count,
    )
    radii = sorted(int(row["radius"]) for row in output_rows)
    calibration_summary = (
        "Calibration: disabled"
        if calibration_model is None
        else (
            f"Calibration shops covered: "
            f"{len(calibration_model.shops) - calibration_state.uncovered_shops}/{len(calibration_model.shops)} | "
            f"Only overloaded cover: {calibration_state.only_overloaded_shops} | "
            f"Circles > soft cap: {calibration_state.soft_overloaded_circles} | "
            f"Circles > hard cap: {calibration_state.hard_overloaded_circles} | "
            f"Max known shops in circle: {calibration_state.max_estimated_shop_load} | "
            f"Mean known shops in circle: {calibration_state.mean_estimated_shop_load:.2f}"
        )
    )
    total_seconds = time.perf_counter() - start
    print(
        f"Built-up circles: {sum(1 for circle in final_circles if circle.stage == 'built_up')} | "
        f"Built-up seeded: {built_result.seeded_count} | "
        f"Built-up patch added: {built_result.patch_added} | "
        f"Built-up exact undercovered area m2: {built_exact_state.undercovered_area:.6f} | "
        f"Built-up min cell cover: {built_min_cover} | "
        f"Built-up cells below min: {built_below_min} | "
        f"Built-up cells above soft max: {built_above_soft} | "
        f"Non-built-up circles: {sum(1 for circle in final_circles if circle.stage == 'non_built_up')} | "
        f"Non-built-up seeded: {non_built_result.seeded_count} | "
        f"Non-built-up patch added: {non_built_result.patch_added} | "
        f"Non-built-up exact undercovered area m2: {non_built_exact_state.undercovered_area:.6f} | "
        f"Non-built-up min cell cover: {non_built_min_cover} | "
        f"Non-built-up cells below min: {non_built_below_min} | "
        f"Non-built-up cells above soft max: {non_built_above_soft} | "
        f"Pruned removed: {prune_removed} | "
        f"Unique radii: {sorted(set(radii))} | "
        f"Min cover target: {args.min_cover_count} | "
        f"Soft max target: {args.soft_max_cover_count} | "
        f"{calibration_summary} | "
        f"Preserved collected exact: {exact_preserved} | "
        f"Preserved collected nearest: {nearest_preserved} | "
        f"Total sec: {total_seconds:.2f} | "
        f"Output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
