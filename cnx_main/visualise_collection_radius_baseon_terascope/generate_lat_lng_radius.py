#!/usr/bin/env python3
"""
Generate a merged collection-radius layer for Chiang Mai.

Pipeline:
1) Load the local Chiang Mai border GeoJSON for projection anchoring.
2) Load the existing 500 m built-up circles from lat_lng_radius.json as frozen support.
3) Read Chiang Mai Terascope 100 m cells and split them into built / non-built masks.
4) Compute the exact remaining non-built deficits after the frozen 500 m layer.
5) Build adaptive 1000/1500/2000 m candidate circles over the non-built area.
6) Greedily add adaptive circles until the non-built mask has exact 2-circle coverage.
7) Prune redundant adaptive circles without changing the frozen 500 m layer.
8) Write the frozen 500 m rows first, then append the adaptive rows.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, Point, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree


REPO_ROOT = Path(__file__).resolve().parent
BORDER_PATH = REPO_ROOT / "chiang_mai_main_area_merged_border.geojson"
TERASCOPE_CSV_PATH = REPO_ROOT.parent / "terascope" / "chiang_mai_terascope_100m_cells.csv"
OUTPUT_PATH = REPO_ROOT / "lat_lng_radius.json"

TARGET_LAND_COVER_LABEL = "Built-up"
FROZEN_RADIUS_M = 500
ADAPTIVE_RADII_M = (2000, 1500, 1000)
GRID_PHASES = ((0.0, 0.0), (0.5, 0.5), (0.5, 0.0), (0.0, 0.5))
MIN_COVER_COUNT = 2
EXACT_TOLERANCE_M2 = 1.0
BUFFER_QUAD_SEGS = 64
METERS_PER_DEGREE = 111320.0
EPS = 1e-9


def _fatal(message: str) -> "None":
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


def _geom_or_empty(geom: BaseGeometry | None) -> BaseGeometry:
    return GeometryCollection() if geom is None else geom


def _non_empty_geometries(geometries: list[BaseGeometry]) -> list[BaseGeometry]:
    return [geom for geom in geometries if not geom.is_empty and geom.area > EPS]


@dataclass(frozen=True)
class LocalProjection:
    lng0: float
    lat0: float
    cos_lat0: float

    def to_xy(
        self,
        lng: float | np.ndarray,
        lat: float | np.ndarray,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        x = (lng - self.lng0) * self.cos_lat0 * METERS_PER_DEGREE
        y = (lat - self.lat0) * METERS_PER_DEGREE
        return x, y

    def to_lng_lat(self, x: float, y: float) -> tuple[float, float]:
        lng = (x / (self.cos_lat0 * METERS_PER_DEGREE)) + self.lng0
        lat = (y / METERS_PER_DEGREE) + self.lat0
        return lng, lat


@dataclass(frozen=True)
class GridTargets:
    built_target_mask: BaseGeometry
    non_built_target_mask: BaseGeometry
    built_center_mask: BaseGeometry
    non_built_center_mask: BaseGeometry
    built_center_x: np.ndarray
    built_center_y: np.ndarray
    built_count: int
    non_built_count: int


@dataclass(frozen=True)
class CirclePlacement:
    x: float
    y: float
    radius: int


@dataclass(frozen=True)
class CandidateCircle:
    x: float
    y: float
    radius: int
    phase_label: str
    full_geom: BaseGeometry
    non_built_geom: BaseGeometry
    need_one_geom: BaseGeometry
    need_two_geom: BaseGeometry
    built_overlap_area: float


@dataclass(frozen=True)
class ExactCoverageState:
    undercovered_geom: BaseGeometry
    undercovered_area: float
    deficit_area: float
    geoms_by_count: dict[int, BaseGeometry]


@dataclass(frozen=True)
class AdaptiveCoverageState:
    need_one_state: ExactCoverageState
    need_two_state: ExactCoverageState
    total_undercovered_area: float
    total_deficit_area: float


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

    polygon_parts = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        geom = shape(geometry)
        if not geom.is_empty:
            polygon_parts.append(geom)

    if not polygon_parts:
        _fatal(f"expected polygon geometry in {path}")

    border_geom = unary_union(polygon_parts)
    if border_geom.is_empty:
        _fatal("border geometry is empty")
    return border_geom


def _build_projection(border_geom: BaseGeometry) -> LocalProjection:
    centroid = border_geom.centroid
    lat0 = float(centroid.y)
    return LocalProjection(
        lng0=float(centroid.x),
        lat0=lat0,
        cos_lat0=math.cos(math.radians(lat0)),
    )


def _union_geometries(parts: list[BaseGeometry]) -> BaseGeometry:
    if not parts:
        return GeometryCollection()
    if len(parts) == 1:
        return parts[0]
    return unary_union(parts)


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        _fatal(f"existing radius JSON not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fatal(f"invalid JSON in {path}: {exc}")

    if not isinstance(data, list):
        _fatal(f"expected a JSON array in {path}")

    return [row for row in data if isinstance(row, dict)]


def _extract_frozen_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frozen_rows: list[dict[str, Any]] = []
    for row in rows:
        radius = row.get("radius")
        if radius != FROZEN_RADIUS_M:
            continue

        lat = _to_float(row.get("lat"))
        lng = _to_float(row.get("lng"))
        collected = row.get("collected")
        if lat is None or lng is None or not isinstance(collected, bool):
            _fatal("existing 500 m rows must contain valid lat/lng and collected values")

        frozen_rows.append(row)

    if not frozen_rows:
        _fatal("found zero frozen 500 m circles in the existing lat_lng_radius.json")

    return frozen_rows


def _rows_to_circle_placements(
    rows: list[dict[str, Any]],
    projection: LocalProjection,
) -> list[CirclePlacement]:
    circles: list[CirclePlacement] = []
    for row in rows:
        lat = _to_float(row.get("lat"))
        lng = _to_float(row.get("lng"))
        radius = _to_float(row.get("radius"))
        if lat is None or lng is None or radius is None:
            _fatal("circle rows must contain numeric lat/lng/radius values")

        x, y = projection.to_xy(lng, lat)
        circles.append(CirclePlacement(x=float(x), y=float(y), radius=int(radius)))
    return circles


def _build_circle_geometry(x: float, y: float, radius: int) -> BaseGeometry:
    return Point(x, y).buffer(float(radius), quad_segs=BUFFER_QUAD_SEGS)


def _circle_geometries(
    circles: list[CirclePlacement],
    clip_to: BaseGeometry | None = None,
) -> list[BaseGeometry]:
    geometries: list[BaseGeometry] = []
    for circle in circles:
        geom = _build_circle_geometry(circle.x, circle.y, circle.radius)
        if clip_to is not None:
            geom = geom.intersection(clip_to)
        if not geom.is_empty and geom.area > EPS:
            geometries.append(geom)
    return geometries


def _load_target_masks(path: Path, projection: LocalProjection) -> GridTargets:
    if not path.exists():
        _fatal(f"Terascope CSV not found: {path}")

    required_columns = {
        "lon_center",
        "lat_center",
        "lon_left",
        "lon_right",
        "lat_bottom",
        "lat_top",
        "dominant_land_cover_label",
    }

    built_parts: list[BaseGeometry] = []
    non_built_parts: list[BaseGeometry] = []
    built_center_x_values: list[float] = []
    built_center_y_values: list[float] = []
    built_count = 0
    non_built_count = 0

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            _fatal(f"CSV has no header row: {path}")

        missing = required_columns.difference(reader.fieldnames)
        if missing:
            _fatal(f"CSV missing columns: {', '.join(sorted(missing))}")

        for raw in reader:
            lon_center = _to_float(raw.get("lon_center"))
            lat_center = _to_float(raw.get("lat_center"))
            lon_left = _to_float(raw.get("lon_left"))
            lon_right = _to_float(raw.get("lon_right"))
            lat_bottom = _to_float(raw.get("lat_bottom"))
            lat_top = _to_float(raw.get("lat_top"))
            label = raw.get("dominant_land_cover_label")

            if (
                lon_center is None
                or lat_center is None
                or lon_left is None
                or lon_right is None
                or lat_bottom is None
                or lat_top is None
                or not isinstance(label, str)
            ):
                continue

            center_x, center_y = projection.to_xy(lon_center, lat_center)
            left_bottom_x, left_bottom_y = projection.to_xy(lon_left, lat_bottom)
            right_top_x, right_top_y = projection.to_xy(lon_right, lat_top)

            cell_geom = box(
                min(left_bottom_x, right_top_x),
                min(left_bottom_y, right_top_y),
                max(left_bottom_x, right_top_x),
                max(left_bottom_y, right_top_y),
            )

            if label == TARGET_LAND_COVER_LABEL:
                built_parts.append(cell_geom)
                built_center_x_values.append(float(center_x))
                built_center_y_values.append(float(center_y))
                built_count += 1
            else:
                non_built_parts.append(cell_geom)
                non_built_count += 1

    if built_count == 0:
        _fatal("found zero dominant built-up Terascope cells")
    if non_built_count == 0:
        _fatal("found zero non-built Terascope cells")

    built_mask = _union_geometries(built_parts)
    non_built_mask = _union_geometries(non_built_parts)
    if built_mask.is_empty or non_built_mask.is_empty:
        _fatal("target masks are empty after processing the Terascope CSV")

    return GridTargets(
        built_target_mask=built_mask,
        non_built_target_mask=non_built_mask,
        built_center_mask=built_mask,
        non_built_center_mask=non_built_mask,
        built_center_x=np.asarray(built_center_x_values, dtype=np.float64),
        built_center_y=np.asarray(built_center_y_values, dtype=np.float64),
        built_count=built_count,
        non_built_count=non_built_count,
    )


def _empty_exact_coverage_state() -> ExactCoverageState:
    return ExactCoverageState(
        undercovered_geom=GeometryCollection(),
        undercovered_area=0.0,
        deficit_area=0.0,
        geoms_by_count={},
    )


def _exact_coverage_state(
    target_mask: BaseGeometry,
    circle_geometries: list[BaseGeometry],
    min_cover_count: int,
) -> ExactCoverageState:
    if target_mask.is_empty or target_mask.area <= EPS:
        return _empty_exact_coverage_state()

    if min_cover_count not in {1, 2}:
        _fatal("this generator only supports exact coverage checks for min_cover_count in {1, 2}")

    valid_geometries = _non_empty_geometries(circle_geometries)
    if not valid_geometries:
        return ExactCoverageState(
            undercovered_geom=target_mask,
            undercovered_area=float(target_mask.area),
            deficit_area=float(target_mask.area * float(min_cover_count)),
            geoms_by_count={0: target_mask},
        )

    if min_cover_count == 1:
        covered_once_geom = _union_geometries(valid_geometries)
        count0_geom = target_mask.difference(covered_once_geom)
        count1_geom = target_mask.difference(count0_geom)
        return ExactCoverageState(
            undercovered_geom=count0_geom,
            undercovered_area=float(count0_geom.area),
            deficit_area=float(count0_geom.area),
            geoms_by_count={0: count0_geom, 1: count1_geom},
        )

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

    return ExactCoverageState(
        undercovered_geom=undercovered_geom,
        undercovered_area=float(undercovered_geom.area),
        deficit_area=float((count0_geom.area * 2.0) + count1_geom.area),
        geoms_by_count={0: count0_geom, 1: count1_geom, 2: covered_twice_or_more},
    )


def _adaptive_requirement_state(
    need_one_mask: BaseGeometry,
    need_two_mask: BaseGeometry,
    adaptive_circles: list[CandidateCircle],
) -> AdaptiveCoverageState:
    need_one_state = _exact_coverage_state(
        target_mask=need_one_mask,
        circle_geometries=[candidate.need_one_geom for candidate in adaptive_circles],
        min_cover_count=1,
    )
    need_two_state = _exact_coverage_state(
        target_mask=need_two_mask,
        circle_geometries=[candidate.need_two_geom for candidate in adaptive_circles],
        min_cover_count=2,
    )
    return AdaptiveCoverageState(
        need_one_state=need_one_state,
        need_two_state=need_two_state,
        total_undercovered_area=need_one_state.undercovered_area + need_two_state.undercovered_area,
        total_deficit_area=need_one_state.deficit_area + need_two_state.deficit_area,
    )


def _grid_base_origin(min_value: float, spacing: int) -> float:
    return math.floor((min_value - float(spacing)) / float(spacing)) * float(spacing)


def _build_square_grid_origins(
    bounds: tuple[float, float, float, float],
    radius: int,
) -> list[tuple[str, float, float]]:
    min_x, min_y, _max_x, _max_y = bounds
    base_x = _grid_base_origin(min_x, radius)
    base_y = _grid_base_origin(min_y, radius)

    origins: list[tuple[str, float, float]] = []
    for phase_x, phase_y in GRID_PHASES:
        origins.append(
            (
                f"{phase_x:.1f},{phase_y:.1f}",
                base_x + (phase_x * float(radius)),
                base_y + (phase_y * float(radius)),
            )
        )
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


def _generate_candidate_circles(
    non_built_target_mask: BaseGeometry,
    non_built_center_mask: BaseGeometry,
    built_target_mask: BaseGeometry,
    need_one_mask: BaseGeometry,
    need_two_mask: BaseGeometry,
) -> list[CandidateCircle]:
    if non_built_target_mask.is_empty or non_built_center_mask.is_empty:
        return []

    center_bounds = non_built_center_mask.bounds
    center_mask_prepared = prep(non_built_center_mask)
    built_target_empty = built_target_mask.is_empty or built_target_mask.area <= EPS
    need_one_empty = need_one_mask.is_empty or need_one_mask.area <= EPS
    need_two_empty = need_two_mask.is_empty or need_two_mask.area <= EPS
    seen_keys: set[tuple[float, float, int]] = set()
    candidates: list[CandidateCircle] = []

    for radius in ADAPTIVE_RADII_M:
        for phase_label, origin_x, origin_y in _build_square_grid_origins(center_bounds, radius):
            for _row, _col, x, y in _iter_square_nodes_in_bounds(center_bounds, radius, origin_x, origin_y):
                key = (round(x, 3), round(y, 3), int(radius))
                if key in seen_keys:
                    continue

                point = Point(x, y)
                if not center_mask_prepared.covers(point):
                    continue

                full_geom = point.buffer(float(radius), quad_segs=BUFFER_QUAD_SEGS)
                non_built_geom = full_geom.intersection(non_built_target_mask)
                if non_built_geom.is_empty or non_built_geom.area <= EPS:
                    continue

                need_one_geom = GeometryCollection() if need_one_empty else non_built_geom.intersection(need_one_mask)
                need_two_geom = GeometryCollection() if need_two_empty else non_built_geom.intersection(need_two_mask)
                if (
                    (need_one_geom.is_empty or need_one_geom.area <= EPS)
                    and (need_two_geom.is_empty or need_two_geom.area <= EPS)
                ):
                    continue

                built_overlap_area = (
                    0.0
                    if built_target_empty
                    else float(full_geom.intersection(built_target_mask).area)
                )

                seen_keys.add(key)
                candidates.append(
                    CandidateCircle(
                        x=float(x),
                        y=float(y),
                        radius=int(radius),
                        phase_label=phase_label,
                        full_geom=full_geom,
                        non_built_geom=non_built_geom,
                        need_one_geom=need_one_geom,
                        need_two_geom=need_two_geom,
                        built_overlap_area=built_overlap_area,
                    )
                )

    return candidates


def _select_adaptive_circles(
    candidates: list[CandidateCircle],
    need_one_mask: BaseGeometry,
    need_two_mask: BaseGeometry,
    frozen_twice_mask: BaseGeometry,
) -> tuple[list[CandidateCircle], AdaptiveCoverageState]:
    remaining = list(candidates)
    selected: list[CandidateCircle] = []

    while True:
        state = _adaptive_requirement_state(
            need_one_mask=need_one_mask,
            need_two_mask=need_two_mask,
            adaptive_circles=selected,
        )
        if state.total_undercovered_area <= EXACT_TOLERANCE_M2 + EPS:
            return selected, state

        satisfied_one_geom = need_one_mask.difference(state.need_one_state.undercovered_geom)
        satisfied_two_geom = _geom_or_empty(state.need_two_state.geoms_by_count.get(2))

        best_index: int | None = None
        best_score: tuple[float, float, float, int, float, float] | None = None
        for index, candidate in enumerate(remaining):
            undercovered_overlap_area = 0.0
            if not candidate.need_one_geom.is_empty and candidate.need_one_geom.area > EPS:
                undercovered_overlap_area += float(
                    candidate.need_one_geom.intersection(state.need_one_state.undercovered_geom).area
                )
            if not candidate.need_two_geom.is_empty and candidate.need_two_geom.area > EPS:
                undercovered_overlap_area += float(
                    candidate.need_two_geom.intersection(state.need_two_state.undercovered_geom).area
                )
            if undercovered_overlap_area <= EPS:
                continue

            redundant_overlap_area = 0.0
            if not frozen_twice_mask.is_empty and frozen_twice_mask.area > EPS:
                redundant_overlap_area += float(candidate.non_built_geom.intersection(frozen_twice_mask).area)
            if not candidate.need_one_geom.is_empty and candidate.need_one_geom.area > EPS:
                redundant_overlap_area += float(candidate.need_one_geom.intersection(satisfied_one_geom).area)
            if not candidate.need_two_geom.is_empty and candidate.need_two_geom.area > EPS:
                redundant_overlap_area += float(candidate.need_two_geom.intersection(satisfied_two_geom).area)

            score = (
                -undercovered_overlap_area,
                candidate.built_overlap_area,
                redundant_overlap_area,
                -candidate.radius,
                candidate.y,
                candidate.x,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_index = index

        if best_index is None:
            _fatal(
                "adaptive selection exhausted all candidates before exact non-built 2-coverage was satisfied"
            )

        selected.append(remaining.pop(best_index))


def _prune_adaptive_circles(
    selected: list[CandidateCircle],
    need_one_mask: BaseGeometry,
    need_two_mask: BaseGeometry,
) -> tuple[list[CandidateCircle], AdaptiveCoverageState, int]:
    pruned = list(selected)
    removed_total = 0

    while True:
        removed_this_pass = 0
        for index in range(len(pruned) - 1, -1, -1):
            trial = pruned[:index] + pruned[index + 1 :]
            state = _adaptive_requirement_state(
                need_one_mask=need_one_mask,
                need_two_mask=need_two_mask,
                adaptive_circles=trial,
            )
            if state.total_undercovered_area <= EXACT_TOLERANCE_M2 + EPS:
                pruned = trial
                removed_total += 1
                removed_this_pass += 1
        if removed_this_pass == 0:
            break

    final_state = _adaptive_requirement_state(
        need_one_mask=need_one_mask,
        need_two_mask=need_two_mask,
        adaptive_circles=pruned,
    )
    return pruned, final_state, removed_total


def _adaptive_rows(
    candidates: list[CandidateCircle],
    projection: LocalProjection,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        lng, lat = projection.to_lng_lat(candidate.x, candidate.y)
        rows.append(
            {
                "lat": round(lat, 9),
                "lng": round(lng, 9),
                "radius": int(candidate.radius),
                "collected": False,
                "population_density": None,
            }
        )
    rows.sort(key=lambda item: (item["radius"], item["lat"], item["lng"]))
    return rows


def _with_circle_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numbered_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for row in rows:
        lat = _to_float(row.get("lat"))
        lng = _to_float(row.get("lng"))
        radius = _to_float(row.get("radius"))
        if lat is None or lng is None or radius is None:
            _fatal("circle rows must contain numeric lat/lng/radius values before assigning circle_id")

        circle_id = f"circle_r{int(radius)}_lat{lat:.9f}_lng{lng:.9f}"
        if circle_id in seen_ids:
            _fatal(f"duplicate circle_id generated: {circle_id}")
        seen_ids.add(circle_id)

        row_without_id = {key: value for key, value in row.items() if key != "circle_id"}
        numbered_rows.append({"circle_id": circle_id, **row_without_id})

    return numbered_rows


def _radius_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for row in rows:
        radius = int(row["radius"])
        counts[radius] = counts.get(radius, 0) + 1
    return counts


def _coverage_counts_for_points(
    point_x: np.ndarray,
    point_y: np.ndarray,
    circles: list[CirclePlacement],
) -> np.ndarray:
    counts = np.zeros(point_x.size, dtype=np.int16)
    for index, (x, y) in enumerate(zip(point_x, point_y)):
        cover_count = 0
        for circle in circles:
            dx = float(x) - circle.x
            dy = float(y) - circle.y
            if (dx * dx) + (dy * dy) <= float(circle.radius * circle.radius) + EPS:
                cover_count += 1
        counts[index] = cover_count
    return counts


def main() -> int:
    border_geom = _load_border_geometry(BORDER_PATH)
    projection = _build_projection(border_geom)

    existing_rows = _load_existing_rows(OUTPUT_PATH)
    frozen_rows = _extract_frozen_rows(existing_rows)
    frozen_circles = _rows_to_circle_placements(frozen_rows, projection)

    targets = _load_target_masks(TERASCOPE_CSV_PATH, projection)

    built_coverage_counts = _coverage_counts_for_points(
        point_x=targets.built_center_x,
        point_y=targets.built_center_y,
        circles=frozen_circles,
    )
    if int(built_coverage_counts.min()) < MIN_COVER_COUNT:
        _fatal("frozen 500 m circles no longer provide at least 2-circle coverage on built-up cells")

    frozen_non_built_geometries = _circle_geometries(
        circles=frozen_circles,
        clip_to=targets.non_built_target_mask,
    )
    frozen_state = _exact_coverage_state(
        target_mask=targets.non_built_target_mask,
        circle_geometries=frozen_non_built_geometries,
        min_cover_count=MIN_COVER_COUNT,
    )

    need_one_mask = _geom_or_empty(frozen_state.geoms_by_count.get(1))
    need_two_mask = _geom_or_empty(frozen_state.geoms_by_count.get(0))
    frozen_twice_mask = _geom_or_empty(frozen_state.geoms_by_count.get(2))

    candidates = _generate_candidate_circles(
        non_built_target_mask=targets.non_built_target_mask,
        non_built_center_mask=targets.non_built_center_mask,
        built_target_mask=targets.built_target_mask,
        need_one_mask=need_one_mask,
        need_two_mask=need_two_mask,
    )
    if frozen_state.undercovered_area > EXACT_TOLERANCE_M2 + EPS and not candidates:
        _fatal("generated zero adaptive candidates for the remaining non-built target area")

    selected, _selected_state = _select_adaptive_circles(
        candidates=candidates,
        need_one_mask=need_one_mask,
        need_two_mask=need_two_mask,
        frozen_twice_mask=frozen_twice_mask,
    )
    pruned, final_adaptive_state, removed_total = _prune_adaptive_circles(
        selected=selected,
        need_one_mask=need_one_mask,
        need_two_mask=need_two_mask,
    )

    if final_adaptive_state.total_undercovered_area > EXACT_TOLERANCE_M2 + EPS:
        _fatal(
            "adaptive pruning left non-built target coverage incomplete: "
            f"{final_adaptive_state.total_undercovered_area:.6f} m^2 undercovered"
        )

    adaptive_rows = _adaptive_rows(pruned, projection)
    merged_rows = _with_circle_ids([*frozen_rows, *adaptive_rows])
    OUTPUT_PATH.write_text(
        json.dumps(merged_rows, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    adaptive_circles = _rows_to_circle_placements(adaptive_rows, projection)
    final_non_built_state = _exact_coverage_state(
        target_mask=targets.non_built_target_mask,
        circle_geometries=_circle_geometries(
            circles=[*frozen_circles, *adaptive_circles],
            clip_to=targets.non_built_target_mask,
        ),
        min_cover_count=MIN_COVER_COUNT,
    )
    if final_non_built_state.undercovered_area > EXACT_TOLERANCE_M2 + EPS:
        _fatal(
            "final merged layer failed exact non-built 2-coverage validation: "
            f"{final_non_built_state.undercovered_area:.6f} m^2 undercovered"
        )

    adaptive_radius_counts = _radius_counts(adaptive_rows)
    print(
        f"Frozen 500m circles: {len(frozen_rows)} | "
        f"Adaptive circles: {len(adaptive_rows)} | "
        f"Adaptive by radius: {adaptive_radius_counts} | "
        f"Remaining non-built undercovered area m2: {final_non_built_state.undercovered_area:.6f} | "
        f"Adaptive pruned removed: {removed_total} | "
        f"Output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
