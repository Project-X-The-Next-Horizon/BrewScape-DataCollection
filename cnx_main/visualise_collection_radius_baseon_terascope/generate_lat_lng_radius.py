#!/usr/bin/env python3
"""
Generate 500 m collection circles from Terascope 100 m built-up cells.

Pipeline:
1) Load the local Chiang Mai border GeoJSON for projection anchoring.
2) Read Chiang Mai Terascope 100 m cells from the sibling CSV.
3) Keep only cells whose dominant land-cover label is Built-up.
4) Build a 500 m square lattice around the built 100 m cell bounds.
5) Keep a circle when it intersects at least one built 100 m square.
6) Verify every built 100 m cell center is covered by at least two circles.
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
from shapely.geometry import shape
from shapely.ops import unary_union


REPO_ROOT = Path(__file__).resolve().parent
BORDER_PATH = REPO_ROOT / "chiang_mai_main_area_merged_border.geojson"
TERASCOPE_CSV_PATH = REPO_ROOT.parent / "terascope" / "chiang_mai_terascope_100m_cells.csv"
OUTPUT_PATH = REPO_ROOT / "lat_lng_radius.json"

TARGET_LAND_COVER_LABEL = "Built-up"
GRID_SPACING_M = 500.0
RADIUS_M = 500.0
RADIUS_SQ_M = RADIUS_M * RADIUS_M
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
class BuiltCellData:
    left_x: np.ndarray
    right_x: np.ndarray
    bottom_y: np.ndarray
    top_y: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray


def _load_border_geometry(path: Path):
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


def _build_projection(border_geom) -> LocalProjection:
    centroid = border_geom.centroid
    lat0 = float(centroid.y)
    return LocalProjection(
        lng0=float(centroid.x),
        lat0=lat0,
        cos_lat0=math.cos(math.radians(lat0)),
    )


def _load_built_cells(path: Path, projection: LocalProjection) -> BuiltCellData:
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

    left_x_values: list[float] = []
    right_x_values: list[float] = []
    bottom_y_values: list[float] = []
    top_y_values: list[float] = []
    center_x_values: list[float] = []
    center_y_values: list[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            _fatal(f"CSV has no header row: {path}")

        missing = required_columns.difference(reader.fieldnames)
        if missing:
            _fatal(f"CSV missing columns: {', '.join(sorted(missing))}")

        for raw in reader:
            if raw.get("dominant_land_cover_label") != TARGET_LAND_COVER_LABEL:
                continue

            lon_center = _to_float(raw.get("lon_center"))
            lat_center = _to_float(raw.get("lat_center"))
            lon_left = _to_float(raw.get("lon_left"))
            lon_right = _to_float(raw.get("lon_right"))
            lat_bottom = _to_float(raw.get("lat_bottom"))
            lat_top = _to_float(raw.get("lat_top"))

            if (
                lon_center is None
                or lat_center is None
                or lon_left is None
                or lon_right is None
                or lat_bottom is None
                or lat_top is None
            ):
                continue

            if not (
                -180.0 <= lon_center <= 180.0
                and -90.0 <= lat_center <= 90.0
                and -180.0 <= lon_left <= 180.0
                and -180.0 <= lon_right <= 180.0
                and -90.0 <= lat_bottom <= 90.0
                and -90.0 <= lat_top <= 90.0
            ):
                continue

            center_x, center_y = projection.to_xy(lon_center, lat_center)
            left_bottom_x, left_bottom_y = projection.to_xy(lon_left, lat_bottom)
            right_top_x, right_top_y = projection.to_xy(lon_right, lat_top)

            left_x_values.append(min(left_bottom_x, right_top_x))
            right_x_values.append(max(left_bottom_x, right_top_x))
            bottom_y_values.append(min(left_bottom_y, right_top_y))
            top_y_values.append(max(left_bottom_y, right_top_y))
            center_x_values.append(center_x)
            center_y_values.append(center_y)

    if not center_x_values:
        _fatal("found zero dominant built-up Terascope cells")

    return BuiltCellData(
        left_x=np.asarray(left_x_values, dtype=np.float64),
        right_x=np.asarray(right_x_values, dtype=np.float64),
        bottom_y=np.asarray(bottom_y_values, dtype=np.float64),
        top_y=np.asarray(top_y_values, dtype=np.float64),
        center_x=np.asarray(center_x_values, dtype=np.float64),
        center_y=np.asarray(center_y_values, dtype=np.float64),
    )


def _build_grid_axis(min_value: float, max_value: float) -> np.ndarray:
    start = math.floor((min_value - RADIUS_M) / GRID_SPACING_M) * GRID_SPACING_M
    end = math.ceil((max_value + RADIUS_M) / GRID_SPACING_M) * GRID_SPACING_M
    count = int(round((end - start) / GRID_SPACING_M)) + 1
    if count <= 0:
        _fatal("invalid grid extent")
    return start + (np.arange(count, dtype=np.float64) * GRID_SPACING_M)


def _build_active_circle_mask(
    built_cells: BuiltCellData,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> np.ndarray:
    active_mask = np.zeros((y_centers.size, x_centers.size), dtype=bool)

    for y_index, center_y in enumerate(y_centers):
        dy = np.maximum(
            np.maximum(built_cells.bottom_y - center_y, 0.0),
            center_y - built_cells.top_y,
        )
        y_overlap = dy <= RADIUS_M
        if not np.any(y_overlap):
            continue

        left_x = built_cells.left_x[y_overlap]
        right_x = built_cells.right_x[y_overlap]
        dy = dy[y_overlap]

        for x_index, center_x in enumerate(x_centers):
            dx = np.maximum(
                np.maximum(left_x - center_x, 0.0),
                center_x - right_x,
            )
            x_overlap = dx <= RADIUS_M
            if not np.any(x_overlap):
                continue

            if np.any((dx[x_overlap] * dx[x_overlap]) + (dy[x_overlap] * dy[x_overlap]) <= RADIUS_SQ_M + EPS):
                active_mask[y_index, x_index] = True

    return active_mask


def _coverage_counts_for_points(
    point_x: np.ndarray,
    point_y: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    counts = np.zeros(point_x.size, dtype=np.int16)
    start_x = float(x_centers[0])
    start_y = float(y_centers[0])
    max_x_index = x_centers.size - 1
    max_y_index = y_centers.size - 1

    for idx in range(point_x.size):
        x = float(point_x[idx])
        y = float(point_y[idx])
        base_x_index = int(math.floor((x - start_x) / GRID_SPACING_M))
        base_y_index = int(math.floor((y - start_y) / GRID_SPACING_M))
        cover_count = 0

        for y_index in range(base_y_index - 1, base_y_index + 2):
            if y_index < 0 or y_index > max_y_index:
                continue
            center_y = float(y_centers[y_index])
            dy = y - center_y
            if abs(dy) > RADIUS_M:
                continue

            for x_index in range(base_x_index - 1, base_x_index + 2):
                if x_index < 0 or x_index > max_x_index or not active_mask[y_index, x_index]:
                    continue

                center_x = float(x_centers[x_index])
                dx = x - center_x
                if abs(dx) > RADIUS_M:
                    continue

                if (dx * dx) + (dy * dy) <= RADIUS_SQ_M + EPS:
                    cover_count += 1

        counts[idx] = cover_count

    return counts


def _rows_from_active_mask(
    active_mask: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    projection: LocalProjection,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()

    for y_index, center_y in enumerate(y_centers):
        for x_index, center_x in enumerate(x_centers):
            if not active_mask[y_index, x_index]:
                continue

            lng, lat = projection.to_lng_lat(float(center_x), float(center_y))
            key = (y_index, x_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            rows.append(
                {
                    "lat": round(lat, 9),
                    "lng": round(lng, 9),
                    "radius": int(RADIUS_M),
                    "collected": False,
                    "population_density": None,
                }
            )

    rows.sort(key=lambda item: (item["lat"], item["lng"]))
    return rows


def main() -> int:
    border_geom = _load_border_geometry(BORDER_PATH)
    projection = _build_projection(border_geom)
    built_cells = _load_built_cells(TERASCOPE_CSV_PATH, projection)

    x_min = float(np.min(built_cells.left_x))
    x_max = float(np.max(built_cells.right_x))
    y_min = float(np.min(built_cells.bottom_y))
    y_max = float(np.max(built_cells.top_y))

    x_centers = _build_grid_axis(x_min, x_max)
    y_centers = _build_grid_axis(y_min, y_max)
    candidate_center_count = int(x_centers.size * y_centers.size)

    active_mask = _build_active_circle_mask(
        built_cells=built_cells,
        x_centers=x_centers,
        y_centers=y_centers,
    )

    kept_circle_count = int(np.count_nonzero(active_mask))
    if kept_circle_count == 0:
        _fatal("grid generation produced zero circles intersecting built-up Terascope cells")

    coverage_counts = _coverage_counts_for_points(
        point_x=built_cells.center_x,
        point_y=built_cells.center_y,
        x_centers=x_centers,
        y_centers=y_centers,
        active_mask=active_mask,
    )
    min_coverage_count = int(coverage_counts.min())
    if min_coverage_count < 2:
        _fatal(
            "built-up coverage target failed: "
            f"minimum built-cell-center coverage count was {min_coverage_count}, expected at least 2"
        )

    output_rows = _rows_from_active_mask(active_mask, x_centers, y_centers, projection)
    OUTPUT_PATH.write_text(
        json.dumps(output_rows, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Dominant built 100m cells used: {built_cells.center_x.size} | "
        f"Candidate centers checked: {candidate_center_count} | "
        f"Circles kept: {kept_circle_count} | "
        f"Minimum built-cell-center coverage count: {min_coverage_count} | "
        f"Output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
