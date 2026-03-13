#!/usr/bin/env python3
"""
Generate 500 m collection circles over built-up WorldCover pixels.

Pipeline:
1) Load the local Chiang Mai border GeoJSON.
2) Read only the raster windows that overlap the border bbox.
3) Keep WorldCover built-up pixel centers inside the border.
4) Lay down a 500 m square lattice buffered by 500 m around built-up pixels.
5) Keep only circles that cover at least one built-up pixel center.
6) Verify each built-up pixel center is covered by at least two kept circles.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import intersects_xy
from shapely.geometry import shape
from shapely.ops import unary_union


REPO_ROOT = Path(__file__).resolve().parent
BORDER_PATH = REPO_ROOT / "chiang_mai_main_area_merged_border.geojson"
OUTPUT_PATH = REPO_ROOT / "lat_lng_radius.json"
RASTER_GLOB = "data/*/*_Map.tif"

BUILT_UP_VALUE = 50
GRID_SPACING_M = 500.0
RADIUS_M = 500.0
RADIUS_SQ_M = RADIUS_M * RADIUS_M
METERS_PER_DEGREE = 111320.0
EPS = 1e-9


def _fatal(message: str) -> "None":
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


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
class RasterProfile:
    crs: str
    res_x: float
    res_y: float


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


def _load_raster_paths() -> list[Path]:
    paths = sorted(REPO_ROOT.glob(RASTER_GLOB))
    if not paths:
        _fatal(f"no raster files found under {REPO_ROOT / 'data'}")
    return paths


def _validate_raster_profile(
    path: Path,
    src: rasterio.io.DatasetReader,
    expected: RasterProfile | None,
) -> RasterProfile:
    if src.crs is None:
        _fatal(f"raster has no CRS: {path}")

    current = RasterProfile(
        crs=str(src.crs),
        res_x=float(src.res[0]),
        res_y=float(src.res[1]),
    )

    if current.crs != "EPSG:4326":
        _fatal(f"expected EPSG:4326 raster, found {current.crs} in {path}")

    if expected is None:
        return current

    if current.crs != expected.crs:
        _fatal(f"raster CRS mismatch in {path}: {current.crs} vs {expected.crs}")

    if not math.isclose(current.res_x, expected.res_x, rel_tol=0.0, abs_tol=1e-12):
        _fatal(f"raster x resolution mismatch in {path}")

    if not math.isclose(current.res_y, expected.res_y, rel_tol=0.0, abs_tol=1e-12):
        _fatal(f"raster y resolution mismatch in {path}")

    return expected


def _window_for_bounds(
    src: rasterio.io.DatasetReader,
    bounds: tuple[float, float, float, float],
) -> Window | None:
    left, bottom, right, top = bounds

    if (
        right <= src.bounds.left + EPS
        or left >= src.bounds.right - EPS
        or top <= src.bounds.bottom + EPS
        or bottom >= src.bounds.top - EPS
    ):
        return None

    raw_window = src.window(left, bottom, right, top)

    row_start = max(0, int(math.floor(raw_window.row_off)))
    col_start = max(0, int(math.floor(raw_window.col_off)))
    row_stop = min(src.height, int(math.ceil(raw_window.row_off + raw_window.height)))
    col_stop = min(src.width, int(math.ceil(raw_window.col_off + raw_window.width)))

    if row_stop <= row_start or col_stop <= col_start:
        return None

    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def _read_built_up_points_xy(
    raster_paths: list[Path],
    border_geom,
    projection: LocalProjection,
) -> tuple[np.ndarray, np.ndarray]:
    expected_profile: RasterProfile | None = None
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for path in raster_paths:
        with rasterio.open(path) as src:
            expected_profile = _validate_raster_profile(path, src, expected_profile)
            window = _window_for_bounds(src, border_geom.bounds)
            if window is None:
                continue

            data = src.read(1, window=window)
            built_up_mask = data == BUILT_UP_VALUE
            if not np.any(built_up_mask):
                continue

            rows, cols = np.nonzero(built_up_mask)
            transform = src.window_transform(window)
            col_positions = cols.astype(np.float64) + 0.5
            row_positions = rows.astype(np.float64) + 0.5
            lngs = (transform.a * col_positions) + (transform.b * row_positions) + transform.c
            lats = (transform.d * col_positions) + (transform.e * row_positions) + transform.f

            inside_border = np.asarray(intersects_xy(border_geom, lngs, lats), dtype=bool)
            if not np.any(inside_border):
                continue

            xs, ys = projection.to_xy(lngs[inside_border], lats[inside_border])
            x_parts.append(np.asarray(xs, dtype=np.float64))
            y_parts.append(np.asarray(ys, dtype=np.float64))

    if not x_parts or not y_parts:
        _fatal("found zero built-up pixels inside the border")

    return np.concatenate(x_parts), np.concatenate(y_parts)


def _build_grid_axis(min_value: float, max_value: float) -> np.ndarray:
    start = math.floor((min_value - RADIUS_M) / GRID_SPACING_M) * GRID_SPACING_M
    end = math.ceil((max_value + RADIUS_M) / GRID_SPACING_M) * GRID_SPACING_M
    count = int(round((end - start) / GRID_SPACING_M)) + 1
    if count <= 0:
        _fatal("invalid grid extent")
    return start + (np.arange(count, dtype=np.float64) * GRID_SPACING_M)


def _build_active_circle_mask(
    point_x: np.ndarray,
    point_y: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
) -> np.ndarray:
    active_mask = np.zeros((y_centers.size, x_centers.size), dtype=bool)

    for y_index, center_y in enumerate(y_centers):
        dy = point_y - center_y
        y_slab = np.abs(dy) <= RADIUS_M
        if not np.any(y_slab):
            continue

        slab_x = point_x[y_slab]
        slab_dy = dy[y_slab]

        for x_index, center_x in enumerate(x_centers):
            dx = slab_x - center_x
            x_slab = np.abs(dx) <= RADIUS_M
            if not np.any(x_slab):
                continue

            if np.any((dx[x_slab] * dx[x_slab]) + (slab_dy[x_slab] * slab_dy[x_slab]) <= RADIUS_SQ_M + EPS):
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
    seen_keys: set[tuple[float, float]] = set()

    for y_index, center_y in enumerate(y_centers):
        for x_index, center_x in enumerate(x_centers):
            if not active_mask[y_index, x_index]:
                continue

            lng, lat = projection.to_lng_lat(float(center_x), float(center_y))
            key = (round(lat, 6), round(lng, 6))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            rows.append(
                {
                    "lat": key[0],
                    "lng": key[1],
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
    raster_paths = _load_raster_paths()
    built_up_x, built_up_y = _read_built_up_points_xy(raster_paths, border_geom, projection)

    x_centers = _build_grid_axis(float(np.min(built_up_x)), float(np.max(built_up_x)))
    y_centers = _build_grid_axis(float(np.min(built_up_y)), float(np.max(built_up_y)))
    candidate_center_count = int(x_centers.size * y_centers.size)

    active_mask = _build_active_circle_mask(
        point_x=built_up_x,
        point_y=built_up_y,
        x_centers=x_centers,
        y_centers=y_centers,
    )

    kept_circle_count = int(np.count_nonzero(active_mask))
    if kept_circle_count == 0:
        _fatal("grid generation produced zero circles covering built-up pixels")

    coverage_counts = _coverage_counts_for_points(
        point_x=built_up_x,
        point_y=built_up_y,
        x_centers=x_centers,
        y_centers=y_centers,
        active_mask=active_mask,
    )
    min_coverage_count = int(coverage_counts.min())
    if min_coverage_count < 2:
        _fatal(
            "built-up coverage target failed: "
            f"minimum coverage count was {min_coverage_count}, expected at least 2"
        )

    output_rows = _rows_from_active_mask(active_mask, x_centers, y_centers, projection)
    OUTPUT_PATH.write_text(
        json.dumps(output_rows, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Built-up pixels inside border: {built_up_x.size} | "
        f"Candidate centers checked: {candidate_center_count} | "
        f"Circles kept: {kept_circle_count} | "
        f"Minimum built-up coverage count: {min_coverage_count} | "
        f"Output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
