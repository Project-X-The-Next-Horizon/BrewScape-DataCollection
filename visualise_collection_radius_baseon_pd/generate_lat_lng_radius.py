#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from shapely.geometry import Point, shape
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import transform, unary_union
except ImportError:  # pragma: no cover
    import sys

    print("Error: install shapely with: pip install shapely", file=sys.stderr)
    raise SystemExit(1)


REPO_ROOT = Path(__file__).resolve().parent
INPUT_DENSITY_PATH = REPO_ROOT / "chiang_mai_population_density_cells.csv"
BORDER_PATH = REPO_ROOT / "chiang_mai_main_area_merged_border.geojson"
OUTPUT_PATH = REPO_ROOT / "lat_lng_radius.json"

ROUNDING_METERS = 25
IDW_NEIGHBORS = 4
IDW_EPS = 1e-12
EPS = 1e-9
EXACT_BUFFER_RESOLUTION = 64


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


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        _fatal("cannot compute percentile of empty list")
    if p <= 0:
        return sorted_values[0]
    if p >= 1:
        return sorted_values[-1]
    idx = (len(sorted_values) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    w = idx - lo
    return (sorted_values[lo] * (1.0 - w)) + (sorted_values[hi] * w)


def _auto_max_radius(min_radius: int, densities: list[float]) -> int:
    if not densities:
        _fatal("cannot choose max radius without density values")
    sorted_density = sorted(densities)
    q10 = _percentile(sorted_density, 0.10)
    q90 = _percentile(sorted_density, 0.90)
    ratio = q90 / max(q10, 1.0)
    if ratio >= 60:
        candidate = min_radius + 1400
    elif ratio >= 30:
        candidate = min_radius + 1200
    elif ratio >= 15:
        candidate = min_radius + 1000
    elif ratio >= 8:
        candidate = min_radius + 800
    else:
        candidate = min_radius + 600
    candidate = max(min_radius + 200, min(2500, candidate))
    return int(round(candidate / 50.0) * 50)


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


def _load_density_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        _fatal(f"input CSV not found: {path}")
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"lon_center", "lat_center", "population_density"}
        if reader.fieldnames is None:
            _fatal(f"CSV has no header row: {path}")
        missing = required.difference(reader.fieldnames)
        if missing:
            _fatal(f"CSV missing columns: {', '.join(sorted(missing))}")
        for raw in reader:
            lon = _to_float(raw.get("lon_center"))
            lat = _to_float(raw.get("lat_center"))
            density = _to_float(raw.get("population_density"))
            if (
                lon is None
                or lat is None
                or density is None
                or not math.isfinite(lon)
                or not math.isfinite(lat)
                or not math.isfinite(density)
            ):
                continue
            if density < 0 or not (-180 <= lon <= 180 and -90 <= lat <= 90):
                continue
            rows.append({"lng": lon, "lat": lat, "population_density": density})
    if not rows:
        _fatal("no valid rows found in density CSV")
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
class DensityPoint:
    x: float
    y: float
    density: float


@dataclass
class CirclePlacement:
    x: float
    y: float
    radius: int
    density: float


@dataclass
class CandidateResult:
    spacing_scale: float
    band_step: int
    circles: list[CirclePlacement]
    initial_honeycomb_count: int
    sample_patch_added: int
    exact_patch_added: int
    prune_removed: int
    sample_uncovered: int
    exact_missing_area: float
    overlap_ratio: float
    mean_multiplicity: float
    spacing_score: float
    elapsed_seconds: float


def _build_projection(border_geom_lng_lat: BaseGeometry) -> LocalProjection:
    centroid = border_geom_lng_lat.centroid
    lat0 = float(centroid.y)
    return LocalProjection(
        lng0=float(centroid.x),
        lat0=lat0,
        cos_lat0=math.cos(math.radians(lat0)),
    )


def _build_density_points(
    density_rows: list[dict[str, float]],
    projection: LocalProjection,
) -> list[DensityPoint]:
    return [
        DensityPoint(*projection.to_xy(float(row["lng"]), float(row["lat"])), float(row["population_density"]))
        for row in density_rows
    ]


def _prepare_density_scale(density_rows: list[dict[str, float]]) -> tuple[float, float]:
    log_values = sorted(math.log1p(max(0.0, row["population_density"])) for row in density_rows)
    return _percentile(log_values, 0.05), _percentile(log_values, 0.95)


def _normalize_density(density: float, low_log: float, high_log: float) -> float:
    if high_log <= low_log:
        return 1.0
    value = math.log1p(max(0.0, density))
    clipped = min(high_log, max(low_log, value))
    return (clipped - low_log) / (high_log - low_log)


def _radius_from_density(
    density: float,
    min_radius: int,
    max_radius: int,
    low_log: float,
    high_log: float,
) -> int:
    normalized = _normalize_density(density, low_log=low_log, high_log=high_log)
    radius = max_radius - (normalized * (max_radius - min_radius))
    rounded = int(round(radius / ROUNDING_METERS) * ROUNDING_METERS)
    return max(min_radius, min(max_radius, rounded))


def _query_density(x: float, y: float, density_points: list[DensityPoint]) -> float:
    if not density_points:
        _fatal("density points are empty")
    best: list[tuple[float, float]] = []
    for point in density_points:
        dx = x - point.x
        dy = y - point.y
        dist_sq = (dx * dx) + (dy * dy)
        if dist_sq <= IDW_EPS:
            return point.density
        if len(best) < IDW_NEIGHBORS:
            best.append((dist_sq, point.density))
            continue
        worst_idx = 0
        for idx in range(1, len(best)):
            if best[idx][0] > best[worst_idx][0]:
                worst_idx = idx
        if dist_sq < best[worst_idx][0]:
            best[worst_idx] = (dist_sq, point.density)
    weighted_sum = 0.0
    weight_total = 0.0
    for dist_sq, density in best:
        weight = 1.0 / max(dist_sq, IDW_EPS)
        weighted_sum += density * weight
        weight_total += weight
    return best[0][1] if weight_total <= 0 else (weighted_sum / weight_total)


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


def _center_key_xy(x: float, y: float) -> tuple[float, float]:
    return round(x, 3), round(y, 3)


def _point_is_covered(x: float, y: float, circles: list[CirclePlacement]) -> bool:
    for circle in circles:
        dx = x - circle.x
        dy = y - circle.y
        if (dx * dx) + (dy * dy) <= (circle.radius * circle.radius) + EPS:
            return True
    return False


def _build_radius_bands(min_radius: int, max_radius: int, band_step: int) -> list[tuple[int, int]]:
    highs: list[int] = []
    current = max_radius
    while current > min_radius:
        highs.append(current)
        current -= band_step
    highs.append(min_radius)
    bands: list[tuple[int, int]] = []
    for idx, high in enumerate(highs):
        low = highs[idx + 1] + 1 if idx + 1 < len(highs) else min_radius
        bands.append((low, high))
    return bands


def _generate_honeycomb_circles(
    border_xy: BaseGeometry,
    density_points: list[DensityPoint],
    min_radius: int,
    max_radius: int,
    low_log: float,
    high_log: float,
    band_step: int,
    spacing_scale: float,
    deadline: float,
) -> list[CirclePlacement] | None:
    circles: list[CirclePlacement] = []
    seen_centers: set[tuple[float, float]] = set()
    bands = _build_radius_bands(min_radius=min_radius, max_radius=max_radius, band_step=band_step)
    for band_low, band_high in bands:
        if time.perf_counter() > deadline:
            return None
        band_radius = band_high
        spacing = math.sqrt(3.0) * band_radius * spacing_scale
        candidate_zone = border_xy.buffer(float(band_radius))
        for x, y in _iter_staggered_grid(candidate_zone.bounds, spacing):
            point = Point(x, y)
            if not candidate_zone.covers(point):
                continue
            if _point_is_covered(x, y, circles):
                continue
            density = _query_density(x, y, density_points)
            adaptive_radius = _radius_from_density(
                density=density,
                min_radius=min_radius,
                max_radius=max_radius,
                low_log=low_log,
                high_log=high_log,
            )
            if adaptive_radius < band_low or adaptive_radius > band_high:
                continue
            key = _center_key_xy(x, y)
            if key in seen_centers:
                continue
            seen_centers.add(key)
            circles.append(CirclePlacement(x=x, y=y, radius=adaptive_radius, density=density))
    return circles


def _build_coverage_samples(area_xy: BaseGeometry, step: float) -> list[tuple[float, float]]:
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
            if area_xy.covers(Point(x, y)):
                samples.append((x, y))
            x += step
        y += step
    return samples


def _build_coverage_state(
    samples: list[tuple[float, float]],
    circles: list[CirclePlacement],
) -> tuple[list[int], list[float]]:
    coverage_counts = [0] * len(samples)
    nearest_center_d2 = [float("inf")] * len(samples)
    for circle in circles:
        cx, cy, r2 = circle.x, circle.y, float(circle.radius * circle.radius)
        for idx, (x, y) in enumerate(samples):
            dx = x - cx
            dy = y - cy
            d2 = (dx * dx) + (dy * dy)
            if d2 < nearest_center_d2[idx]:
                nearest_center_d2[idx] = d2
            if d2 <= r2 + EPS:
                coverage_counts[idx] += 1
    return coverage_counts, nearest_center_d2


def _update_coverage_state_for_circle(
    samples: list[tuple[float, float]],
    coverage_counts: list[int],
    nearest_center_d2: list[float],
    circle: CirclePlacement,
) -> None:
    cx, cy, r2 = circle.x, circle.y, float(circle.radius * circle.radius)
    for idx, (x, y) in enumerate(samples):
        dx = x - cx
        dy = y - cy
        d2 = (dx * dx) + (dy * dy)
        if d2 < nearest_center_d2[idx]:
            nearest_center_d2[idx] = d2
        if d2 <= r2 + EPS:
            coverage_counts[idx] += 1


def _count_uncovered(coverage_counts: list[int]) -> int:
    return sum(1 for count in coverage_counts if count == 0)


def _patch_sample_coverage_max_gap(
    samples: list[tuple[float, float]],
    circles: list[CirclePlacement],
    coverage_counts: list[int],
    nearest_center_d2: list[float],
    density_points: list[DensityPoint],
    min_radius: int,
    max_radius: int,
    low_log: float,
    high_log: float,
    max_gap_patch_limit: int,
    deadline: float,
) -> tuple[int, int] | None:
    initial_uncovered = _count_uncovered(coverage_counts)
    if initial_uncovered == 0:
        return 0, 0
    added = 0
    seen_centers = {_center_key_xy(circle.x, circle.y) for circle in circles}
    while True:
        if time.perf_counter() > deadline:
            return None
        best_idx = -1
        best_gap = -1.0
        for idx, covered_count in enumerate(coverage_counts):
            if covered_count != 0:
                continue
            gap = nearest_center_d2[idx]
            if gap > best_gap:
                best_gap = gap
                best_idx = idx
        if best_idx == -1:
            break
        x, y = samples[best_idx]
        key = _center_key_xy(x, y)
        if key in seen_centers:
            coverage_counts[best_idx] = 1
            continue
        density = _query_density(x, y, density_points)
        radius = _radius_from_density(
            density=density,
            min_radius=min_radius,
            max_radius=max_radius,
            low_log=low_log,
            high_log=high_log,
        )
        circle = CirclePlacement(x=x, y=y, radius=radius, density=density)
        circles.append(circle)
        seen_centers.add(key)
        added += 1
        _update_coverage_state_for_circle(samples, coverage_counts, nearest_center_d2, circle)
        if added > max_gap_patch_limit:
            return None
    return None if _count_uncovered(coverage_counts) != 0 else (initial_uncovered, added)


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


def _circle_geometries(circles: list[CirclePlacement]) -> list[BaseGeometry]:
    return [
        Point(circle.x, circle.y).buffer(float(circle.radius), resolution=EXACT_BUFFER_RESOLUTION)
        for circle in circles
    ]


def _exact_missing_geometry(
    border_xy: BaseGeometry,
    circle_geometries: list[BaseGeometry],
) -> tuple[BaseGeometry, float]:
    if not circle_geometries:
        return border_xy, float(border_xy.area)
    missing = border_xy.difference(unary_union(circle_geometries))
    return missing, float(missing.area)


def _patch_exact_coverage(
    border_xy: BaseGeometry,
    samples: list[tuple[float, float]],
    circles: list[CirclePlacement],
    coverage_counts: list[int],
    nearest_center_d2: list[float],
    density_points: list[DensityPoint],
    min_radius: int,
    max_radius: int,
    low_log: float,
    high_log: float,
    exact_tolerance_m2: float,
    exact_patch_limit: int,
    deadline: float,
) -> tuple[int, float] | None:
    added = 0
    seen_centers = {_center_key_xy(circle.x, circle.y) for circle in circles}
    while True:
        if time.perf_counter() > deadline:
            return None
        missing_geom, missing_area = _exact_missing_geometry(border_xy, _circle_geometries(circles))
        if missing_area <= exact_tolerance_m2 + EPS:
            return added, missing_area
        if added >= exact_patch_limit:
            return None
        parts = [part for part in _iter_polygon_parts(missing_geom) if part.area > EPS]
        if not parts:
            return None
        parts.sort(key=lambda part: part.area, reverse=True)
        selected_point: tuple[float, float] | None = None
        for part in parts:
            rep = part.representative_point()
            key = _center_key_xy(float(rep.x), float(rep.y))
            if key in seen_centers:
                continue
            selected_point = (float(rep.x), float(rep.y))
            break
        if selected_point is None:
            return None
        x, y = selected_point
        density = _query_density(x, y, density_points)
        radius = _radius_from_density(
            density=density,
            min_radius=min_radius,
            max_radius=max_radius,
            low_log=low_log,
            high_log=high_log,
        )
        circle = CirclePlacement(x=x, y=y, radius=radius, density=density)
        circles.append(circle)
        seen_centers.add(_center_key_xy(x, y))
        added += 1
        _update_coverage_state_for_circle(samples, coverage_counts, nearest_center_d2, circle)


def _build_cover_index(
    samples: list[tuple[float, float]],
    circles: list[CirclePlacement],
    active: list[bool],
) -> tuple[list[set[int]], list[int]]:
    cover_by_circle = [set() for _ in circles]
    cover_counts = [0] * len(samples)
    for ci, circle in enumerate(circles):
        if not active[ci]:
            continue
        cx, cy, r2 = circle.x, circle.y, float(circle.radius * circle.radius)
        for si, (x, y) in enumerate(samples):
            dx = x - cx
            dy = y - cy
            if (dx * dx) + (dy * dy) <= r2 + EPS:
                cover_by_circle[ci].add(si)
                cover_counts[si] += 1
    return cover_by_circle, cover_counts


def _exact_missing_for_active(
    border_xy: BaseGeometry,
    circle_geometries: list[BaseGeometry],
    active: list[bool],
) -> float:
    selected = [geom for idx, geom in enumerate(circle_geometries) if active[idx]]
    _missing, area = _exact_missing_geometry(border_xy, selected)
    return area


def _prune_circles(
    border_xy: BaseGeometry,
    samples: list[tuple[float, float]],
    circles: list[CirclePlacement],
    exact_tolerance_m2: float,
    prune_max_passes: int,
    deadline: float,
) -> tuple[list[CirclePlacement], int, list[int], float] | None:
    if not circles:
        return [], 0, [0] * len(samples), float(border_xy.area)
    active = [True] * len(circles)
    circle_geometries = _circle_geometries(circles)
    removed_total = 0
    for _ in range(prune_max_passes):
        if time.perf_counter() > deadline:
            return None
        cover_by_circle, cover_counts = _build_cover_index(samples, circles, active)
        active_indices = [idx for idx, is_active in enumerate(active) if is_active]
        if not active_indices:
            break
        unique_support = {
            ci: sum(1 for si in cover_by_circle[ci] if cover_counts[si] == 1) for ci in active_indices
        }
        order = sorted(
            active_indices,
            key=lambda ci: (unique_support[ci], len(cover_by_circle[ci]), circles[ci].radius),
        )
        removed_this_pass = 0
        for ci in order:
            if time.perf_counter() > deadline:
                return None
            if not active[ci]:
                continue
            if any(cover_counts[si] <= 1 for si in cover_by_circle[ci]):
                continue
            active[ci] = False
            missing_area = _exact_missing_for_active(border_xy, circle_geometries, active)
            if missing_area <= exact_tolerance_m2 + EPS:
                removed_this_pass += 1
                removed_total += 1
                for si in cover_by_circle[ci]:
                    cover_counts[si] -= 1
            else:
                active[ci] = True
        if removed_this_pass == 0:
            break
    pruned = [circle for idx, circle in enumerate(circles) if active[idx]]
    _cover_map, final_cover_counts = _build_cover_index(samples, circles, active)
    final_missing_area = _exact_missing_for_active(border_xy, circle_geometries, active)
    return pruned, removed_total, final_cover_counts, final_missing_area


def _spacing_score(circles: list[CirclePlacement]) -> float:
    if len(circles) <= 1:
        return float("inf")
    mins: list[float] = []
    for idx, ci in enumerate(circles):
        best = float("inf")
        for jdx, cj in enumerate(circles):
            if idx == jdx:
                continue
            dx = ci.x - cj.x
            dy = ci.y - cj.y
            dist = math.sqrt((dx * dx) + (dy * dy))
            denom = max(float(ci.radius + cj.radius), EPS)
            norm = dist / denom
            if norm < best:
                best = norm
        mins.append(best)
    mins.sort()
    return _percentile(mins, 0.25)


def _overlap_metrics(coverage_counts: list[int]) -> tuple[float, float, int]:
    if not coverage_counts:
        return 1.0, 0.0, 0
    total = len(coverage_counts)
    overlap_samples = sum(1 for c in coverage_counts if c >= 2)
    mean_mult = sum(coverage_counts) / total
    uncovered = sum(1 for c in coverage_counts if c == 0)
    return overlap_samples / total, mean_mult, uncovered


def _candidate_sort_key(candidate: CandidateResult) -> tuple[Any, ...]:
    return (
        len(candidate.circles),
        -candidate.spacing_score,
        candidate.overlap_ratio,
        candidate.mean_multiplicity,
    )


def _evaluate_candidate(
    border_xy: BaseGeometry,
    samples: list[tuple[float, float]],
    density_points: list[DensityPoint],
    min_radius: int,
    max_radius: int,
    low_log: float,
    high_log: float,
    spacing_scale: float,
    band_step: int,
    exact_tolerance_m2: float,
    max_gap_patch_limit: int,
    exact_patch_limit: int,
    prune_max_passes: int,
    deadline: float,
) -> CandidateResult | None:
    start = time.perf_counter()
    circles = _generate_honeycomb_circles(
        border_xy=border_xy,
        density_points=density_points,
        min_radius=min_radius,
        max_radius=max_radius,
        low_log=low_log,
        high_log=high_log,
        band_step=band_step,
        spacing_scale=spacing_scale,
        deadline=deadline,
    )
    if circles is None:
        return None
    initial_honeycomb_count = len(circles)
    coverage_counts, nearest_center_d2 = _build_coverage_state(samples, circles)
    sample_patch = _patch_sample_coverage_max_gap(
        samples=samples,
        circles=circles,
        coverage_counts=coverage_counts,
        nearest_center_d2=nearest_center_d2,
        density_points=density_points,
        min_radius=min_radius,
        max_radius=max_radius,
        low_log=low_log,
        high_log=high_log,
        max_gap_patch_limit=max_gap_patch_limit,
        deadline=deadline,
    )
    if sample_patch is None:
        return None
    _initial_uncovered, sample_patch_added = sample_patch
    exact_patch = _patch_exact_coverage(
        border_xy=border_xy,
        samples=samples,
        circles=circles,
        coverage_counts=coverage_counts,
        nearest_center_d2=nearest_center_d2,
        density_points=density_points,
        min_radius=min_radius,
        max_radius=max_radius,
        low_log=low_log,
        high_log=high_log,
        exact_tolerance_m2=exact_tolerance_m2,
        exact_patch_limit=exact_patch_limit,
        deadline=deadline,
    )
    if exact_patch is None:
        return None
    exact_patch_added, _missing_after_exact = exact_patch
    pruned = _prune_circles(
        border_xy=border_xy,
        samples=samples,
        circles=circles,
        exact_tolerance_m2=exact_tolerance_m2,
        prune_max_passes=prune_max_passes,
        deadline=deadline,
    )
    if pruned is None:
        return None
    pruned_circles, prune_removed, final_cover_counts, final_missing = pruned
    overlap_ratio, mean_mult, uncovered = _overlap_metrics(final_cover_counts)
    if uncovered != 0 or final_missing > exact_tolerance_m2 + EPS:
        return None
    return CandidateResult(
        spacing_scale=spacing_scale,
        band_step=band_step,
        circles=pruned_circles,
        initial_honeycomb_count=initial_honeycomb_count,
        sample_patch_added=sample_patch_added,
        exact_patch_added=exact_patch_added,
        prune_removed=prune_removed,
        sample_uncovered=uncovered,
        exact_missing_area=final_missing,
        overlap_ratio=overlap_ratio,
        mean_multiplicity=mean_mult,
        spacing_score=_spacing_score(pruned_circles),
        elapsed_seconds=time.perf_counter() - start,
    )


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
                "population_density": round(circle.density, 6),
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


def _parse_float_values_list(raw: str, name: str) -> list[float]:
    values: list[float] = []
    for piece in raw.split(","):
        text = piece.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            _fatal(f"invalid {name} value: {text!r}")
    if not values:
        _fatal(f"{name} produced an empty list")
    for value in values:
        if not math.isfinite(value) or value <= 0:
            _fatal(f"{name} contains non-positive value: {value}")
    return sorted(set(values))


def _parse_int_values_list(raw: str, name: str) -> list[int]:
    values: list[int] = []
    for piece in raw.split(","):
        text = piece.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError:
            _fatal(f"invalid {name} value: {text!r}")
    if not values:
        _fatal(f"{name} produced an empty list")
    for value in values:
        if value <= 0:
            _fatal(f"{name} contains non-positive value: {value}")
    return sorted(set(values))


def _candidate_combinations(spacing_scales: list[float], band_steps: list[int]) -> list[tuple[float, int]]:
    combos = [(scale, band) for scale in spacing_scales for band in band_steps]
    combos.sort(key=lambda item: (abs(item[0] - 0.94), abs(item[1] - 100), item[0], item[1]))
    return combos


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate optimized honeycomb adaptive circles for Chiang Mai."
    )
    parser.add_argument("--min-radius", type=int, default=500)
    parser.add_argument("--max-radius", type=int, default=None)
    parser.add_argument("--coverage-step", type=float, default=100.0)
    parser.add_argument("--band-step", type=int, default=100)
    parser.add_argument("--preserve-distance", type=float, default=300.0)
    parser.add_argument("--opt-max-seconds", type=float, default=60.0)
    parser.add_argument("--spacing-scale-values", type=str, default="0.86,0.90,0.94,0.98,1.02")
    parser.add_argument("--band-step-values", type=str, default="75,100,125")
    parser.add_argument("--exact-tolerance-m2", type=float, default=1.0)
    parser.add_argument("--max-gap-patch-limit", type=int, default=1200)
    parser.add_argument("--exact-patch-limit", type=int, default=1500)
    parser.add_argument("--prune-max-passes", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.min_radius < 100:
        _fatal("--min-radius must be >= 100")
    if args.coverage_step <= 0:
        _fatal("--coverage-step must be > 0")
    if args.band_step <= 0:
        _fatal("--band-step must be > 0")
    if args.preserve_distance < 0:
        _fatal("--preserve-distance must be >= 0")
    if args.opt_max_seconds <= 0:
        _fatal("--opt-max-seconds must be > 0")
    if args.exact_tolerance_m2 < 0:
        _fatal("--exact-tolerance-m2 must be >= 0")
    if args.max_gap_patch_limit <= 0:
        _fatal("--max-gap-patch-limit must be > 0")
    if args.exact_patch_limit <= 0:
        _fatal("--exact-patch-limit must be > 0")
    if args.prune_max_passes <= 0:
        _fatal("--prune-max-passes must be > 0")

    border_lng_lat = _load_border_geometry(BORDER_PATH)
    density_rows = _load_density_rows(INPUT_DENSITY_PATH)
    densities = [row["population_density"] for row in density_rows]

    max_radius = args.max_radius
    if max_radius is None:
        max_radius = _auto_max_radius(args.min_radius, densities)
    if max_radius <= args.min_radius:
        _fatal("--max-radius must be greater than --min-radius")

    spacing_scales = _parse_float_values_list(args.spacing_scale_values, "--spacing-scale-values")
    band_steps = _parse_int_values_list(args.band_step_values, "--band-step-values")
    if args.band_step not in band_steps:
        band_steps.append(args.band_step)
        band_steps = sorted(set(band_steps))
    combinations = _candidate_combinations(spacing_scales, band_steps)

    projection = _build_projection(border_lng_lat)
    border_xy = projection.geom_to_xy(border_lng_lat)
    density_points = _build_density_points(density_rows, projection)
    low_log, high_log = _prepare_density_scale(density_rows)
    samples = _build_coverage_samples(border_xy, args.coverage_step)

    optimization_start = time.perf_counter()
    deadline = optimization_start + args.opt_max_seconds
    tested = 0
    valid = 0
    best: CandidateResult | None = None

    for spacing_scale, band_step in combinations:
        if time.perf_counter() > deadline and tested > 0:
            break
        tested += 1
        candidate = _evaluate_candidate(
            border_xy=border_xy,
            samples=samples,
            density_points=density_points,
            min_radius=args.min_radius,
            max_radius=max_radius,
            low_log=low_log,
            high_log=high_log,
            spacing_scale=spacing_scale,
            band_step=band_step,
            exact_tolerance_m2=args.exact_tolerance_m2,
            max_gap_patch_limit=args.max_gap_patch_limit,
            exact_patch_limit=args.exact_patch_limit,
            prune_max_passes=args.prune_max_passes,
            deadline=deadline,
        )
        if candidate is None:
            continue
        valid += 1
        if best is None or _candidate_sort_key(candidate) < _candidate_sort_key(best):
            best = candidate

    if best is None:
        _fatal(
            "optimization failed: no valid candidate met strict sample + exact coverage "
            f"(tested {tested} candidate(s))."
        )

    output_rows = _circles_to_rows(best.circles, projection)
    old_rows = _load_existing_rows(OUTPUT_PATH)
    exact_preserved, nearest_preserved = _apply_collected_preservation(
        new_rows=output_rows,
        old_rows=old_rows,
        projection=projection,
        preserve_distance=args.preserve_distance,
    )

    OUTPUT_PATH.write_text(
        json.dumps(output_rows, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    radii = [int(row["radius"]) for row in output_rows]
    unique_radii = sorted(set(radii))
    total_runtime = time.perf_counter() - optimization_start

    print(
        f"Candidates tested: {tested} ({valid} valid) | "
        f"Best config: spacing-scale={best.spacing_scale:.3f}, band-step={best.band_step} | "
        f"Circles: {len(output_rows)} | "
        f"Unique radii: {len(unique_radii)} | "
        f"Min radius: {min(radii)} | "
        f"Max radius: {max(radii)} | "
        f"Honeycomb seed: {best.initial_honeycomb_count} | "
        f"Sample patch added: {best.sample_patch_added} | "
        f"Exact patch added: {best.exact_patch_added} | "
        f"Pruned removed: {best.prune_removed} | "
        f"Sample uncovered: {best.sample_uncovered} | "
        f"Exact missing area m2: {best.exact_missing_area:.6f} | "
        f"Overlap ratio: {best.overlap_ratio:.4f} | "
        f"Mean multiplicity: {best.mean_multiplicity:.4f} | "
        f"Spacing score p25: {best.spacing_score:.4f} | "
        f"Preserved collected exact: {exact_preserved} | "
        f"Preserved collected nearest: {nearest_preserved} | "
        f"Best candidate sec: {best.elapsed_seconds:.2f} | "
        f"Total sec: {total_runtime:.2f} | "
        f"Output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
