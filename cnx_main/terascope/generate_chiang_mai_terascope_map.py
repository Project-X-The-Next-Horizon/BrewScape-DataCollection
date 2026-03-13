#!/usr/bin/env python3
"""
Generate Chiang Mai-only 100 m Terascope outputs from ESA WorldCover GeoTIFF data.

Outputs:
1) chiang_mai_terascope_100m_cells.csv
2) chiang_mai_terascope_map.html

Dependencies:
    pip install folium shapely tifffile imagecodecs numpy branca

Pipeline:
1) Load the Chiang Mai border geometry from the existing project GeoJSON.
2) Build a local meter-space projection and a shared 100 m grid.
3) Clip each Terascope TIFF to the Chiang Mai bounds.
4) Keep only source pixels whose centers intersect the border.
5) Aggregate native pixels into 100 m cells and render an interactive map.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

import folium
import numpy as np
import tifffile
from branca.colormap import LinearColormap
from shapely import intersects_xy
from shapely.geometry import mapping, shape


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
BORDER_PATH = REPO_ROOT.parent / "population_density" / "chiang_mai_main_area_merged_border.geojson"
OUTPUT_CSV_PATH = REPO_ROOT / "chiang_mai_terascope_100m_cells.csv"
OUTPUT_HTML_PATH = REPO_ROOT / "chiang_mai_terascope_map.html"

GRID_SIZE_M = 100.0
GRID_INDEX_EPS = 1e-9
METERS_PER_DEGREE = 111320.0

LAND_COVER_LAYER_NAME = "Land Cover"
CHIANG_MAI_BORDER_COLOR = "#ff4d4d"
CHIANG_MAI_BORDER_HALO_COLOR = "#ffffff"

ESA_WORLD_COVER_FALLBACK = (
    (10, "Tree cover", "#006400"),
    (20, "Shrubland", "#ffbb22"),
    (30, "Grassland", "#ffff4c"),
    (40, "Cropland", "#f096ff"),
    (50, "Built-up", "#fa0000"),
    (60, "Bare/sparse vegetation", "#b4b4b4"),
    (70, "Snow and ice", "#f0f0f0"),
    (80, "Permanent water bodies", "#0064c8"),
    (90, "Herbaceous wetland", "#0096a0"),
    (95, "Mangroves", "#00cf75"),
    (100, "Moss and lichen", "#fae6a0"),
)
ESA_CLASS_CODES = tuple(code for code, _label, _color in ESA_WORLD_COVER_FALLBACK)
ESA_CLASS_RANK = {code: idx for idx, code in enumerate(ESA_CLASS_CODES)}


@dataclass(frozen=True)
class LocalProjection:
    """Local equirectangular-like projection centered on the border centroid."""

    lng0: float
    lat0: float
    cos_lat0: float

    def to_xy(self, lng: float, lat: float) -> tuple[float, float]:
        """Convert geographic lon/lat into local meter coordinates."""
        x = (lng - self.lng0) * self.cos_lat0 * METERS_PER_DEGREE
        y = (lat - self.lat0) * METERS_PER_DEGREE
        return x, y

    def to_lng_lat(self, x: float, y: float) -> tuple[float, float]:
        """Convert local meter coordinates back to geographic lon/lat."""
        lng = (x / (self.cos_lat0 * METERS_PER_DEGREE)) + self.lng0
        lat = (y / METERS_PER_DEGREE) + self.lat0
        return lng, lat


@dataclass(frozen=True)
class GridDefinition:
    """Shared 100 m grid definition in local meter space."""

    origin_x: float
    origin_y: float
    grid_size_m: float
    projection: LocalProjection


@dataclass(frozen=True)
class LandCoverClass:
    """Display metadata for one ESA WorldCover class."""

    code: int
    label: str
    color: str


@dataclass(frozen=True)
class QualityLayerSpec:
    """Display metadata for one input-quality layer."""

    band_index: int
    name: str
    mean_key: str
    count_key: str
    mean_label: str
    palette: tuple[str, ...]
    legend_title: str
    legend_description: str
    value_suffix: str = ""
    decimals: int = 1


@dataclass
class GridCellAggregate:
    """Mutable per-cell aggregation buffers."""

    row: int
    col: int
    class_counts: dict[int, int] = field(default_factory=dict)
    quality_sums: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quality_counts: list[int] = field(default_factory=lambda: [0, 0, 0])


QUALITY_LAYER_SPECS = (
    QualityLayerSpec(
        band_index=0,
        name="Sentinel-1 Observations",
        mean_key="quality_band_1_mean",
        count_key="quality_band_1_count",
        mean_label="Mean Sentinel-1 observations",
        palette=("#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"),
        legend_title="Sentinel-1 Observations",
        legend_description="Mean number of Sentinel-1 GAMMA0 observations per 100 m cell.",
    ),
    QualityLayerSpec(
        band_index=1,
        name="Sentinel-2 Observations",
        mean_key="quality_band_2_mean",
        count_key="quality_band_2_count",
        mean_label="Mean Sentinel-2 observations",
        palette=("#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"),
        legend_title="Sentinel-2 Observations",
        legend_description="Mean number of Sentinel-2 L2A observations per 100 m cell.",
    ),
    QualityLayerSpec(
        band_index=2,
        name="Invalid Sentinel-2 %",
        mean_key="quality_band_3_mean",
        count_key="quality_band_3_count",
        mean_label="Mean invalid Sentinel-2 %",
        palette=("#ffffcc", "#fed976", "#fd8d3c", "#f03b20", "#bd0026"),
        legend_title="Invalid Sentinel-2 %",
        legend_description="Mean percent of invalid Sentinel-2 observations per 100 m cell.",
        value_suffix="%",
    ),
)


def _fatal(message: str) -> "None":
    """Abort execution with a consistent error format."""
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _default_land_cover_classes() -> dict[int, LandCoverClass]:
    """Return the built-in ESA WorldCover fallback legend."""
    return {
        code: LandCoverClass(code=code, label=label, color=color)
        for code, label, color in ESA_WORLD_COVER_FALLBACK
    }


def _load_border_geometry(path: Path):
    """Load and validate the Chiang Mai border polygon from GeoJSON."""
    if not path.exists():
        _fatal(f"border file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fatal(f"invalid JSON in border file {path}: {exc}")

    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        _fatal("border GeoJSON must be a FeatureCollection")

    features = payload.get("features")
    if not isinstance(features, list) or not features:
        _fatal("border GeoJSON has no features")

    geometry = features[0].get("geometry") if isinstance(features[0], dict) else None
    if not isinstance(geometry, dict):
        _fatal("border feature geometry is missing or invalid")

    border = shape(geometry)
    if border.is_empty:
        _fatal("border geometry is empty")
    return border


def _build_projection(border_geometry) -> LocalProjection:
    """Build a local meter-space projection around the border centroid."""
    centroid = border_geometry.centroid
    if centroid.is_empty:
        min_lon, min_lat, max_lon, max_lat = border_geometry.bounds
        lng0 = (min_lon + max_lon) / 2.0
        lat0 = (min_lat + max_lat) / 2.0
    else:
        lng0 = float(centroid.x)
        lat0 = float(centroid.y)
    return LocalProjection(
        lng0=lng0,
        lat0=lat0,
        cos_lat0=math.cos(math.radians(lat0)),
    )


def _build_grid_definition(
    border_geometry,
    projection: LocalProjection,
    grid_size_m: float,
) -> GridDefinition:
    """Build the shared 100 m grid anchored to the Chiang Mai bounds."""
    min_lon, min_lat, _max_lon, _max_lat = border_geometry.bounds
    min_x, min_y = projection.to_xy(min_lon, min_lat)
    origin_x = math.floor(min_x / grid_size_m) * grid_size_m
    origin_y = math.floor(min_y / grid_size_m) * grid_size_m
    return GridDefinition(
        origin_x=origin_x,
        origin_y=origin_y,
        grid_size_m=grid_size_m,
        projection=projection,
    )


def _find_raster_paths(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """Resolve the land-cover and input-quality TIFF files."""
    if not data_dir.exists():
        _fatal(f"data directory not found: {data_dir}")

    land_cover_paths = sorted(data_dir.rglob("*_Map.tif"))
    quality_paths = sorted(data_dir.rglob("*InputQuality.tif"))

    if not land_cover_paths:
        _fatal(f"no land-cover TIFFs were found under {data_dir}")
    if not quality_paths:
        _fatal(f"no input-quality TIFFs were found under {data_dir}")

    return land_cover_paths, quality_paths


def _parse_nodata(tag_value: Any) -> float | None:
    """Parse GDAL_NODATA tag values that can arrive in multiple scalar formats."""
    if tag_value is None:
        return None
    if isinstance(tag_value, bytes):
        tag_value = tag_value.decode("utf-8", errors="ignore")
    if isinstance(tag_value, str):
        text = tag_value.strip().strip("\x00")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    if isinstance(tag_value, (int, float)):
        return float(tag_value)
    return None


def _compute_window(
    border_bounds: tuple[float, float, float, float],
    raster_width: int,
    raster_height: int,
    origin_lon: float,
    origin_lat: float,
    pixel_width: float,
    pixel_height: float,
) -> tuple[int, int, int, int]:
    """Convert border lon/lat bounds to clamped raster row/col bounds."""
    min_lon, min_lat, max_lon, max_lat = border_bounds

    col_min = math.floor((min_lon - origin_lon) / pixel_width)
    col_max = math.ceil((max_lon - origin_lon) / pixel_width) - 1
    row_min = math.floor((origin_lat - max_lat) / pixel_height)
    row_max = math.ceil((origin_lat - min_lat) / pixel_height) - 1

    col_min = max(0, col_min)
    row_min = max(0, row_min)
    col_max = min(raster_width - 1, col_max)
    row_max = min(raster_height - 1, row_max)

    if col_min > col_max or row_min > row_max:
        _fatal("Chiang Mai border does not overlap a raster extent")

    return row_min, row_max, col_min, col_max


def _load_clipped_window(
    path: Path,
    border_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Load one TIFF, clip it to the Chiang Mai bbox, and return window centers."""
    if not path.exists():
        _fatal(f"raster file not found: {path}")

    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            _fatal(f"raster TIFF has no pages: {path}")

        page = tif.pages[0]
        scale_tag = page.tags.get("ModelPixelScaleTag")
        tie_tag = page.tags.get("ModelTiepointTag")
        if scale_tag is None or tie_tag is None:
            _fatal(
                f"raster {path} is missing required GeoTIFF tags "
                "(ModelPixelScaleTag/ModelTiepointTag)"
            )

        scale = scale_tag.value
        tie = tie_tag.value
        if not isinstance(scale, tuple) or len(scale) < 2:
            _fatal(f"invalid ModelPixelScaleTag values in {path}")
        if not isinstance(tie, tuple) or len(tie) < 6:
            _fatal(f"invalid ModelTiepointTag values in {path}")

        pixel_width = float(scale[0])
        pixel_height = float(scale[1])
        origin_lon = float(tie[3])
        origin_lat = float(tie[4])
        if pixel_width <= 0 or pixel_height <= 0:
            _fatal(f"unexpected non-positive pixel scale in {path}")

        nodata_tag = page.tags.get("GDAL_NODATA")
        nodata_value = _parse_nodata(None if nodata_tag is None else nodata_tag.value)

        try:
            raster = page.asarray()
        except Exception as exc:  # pragma: no cover
            _fatal(f"unable to read raster pixels from {path}: {exc}")

    if raster.ndim == 2:
        raster_height, raster_width = raster.shape
    elif raster.ndim == 3:
        _bands, raster_height, raster_width = raster.shape
    else:
        _fatal(f"unsupported raster shape {raster.shape!r} in {path}")

    row_min, row_max, col_min, col_max = _compute_window(
        border_bounds=border_bounds,
        raster_width=raster_width,
        raster_height=raster_height,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )

    if raster.ndim == 2:
        window = np.asarray(raster[row_min : row_max + 1, col_min : col_max + 1]).copy()
    else:
        window = np.asarray(raster[:, row_min : row_max + 1, col_min : col_max + 1]).copy()
    del raster
    gc.collect()

    lon_centers = origin_lon + ((np.arange(col_min, col_max + 1, dtype=np.float64) + 0.5) * pixel_width)
    lat_centers = origin_lat - ((np.arange(row_min, row_max + 1, dtype=np.float64) + 0.5) * pixel_height)
    return window, lon_centers, lat_centers, nodata_value


def _legend_labels_from_metadata(metadata_text: str | None) -> dict[int, str]:
    """Extract class labels from GDAL metadata XML."""
    if not metadata_text:
        return {}
    try:
        root = ET.fromstring(metadata_text)
    except ET.ParseError:
        return {}

    legend_text = ""
    for item in root.findall("Item"):
        if item.attrib.get("name") == "legend":
            legend_text = item.text or ""
            break

    labels: dict[int, str] = {}
    for raw_line in legend_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            code = int(parts[0])
        except ValueError:
            continue
        label = parts[1].strip() if len(parts) > 1 else ""
        if label:
            labels[code] = label
    return labels


def _palette_color_from_colormap(colormap: np.ndarray | None, code: int) -> str | None:
    """Extract an 8-bit RGB hex color from the TIFF color palette."""
    if colormap is None or code < 0 or code >= colormap.shape[1]:
        return None
    red = int(colormap[0, code] / 257)
    green = int(colormap[1, code] / 257)
    blue = int(colormap[2, code] / 257)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _load_land_cover_classes(path: Path) -> dict[int, LandCoverClass]:
    """Load the land-cover legend and colors from the first map TIFF."""
    classes = _default_land_cover_classes()
    if not path.exists():
        return classes

    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            return classes
        page = tif.pages[0]
        metadata_tag = page.tags.get("GDAL_METADATA")
        metadata_text = None if metadata_tag is None else str(metadata_tag.value)
        parsed_labels = _legend_labels_from_metadata(metadata_text)
        colormap = page.colormap

    for code in ESA_CLASS_CODES:
        fallback = classes[code]
        label = parsed_labels.get(code, fallback.label)
        color = _palette_color_from_colormap(colormap, code) or fallback.color
        classes[code] = LandCoverClass(code=code, label=label, color=color)

    return classes


def _grid_cell_ids_for_lon_lat(
    lon_values: np.ndarray,
    lat_values: np.ndarray,
    grid_definition: GridDefinition,
) -> np.ndarray:
    """Vectorize 100 m cell assignment for arrays of point centers."""
    projection = grid_definition.projection
    x_values = (lon_values - projection.lng0) * projection.cos_lat0 * METERS_PER_DEGREE
    y_values = (lat_values - projection.lat0) * METERS_PER_DEGREE

    col_values = np.floor(
        ((x_values - grid_definition.origin_x) / grid_definition.grid_size_m) + GRID_INDEX_EPS
    ).astype(np.int64)
    row_values = np.floor(
        ((y_values - grid_definition.origin_y) / grid_definition.grid_size_m) + GRID_INDEX_EPS
    ).astype(np.int64)
    return (row_values << 32) | (col_values & np.int64(0xFFFFFFFF))


def _decode_cell_id(cell_id: int) -> tuple[int, int]:
    """Convert a packed 64-bit cell identifier back into row/col indices."""
    return int(cell_id >> 32), int(cell_id & 0xFFFFFFFF)


def _ensure_cell(
    cells: dict[tuple[int, int], GridCellAggregate],
    row: int,
    col: int,
) -> GridCellAggregate:
    """Create a cell accumulator on demand and return it."""
    key = (row, col)
    cell = cells.get(key)
    if cell is None:
        cell = GridCellAggregate(row=row, col=col)
        cells[key] = cell
    return cell


def _accumulate_land_cover_samples(
    cells: dict[tuple[int, int], GridCellAggregate],
    cell_ids: np.ndarray,
    values: np.ndarray,
) -> None:
    """Add flattened land-cover samples into per-cell class counts."""
    if cell_ids.size == 0:
        return

    unique_ids, inverse = np.unique(cell_ids.astype(np.int64, copy=False), return_inverse=True)
    for code in np.unique(values.astype(np.int16, copy=False)).tolist():
        mask = values == code
        counts = np.bincount(inverse[mask], minlength=unique_ids.size)
        for idx in np.flatnonzero(counts):
            row, col = _decode_cell_id(int(unique_ids[idx]))
            cell = _ensure_cell(cells, row, col)
            cell.class_counts[int(code)] = cell.class_counts.get(int(code), 0) + int(counts[idx])


def _accumulate_quality_samples(
    cells: dict[tuple[int, int], GridCellAggregate],
    cell_ids: np.ndarray,
    band_values: np.ndarray,
    band_index: int,
    nodata_value: float | None,
) -> None:
    """Add flattened quality samples into per-cell mean buffers."""
    if cell_ids.size == 0:
        return

    values = band_values.astype(np.float64, copy=False)
    valid = np.isfinite(values)
    if nodata_value is not None:
        valid &= values != nodata_value
    if not np.any(valid):
        return

    valid_ids = cell_ids[valid].astype(np.int64, copy=False)
    valid_values = values[valid]
    unique_ids, inverse = np.unique(valid_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=valid_values, minlength=unique_ids.size)
    counts = np.bincount(inverse, minlength=unique_ids.size)

    for idx in np.flatnonzero(counts):
        row, col = _decode_cell_id(int(unique_ids[idx]))
        cell = _ensure_cell(cells, row, col)
        cell.quality_sums[band_index] += float(sums[idx])
        cell.quality_counts[band_index] += int(counts[idx])


def _accumulate_land_cover_window(
    cells: dict[tuple[int, int], GridCellAggregate],
    raster_window: np.ndarray,
    lon_centers: np.ndarray,
    lat_centers: np.ndarray,
    border_geometry,
    grid_definition: GridDefinition,
    nodata_value: float | None,
) -> int:
    """Aggregate one land-cover raster window into the shared 100 m grid."""
    if raster_window.ndim != 2:
        _fatal(f"expected a 2-D land-cover raster window, got shape {raster_window.shape!r}")
    if raster_window.shape != (lat_centers.size, lon_centers.size):
        _fatal("land-cover window shape does not match center coordinate vectors")

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    inside_mask = intersects_xy(border_geometry, lon_grid, lat_grid)
    if nodata_value is not None:
        inside_mask &= raster_window != nodata_value
    if not np.any(inside_mask):
        return 0

    selected_lon = lon_grid[inside_mask]
    selected_lat = lat_grid[inside_mask]
    selected_values = raster_window[inside_mask]
    cell_ids = _grid_cell_ids_for_lon_lat(selected_lon, selected_lat, grid_definition)
    _accumulate_land_cover_samples(cells, cell_ids, selected_values)
    return int(selected_values.size)


def _accumulate_quality_window(
    cells: dict[tuple[int, int], GridCellAggregate],
    raster_window: np.ndarray,
    lon_centers: np.ndarray,
    lat_centers: np.ndarray,
    border_geometry,
    grid_definition: GridDefinition,
    nodata_value: float | None,
) -> int:
    """Aggregate one three-band quality raster window into the shared 100 m grid."""
    if raster_window.ndim != 3 or raster_window.shape[0] != 3:
        _fatal(f"expected a 3-band quality raster window, got shape {raster_window.shape!r}")
    if raster_window.shape[1:] != (lat_centers.size, lon_centers.size):
        _fatal("quality window shape does not match center coordinate vectors")

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
    inside_mask = intersects_xy(border_geometry, lon_grid, lat_grid)
    if not np.any(inside_mask):
        return 0

    selected_lon = lon_grid[inside_mask]
    selected_lat = lat_grid[inside_mask]
    cell_ids = _grid_cell_ids_for_lon_lat(selected_lon, selected_lat, grid_definition)

    for band_index in range(raster_window.shape[0]):
        _accumulate_quality_samples(
            cells=cells,
            cell_ids=cell_ids,
            band_values=raster_window[band_index][inside_mask],
            band_index=band_index,
            nodata_value=nodata_value,
        )
    return int(cell_ids.size)


def _aggregate_terascope_cells(
    border_geometry,
    grid_definition: GridDefinition,
    land_cover_paths: list[Path],
    quality_paths: list[Path],
) -> dict[tuple[int, int], GridCellAggregate]:
    """Aggregate all Terascope TIFFs into the shared 100 m grid."""
    cells: dict[tuple[int, int], GridCellAggregate] = {}
    border_bounds = border_geometry.bounds

    for path in land_cover_paths:
        window, lon_centers, lat_centers, nodata_value = _load_clipped_window(path, border_bounds)
        _accumulate_land_cover_window(
            cells=cells,
            raster_window=window,
            lon_centers=lon_centers,
            lat_centers=lat_centers,
            border_geometry=border_geometry,
            grid_definition=grid_definition,
            nodata_value=nodata_value,
        )
        del window, lon_centers, lat_centers
        gc.collect()

    for path in quality_paths:
        window, lon_centers, lat_centers, nodata_value = _load_clipped_window(path, border_bounds)
        _accumulate_quality_window(
            cells=cells,
            raster_window=window,
            lon_centers=lon_centers,
            lat_centers=lat_centers,
            border_geometry=border_geometry,
            grid_definition=grid_definition,
            nodata_value=nodata_value,
        )
        del window, lon_centers, lat_centers
        gc.collect()

    return cells


def _cell_bounds_lng_lat(
    row: int,
    col: int,
    grid_definition: GridDefinition,
) -> tuple[float, float, float, float, float, float]:
    """Convert one projected 100 m grid cell back into lon/lat bounds and center."""
    x_left = grid_definition.origin_x + (col * grid_definition.grid_size_m)
    x_right = x_left + grid_definition.grid_size_m
    y_bottom = grid_definition.origin_y + (row * grid_definition.grid_size_m)
    y_top = y_bottom + grid_definition.grid_size_m

    lon_left, lat_bottom = grid_definition.projection.to_lng_lat(x_left, y_bottom)
    lon_right, lat_top = grid_definition.projection.to_lng_lat(x_right, y_top)
    lon_center, lat_center = grid_definition.projection.to_lng_lat(
        (x_left + x_right) / 2.0,
        (y_bottom + y_top) / 2.0,
    )
    return lon_left, lon_right, lat_bottom, lat_top, lon_center, lat_center


def _dominant_class(
    class_counts: dict[int, int],
    classes: dict[int, LandCoverClass],
) -> tuple[int | None, str, float]:
    """Resolve the dominant land-cover class label and percentage."""
    total = sum(class_counts.values())
    if total <= 0:
        return None, "No land cover data", 0.0

    dominant_code = max(
        ESA_CLASS_CODES,
        key=lambda code: (class_counts.get(code, 0), -ESA_CLASS_RANK[code]),
    )
    dominant_count = class_counts.get(dominant_code, 0)
    return dominant_code, classes[dominant_code].label, (dominant_count / total) * 100.0


def _top_class_summary(
    class_counts: dict[int, int],
    classes: dict[int, LandCoverClass],
    total: int,
    limit: int = 3,
) -> str:
    """Render a compact top-N summary for tooltips."""
    if total <= 0:
        return "No land cover data"

    ranked_codes = sorted(
        (code for code in ESA_CLASS_CODES if class_counts.get(code, 0) > 0),
        key=lambda code: (-class_counts[code], ESA_CLASS_RANK[code]),
    )
    parts = []
    for code in ranked_codes[:limit]:
        share_pct = (class_counts[code] / total) * 100.0
        parts.append(f"{classes[code].label} {share_pct:.1f}%")
    return ", ".join(parts)


def _cell_has_any_data(cell: GridCellAggregate) -> bool:
    """Return True when a cell received any valid source samples."""
    return bool(cell.class_counts) or any(count > 0 for count in cell.quality_counts)


def _build_records(
    cells: dict[tuple[int, int], GridCellAggregate],
    grid_definition: GridDefinition,
    classes: dict[int, LandCoverClass],
) -> list[dict[str, Any]]:
    """Finalize aggregated cells into stable record dictionaries."""
    records: list[dict[str, Any]] = []

    for cell in sorted(cells.values(), key=lambda item: (-item.row, item.col)):
        if not _cell_has_any_data(cell):
            continue

        lon_left, lon_right, lat_bottom, lat_top, lon_center, lat_center = _cell_bounds_lng_lat(
            cell.row,
            cell.col,
            grid_definition,
        )
        land_cover_pixel_count = sum(cell.class_counts.values())
        dominant_code, dominant_label, dominant_share_pct = _dominant_class(cell.class_counts, classes)

        record: dict[str, Any] = {
            "grid_row": cell.row,
            "grid_col": cell.col,
            "lon_center": lon_center,
            "lat_center": lat_center,
            "lon_left": lon_left,
            "lon_right": lon_right,
            "lat_bottom": lat_bottom,
            "lat_top": lat_top,
            "dominant_land_cover_code": dominant_code,
            "dominant_land_cover_label": dominant_label,
            "dominant_land_cover_share_pct": dominant_share_pct,
            "land_cover_pixel_count": land_cover_pixel_count,
            "top_three_land_cover_summary": _top_class_summary(
                cell.class_counts,
                classes,
                land_cover_pixel_count,
            ),
        }

        for code in ESA_CLASS_CODES:
            class_count = cell.class_counts.get(code, 0)
            share_pct = ((class_count / land_cover_pixel_count) * 100.0) if land_cover_pixel_count else 0.0
            record[f"land_cover_{code}_count"] = class_count
            record[f"land_cover_{code}_share_pct"] = share_pct

        for spec in QUALITY_LAYER_SPECS:
            count = cell.quality_counts[spec.band_index]
            mean_value = (cell.quality_sums[spec.band_index] / count) if count else None
            record[spec.mean_key] = mean_value
            record[spec.count_key] = count

        records.append(record)

    return records


def _csv_fieldnames() -> list[str]:
    """Return the fixed output CSV header order."""
    fieldnames = [
        "grid_row",
        "grid_col",
        "lon_center",
        "lat_center",
        "lon_left",
        "lon_right",
        "lat_bottom",
        "lat_top",
        "dominant_land_cover_code",
        "dominant_land_cover_label",
        "dominant_land_cover_share_pct",
        "land_cover_pixel_count",
    ]
    for code in ESA_CLASS_CODES:
        fieldnames.extend([f"land_cover_{code}_count", f"land_cover_{code}_share_pct"])
    for spec in QUALITY_LAYER_SPECS:
        fieldnames.extend([spec.mean_key, spec.count_key])
    return fieldnames


def _format_csv_value(fieldname: str, value: Any) -> str | int:
    """Format CSV values using stable numeric precision."""
    integer_fields = {
        "grid_row",
        "grid_col",
        "dominant_land_cover_code",
        "land_cover_pixel_count",
        *(f"land_cover_{code}_count" for code in ESA_CLASS_CODES),
        *(spec.count_key for spec in QUALITY_LAYER_SPECS),
    }

    if value is None:
        return ""
    if fieldname in integer_fields:
        return int(value)
    if fieldname in {"lon_center", "lat_center", "lon_left", "lon_right", "lat_bottom", "lat_top"}:
        return f"{float(value):.8f}"
    if fieldname.endswith("_share_pct") or fieldname.endswith("_mean"):
        return f"{float(value):.6f}"
    return str(value)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write the finalized 100 m grid cells to CSV."""
    fieldnames = _csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: _format_csv_value(name, record.get(name)) for name in fieldnames})


def _compute_display_scale(values: list[float]) -> tuple[float, float]:
    """Compute a clamped display scale using 5th/95th percentiles."""
    min_value = min(values)
    max_value = max(values)
    if len(values) >= 5:
        p_low = float(np.percentile(values, 5))
        p_high = float(np.percentile(values, 95))
        if p_high > p_low:
            scale_min = p_low
            scale_max = p_high
        else:
            scale_min = min_value
            scale_max = max_value
    else:
        scale_min = min_value
        scale_max = max_value

    if math.isclose(scale_min, scale_max):
        scale_min = min_value
        scale_max = min_value + 1.0
    return scale_min, scale_max


def _clamped_color(colormap: LinearColormap, value: float, scale_min: float, scale_max: float) -> str:
    """Map one scalar value into a clamped colormap hex string."""
    clamped = min(max(value, scale_min), scale_max)
    return str(colormap(clamped))


def _land_cover_feature(record: dict[str, Any], classes: dict[int, LandCoverClass]) -> dict[str, Any] | None:
    """Build one GeoJSON feature for the land-cover layer."""
    dominant_code = record.get("dominant_land_cover_code")
    if dominant_code is None or int(record["land_cover_pixel_count"]) <= 0:
        return None

    dominant_code = int(dominant_code)
    feature_properties = {
        "fill_color": classes[dominant_code].color,
        "dominant_class": classes[dominant_code].label,
        "dominant_share_pct": f"{float(record['dominant_land_cover_share_pct']):.1f}%",
        "top_three_summary": str(record["top_three_land_cover_summary"]),
        "pixel_count": int(record["land_cover_pixel_count"]),
        "cell_center": f"{float(record['lat_center']):.6f}, {float(record['lon_center']):.6f}",
    }
    return {
        "type": "Feature",
        "properties": feature_properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [float(record["lon_left"]), float(record["lat_bottom"])],
                [float(record["lon_right"]), float(record["lat_bottom"])],
                [float(record["lon_right"]), float(record["lat_top"])],
                [float(record["lon_left"]), float(record["lat_top"])],
                [float(record["lon_left"]), float(record["lat_bottom"])],
            ]],
        },
    }


def _quality_feature(
    record: dict[str, Any],
    spec: QualityLayerSpec,
    colormap: LinearColormap,
    scale_min: float,
    scale_max: float,
) -> dict[str, Any] | None:
    """Build one GeoJSON feature for a continuous quality layer."""
    count = int(record[spec.count_key])
    mean_value = record[spec.mean_key]
    if count <= 0 or mean_value is None:
        return None

    mean_value = float(mean_value)
    if spec.value_suffix:
        mean_display = f"{mean_value:.{spec.decimals}f}{spec.value_suffix}"
    else:
        mean_display = f"{mean_value:.{spec.decimals}f}"

    return {
        "type": "Feature",
        "properties": {
            "fill_color": _clamped_color(colormap, mean_value, scale_min, scale_max),
            "mean_value": mean_display,
            "pixel_count": count,
            "cell_center": f"{float(record['lat_center']):.6f}, {float(record['lon_center']):.6f}",
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [float(record["lon_left"]), float(record["lat_bottom"])],
                [float(record["lon_right"]), float(record["lat_bottom"])],
                [float(record["lon_right"]), float(record["lat_top"])],
                [float(record["lon_left"]), float(record["lat_top"])],
                [float(record["lon_left"]), float(record["lat_bottom"])],
            ]],
        },
    }


def _rectangle_style(feature: dict[str, Any]) -> dict[str, Any]:
    """Style one grid-cell polygon using its precomputed fill color."""
    fill_color = feature["properties"]["fill_color"]
    return {
        "color": fill_color,
        "weight": 0,
        "fillColor": fill_color,
        "fillOpacity": 0.68,
    }


def _rectangle_highlight(_feature: dict[str, Any]) -> dict[str, Any]:
    """Use a subtle stroke highlight to improve hover feedback."""
    return {
        "color": "#222222",
        "weight": 1,
        "fillOpacity": 0.82,
    }


def _format_legend_value(value: float, spec: QualityLayerSpec) -> str:
    """Format continuous legend labels consistently with tooltip precision."""
    text = f"{value:.{spec.decimals}f}"
    return text + spec.value_suffix


def _build_land_cover_legend_html(classes: dict[int, LandCoverClass]) -> str:
    """Render the categorical legend content for the land-cover layer."""
    items = []
    for code in ESA_CLASS_CODES:
        info = classes[code]
        items.append(
            "<div style='display:flex; align-items:center; gap:8px; margin-top:4px;'>"
            f"<span style='display:inline-block; width:12px; height:12px; border:1px solid #888; "
            f"background:{escape(info.color)};'></span>"
            f"<span>{escape(info.label)}</span>"
            "</div>"
        )
    return (
        "<div style='font-weight:600; margin-bottom:6px;'>Land Cover</div>"
        "<div style='color:#555; margin-bottom:6px;'>"
        "Dominant ESA WorldCover class per 100 m cell."
        "</div>"
        f"{''.join(items)}"
    )


def _build_quality_legend_html(
    spec: QualityLayerSpec,
    scale_min: float,
    scale_max: float,
) -> str:
    """Render the legend content for one continuous quality layer."""
    gradient = ", ".join(spec.palette)
    return (
        f"<div style='font-weight:600; margin-bottom:6px;'>{escape(spec.legend_title)}</div>"
        f"<div style='color:#555; margin-bottom:6px;'>{escape(spec.legend_description)}</div>"
        "<div style='height:10px; border:1px solid #bbb; border-radius:4px; "
        f"background: linear-gradient(90deg, {gradient});'></div>"
        "<div style='display:flex; justify-content:space-between; margin-top:4px;'>"
        f"<span>{escape(_format_legend_value(scale_min, spec))}</span>"
        f"<span>{escape(_format_legend_value(scale_max, spec))}</span>"
        "</div>"
        "<div style='color:#666; margin-top:6px;'>"
        "Colors are clamped to the 5th-95th percentile display range."
        "</div>"
    )


def _add_theme_legend_and_behavior(
    terascope_map: folium.Map,
    theme_groups: dict[str, folium.FeatureGroup],
    legend_html_by_theme: dict[str, str],
) -> None:
    """Inject one fixed legend plus JS that makes overlays behave like radio buttons."""
    overlay_html = (
        "<div style='"
        "position: fixed; bottom: 24px; left: 24px; z-index: 9999; "
        "background: rgba(255, 255, 255, 0.96); border: 1px solid #bbb; "
        "border-radius: 8px; padding: 10px 12px; max-width: 320px; "
        "box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15); "
        "font-family: Arial, sans-serif; font-size: 12px; line-height: 1.35;'>"
        "<div id='terascope-legend-content'></div>"
        "</div>"
    )
    terascope_map.get_root().html.add_child(folium.Element(overlay_html))

    map_name = terascope_map.get_name()
    theme_layers_js = ",\n".join(
        f"        {json.dumps(name)}: {group.get_name()}" for name, group in theme_groups.items()
    )
    script = f"""
    const terascopeLegendHtml = {json.dumps(legend_html_by_theme)};

    window.addEventListener("load", function() {{
        const terascopeMap = {map_name};
        const themeLayers = {{
{theme_layers_js}
        }};
        const legendRoot = document.getElementById("terascope-legend-content");

        function setLegend(themeName) {{
            if (!legendRoot) {{
                return;
            }}
            legendRoot.innerHTML = terascopeLegendHtml[themeName] || "";
        }}

        function activeThemeName() {{
            for (const [themeName, layer] of Object.entries(themeLayers)) {{
                if (terascopeMap.hasLayer(layer)) {{
                    return themeName;
                }}
            }}
            return null;
        }}

        function enforceSingleTheme(activeName) {{
            for (const [themeName, layer] of Object.entries(themeLayers)) {{
                if (themeName !== activeName && terascopeMap.hasLayer(layer)) {{
                    terascopeMap.removeLayer(layer);
                }}
            }}

            if (themeLayers[activeName] && !terascopeMap.hasLayer(themeLayers[activeName])) {{
                terascopeMap.addLayer(themeLayers[activeName]);
            }}
            setLegend(activeName);
        }}

        terascopeMap.on("overlayadd", function(event) {{
            for (const [themeName, layer] of Object.entries(themeLayers)) {{
                if (layer === event.layer) {{
                    enforceSingleTheme(themeName);
                    return;
                }}
            }}
        }});

        terascopeMap.on("overlayremove", function(event) {{
            for (const [themeName, layer] of Object.entries(themeLayers)) {{
                if (layer === event.layer) {{
                    const nextActiveTheme = activeThemeName();
                    if (nextActiveTheme) {{
                        setLegend(nextActiveTheme);
                    }} else {{
                        enforceSingleTheme({json.dumps(LAND_COVER_LAYER_NAME)});
                    }}
                    return;
                }}
            }}
        }});

        enforceSingleTheme(activeThemeName() || {json.dumps(LAND_COVER_LAYER_NAME)});
    }});
    """
    terascope_map.get_root().script.add_child(folium.Element(script))


def _build_map(
    records: list[dict[str, Any]],
    border_geometry,
    classes: dict[int, LandCoverClass],
) -> folium.Map:
    """Render the multi-layer Terascope map using one 100 m GeoJSON grid per theme."""
    if not records:
        _fatal("no Terascope records are available to render")

    min_lon, min_lat, max_lon, max_lat = border_geometry.bounds
    center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
    terascope_map = folium.Map(
        location=center,
        tiles="OpenStreetMap",
        zoom_start=12,
        prefer_canvas=True,
    )

    theme_groups: dict[str, folium.FeatureGroup] = {}
    legend_html_by_theme = {
        LAND_COVER_LAYER_NAME: _build_land_cover_legend_html(classes),
    }

    land_cover_group = folium.FeatureGroup(
        name=LAND_COVER_LAYER_NAME,
        overlay=True,
        control=True,
        show=True,
    )
    land_cover_features = [
        feature
        for record in records
        for feature in [_land_cover_feature(record, classes)]
        if feature is not None
    ]
    folium.GeoJson(
        data={"type": "FeatureCollection", "features": land_cover_features},
        name=LAND_COVER_LAYER_NAME,
        style_function=_rectangle_style,
        highlight_function=_rectangle_highlight,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "dominant_class",
                "dominant_share_pct",
                "top_three_summary",
                "pixel_count",
                "cell_center",
            ],
            aliases=[
                "Dominant class",
                "Dominant share",
                "Top 3 classes",
                "Native pixels",
                "Cell center",
            ],
            labels=True,
            sticky=True,
        ),
    ).add_to(land_cover_group)
    land_cover_group.add_to(terascope_map)
    theme_groups[LAND_COVER_LAYER_NAME] = land_cover_group

    for spec in QUALITY_LAYER_SPECS:
        values = [
            float(record[spec.mean_key])
            for record in records
            if record.get(spec.mean_key) is not None and int(record[spec.count_key]) > 0
        ]
        if not values:
            continue

        scale_min, scale_max = _compute_display_scale(values)
        colormap = LinearColormap(colors=list(spec.palette), vmin=scale_min, vmax=scale_max)
        legend_html_by_theme[spec.name] = _build_quality_legend_html(spec, scale_min, scale_max)

        quality_group = folium.FeatureGroup(
            name=spec.name,
            overlay=True,
            control=True,
            show=False,
        )
        quality_features = [
            feature
            for record in records
            for feature in [_quality_feature(record, spec, colormap, scale_min, scale_max)]
            if feature is not None
        ]
        folium.GeoJson(
            data={"type": "FeatureCollection", "features": quality_features},
            name=spec.name,
            style_function=_rectangle_style,
            highlight_function=_rectangle_highlight,
            tooltip=folium.GeoJsonTooltip(
                fields=["mean_value", "pixel_count", "cell_center"],
                aliases=[spec.mean_label, "Contributing pixels", "Cell center"],
                labels=True,
                sticky=True,
            ),
        ).add_to(quality_group)
        quality_group.add_to(terascope_map)
        theme_groups[spec.name] = quality_group

    folium.GeoJson(
        {
            "type": "Feature",
            "properties": {"name": "Chiang Mai Border Halo"},
            "geometry": mapping(border_geometry),
        },
        style_function=lambda _feature: {
            "color": CHIANG_MAI_BORDER_HALO_COLOR,
            "weight": 7,
            "fill": False,
            "opacity": 1.0,
        },
    ).add_to(terascope_map)

    folium.GeoJson(
        {
            "type": "Feature",
            "properties": {"name": "Chiang Mai Border"},
            "geometry": mapping(border_geometry),
        },
        style_function=lambda _feature: {
            "color": CHIANG_MAI_BORDER_COLOR,
            "weight": 3,
            "fill": False,
            "dashArray": "4, 4",
            "opacity": 1.0,
        },
        tooltip="Chiang Mai Border",
    ).add_to(terascope_map)

    folium.LayerControl(collapsed=True).add_to(terascope_map)
    terascope_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    _add_theme_legend_and_behavior(terascope_map, theme_groups, legend_html_by_theme)
    return terascope_map


def main() -> int:
    """Entrypoint for building the Chiang Mai Terascope CSV and HTML map."""
    border_geometry = _load_border_geometry(BORDER_PATH)
    projection = _build_projection(border_geometry)
    grid_definition = _build_grid_definition(border_geometry, projection, GRID_SIZE_M)
    land_cover_paths, quality_paths = _find_raster_paths(DATA_DIR)
    classes = _load_land_cover_classes(land_cover_paths[0])

    cells = _aggregate_terascope_cells(
        border_geometry=border_geometry,
        grid_definition=grid_definition,
        land_cover_paths=land_cover_paths,
        quality_paths=quality_paths,
    )
    records = _build_records(cells, grid_definition, classes)
    if not records:
        print(
            "Error: no Terascope cells were found inside the Chiang Mai border.",
            file=sys.stderr,
        )
        return 1

    _write_csv(OUTPUT_CSV_PATH, records)
    terascope_map = _build_map(records, border_geometry, classes)
    terascope_map.save(str(OUTPUT_HTML_PATH))

    print(
        f"Grid cells: {len(records)} | "
        f"Land-cover TIFFs: {len(land_cover_paths)} | "
        f"Quality TIFFs: {len(quality_paths)} | "
        f"CSV output: {OUTPUT_CSV_PATH} | "
        f"Map output: {OUTPUT_HTML_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
