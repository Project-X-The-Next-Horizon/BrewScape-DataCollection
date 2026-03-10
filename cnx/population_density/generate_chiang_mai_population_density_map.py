#!/usr/bin/env python3
"""
Generate Chiang Mai-only population-density outputs from Thailand GeoTIFF data.

Outputs:
1) chiang_mai_population_density_cells.csv
2) chiang_mai_population_density_map.html

Dependencies:
    pip install folium shapely tifffile imagecodecs numpy branca

Pipeline:
1) Read Chiang Mai border geometry from GeoJSON.
2) Read georeferenced raster data (population density per cell).
3) Convert border bounds to raster row/col window.
4) Keep only cells whose centers fall inside the border.
5) Export filtered cells to CSV and render a heat-style map.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import folium
import numpy as np
import tifffile
from branca.colormap import LinearColormap
from shapely.geometry import Point, mapping, shape


REPO_ROOT = Path(__file__).resolve().parent
# Source data inputs.
RASTER_PATH = REPO_ROOT / "thailand_2020_population_data.tif"
BORDER_PATH = REPO_ROOT / "chiang_mai_main_area_merged_border.geojson"
# Generated outputs.
OUTPUT_CSV_PATH = REPO_ROOT / "chiang_mai_population_density_cells.csv"
OUTPUT_HTML_PATH = REPO_ROOT / "chiang_mai_population_density_map.html"


def _load_border_geometry(path: Path):
    """Load and validate the Chiang Mai border polygon from GeoJSON."""
    if not path.exists():
        print(f"Error: border file not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in border file {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        print("Error: border GeoJSON must be a FeatureCollection.", file=sys.stderr)
        raise SystemExit(1)

    features = payload.get("features")
    if not isinstance(features, list) or not features:
        print("Error: border GeoJSON has no features.", file=sys.stderr)
        raise SystemExit(1)

    feature = features[0]
    if not isinstance(feature, dict):
        print("Error: invalid first feature in border GeoJSON.", file=sys.stderr)
        raise SystemExit(1)

    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        print("Error: border feature geometry is missing or invalid.", file=sys.stderr)
        raise SystemExit(1)

    border = shape(geometry)
    if border.is_empty:
        print("Error: border geometry is empty.", file=sys.stderr)
        raise SystemExit(1)

    return border


def _parse_nodata(tag_value: Any) -> float | None:
    """Parse GDAL_NODATA tag values that can arrive in multiple scalar formats."""
    if tag_value is None:
        return None

    if isinstance(tag_value, bytes):
        text = tag_value.decode("utf-8", errors="ignore").strip().strip("\x00")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

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


def _load_raster(path: Path) -> tuple[np.ndarray, float | None, float, float, float, float]:
    """Load the first GeoTIFF band and required georeferencing metadata.

    Returns:
    - raster array
    - nodata sentinel (if defined)
    - raster origin lon/lat
    - pixel width/height in degrees
    """
    if not path.exists():
        print(f"Error: raster file not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    with tifffile.TiffFile(path) as tif:
        if not tif.pages:
            print("Error: raster TIFF has no pages.", file=sys.stderr)
            raise SystemExit(1)

        page = tif.pages[0]
        try:
            raster = page.asarray()
        except Exception as exc:  # pragma: no cover
            print(f"Error: unable to read raster pixels from {path}: {exc}", file=sys.stderr)
            raise SystemExit(1)

        # GeoTIFF georeferencing comes from pixel scale + tiepoint tags.
        scale_tag = page.tags.get("ModelPixelScaleTag")
        tie_tag = page.tags.get("ModelTiepointTag")
        if scale_tag is None or tie_tag is None:
            print(
                "Error: raster is missing required GeoTIFF tags "
                "(ModelPixelScaleTag/ModelTiepointTag).",
                file=sys.stderr,
            )
            raise SystemExit(1)

        scale = scale_tag.value
        tie = tie_tag.value
        if not isinstance(scale, tuple) or len(scale) < 2:
            print("Error: invalid ModelPixelScaleTag values.", file=sys.stderr)
            raise SystemExit(1)
        if not isinstance(tie, tuple) or len(tie) < 6:
            print("Error: invalid ModelTiepointTag values.", file=sys.stderr)
            raise SystemExit(1)

        pixel_width = float(scale[0])
        pixel_height = float(scale[1])
        origin_lon = float(tie[3])
        origin_lat = float(tie[4])
        if pixel_width <= 0 or pixel_height <= 0:
            print("Error: unexpected non-positive pixel scale in raster.", file=sys.stderr)
            raise SystemExit(1)

        # Nodata means "no valid population value" and should be skipped later.
        nodata_tag = page.tags.get("GDAL_NODATA")
        nodata_value = _parse_nodata(None if nodata_tag is None else nodata_tag.value)

    return raster, nodata_value, origin_lon, origin_lat, pixel_width, pixel_height


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

    # Transform world coordinates into raster indices.
    col_min = math.floor((min_lon - origin_lon) / pixel_width)
    col_max = math.ceil((max_lon - origin_lon) / pixel_width) - 1
    row_min = math.floor((origin_lat - max_lat) / pixel_height)
    row_max = math.ceil((origin_lat - min_lat) / pixel_height) - 1

    # Clamp the computed window to raster dimensions.
    col_min = max(0, col_min)
    row_min = max(0, row_min)
    col_max = min(raster_width - 1, col_max)
    row_max = min(raster_height - 1, row_max)

    if col_min > col_max or row_min > row_max:
        print(
            "Error: Chiang Mai border does not overlap the raster extent.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return row_min, row_max, col_min, col_max


def _extract_cells(
    raster: np.ndarray,
    border_geometry,
    nodata_value: float | None,
    origin_lon: float,
    origin_lat: float,
    pixel_width: float,
    pixel_height: float,
) -> list[dict[str, float | int]]:
    """Extract valid raster cells whose centers fall inside the border polygon."""
    if raster.ndim != 2:
        print(
            f"Error: expected a single-band raster, but got shape {raster.shape!r}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    raster_height, raster_width = raster.shape
    row_min, row_max, col_min, col_max = _compute_window(
        border_geometry.bounds,
        raster_width=raster_width,
        raster_height=raster_height,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )

    records: list[dict[str, float | int]] = []

    for row in range(row_min, row_max + 1):
        # Convert row index back to geographic coordinates for this pixel band.
        lat_top = origin_lat - (row * pixel_height)
        lat_bottom = lat_top - pixel_height
        lat_center = (lat_top + lat_bottom) / 2.0

        for col in range(col_min, col_max + 1):
            value = float(raster[row, col])
            if not math.isfinite(value):
                continue
            if nodata_value is not None and value == nodata_value:
                continue

            # Compute cell bounds/center in lon-lat.
            lon_left = origin_lon + (col * pixel_width)
            lon_right = lon_left + pixel_width
            lon_center = (lon_left + lon_right) / 2.0

            # Center-point inclusion keeps selection deterministic at borders.
            if not border_geometry.covers(Point(lon_center, lat_center)):
                continue

            records.append(
                {
                    "row": row,
                    "col": col,
                    "lon_center": lon_center,
                    "lat_center": lat_center,
                    "lon_left": lon_left,
                    "lon_right": lon_right,
                    "lat_bottom": lat_bottom,
                    "lat_top": lat_top,
                    "population_density": value,
                }
            )

    return records


def _write_csv(path: Path, records: list[dict[str, float | int]]) -> None:
    """Write filtered cells to CSV using stable numeric formatting."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "row",
                "col",
                "lon_center",
                "lat_center",
                "lon_left",
                "lon_right",
                "lat_bottom",
                "lat_top",
                "population_density",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    int(record["row"]),
                    int(record["col"]),
                    f"{float(record['lon_center']):.8f}",
                    f"{float(record['lat_center']):.8f}",
                    f"{float(record['lon_left']):.8f}",
                    f"{float(record['lon_right']):.8f}",
                    f"{float(record['lat_bottom']):.8f}",
                    f"{float(record['lat_top']):.8f}",
                    f"{float(record['population_density']):.6f}",
                ]
            )


def _build_map(records: list[dict[str, float | int]], border_geometry) -> folium.Map:
    """Render a choropleth-like rectangle grid over Chiang Mai border."""
    densities = [float(record["population_density"]) for record in records]
    min_density = min(densities)
    max_density = max(densities)

    # Clip the color scale to robust percentiles to reduce outlier dominance.
    if len(densities) >= 5:
        percentile_low = float(np.percentile(densities, 2))
        percentile_high = float(np.percentile(densities, 98))
        if percentile_high > percentile_low:
            scale_min = percentile_low
            scale_max = percentile_high
        else:
            scale_min = min_density
            scale_max = max_density
    else:
        scale_min = min_density
        scale_max = max_density

    # Avoid degenerate colormap ranges in near-constant data.
    if math.isclose(scale_min, scale_max):
        scale_min = min_density
        scale_max = min_density + 1.0

    colormap = LinearColormap(
        colors=["#f7fcf0", "#ccebc5", "#7bccc4", "#2b8cbe", "#084081"],
        vmin=scale_min,
        vmax=scale_max,
        caption="Population Density (people per km^2)",
    )

    min_lon, min_lat, max_lon, max_lat = border_geometry.bounds
    center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
    density_map = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=12)

    for record in records:
        value = float(record["population_density"])
        color = colormap(value)
        tooltip = (
            f"Density: {value:,.2f} people/km^2<br>"
            f"Cell center: {float(record['lat_center']):.5f}, {float(record['lon_center']):.5f}"
        )

        # Draw each raster cell as a colored rectangle in map coordinates.
        folium.Rectangle(
            bounds=[
                [float(record["lat_bottom"]), float(record["lon_left"])],
                [float(record["lat_top"]), float(record["lon_right"])],
            ],
            color=None,
            weight=0,
            fill=True,
            fill_color=color,
            fill_opacity=0.65,
            tooltip=tooltip,
        ).add_to(density_map)

    # Overlay border outline for geographic context.
    folium.GeoJson(
        {
            "type": "Feature",
            "properties": {"name": "Chiang Mai Main Area"},
            "geometry": mapping(border_geometry),
        },
        style_function=lambda _feature: {
            "color": "#ff4d4d",
            "weight": 2,
            "fill": False,
            "opacity": 1.0,
        },
        tooltip="Chiang Mai border",
    ).add_to(density_map)

    colormap.add_to(density_map)
    density_map.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return density_map


def main() -> int:
    """Entrypoint: build Chiang Mai cell CSV and preview map from source raster."""
    border_geometry = _load_border_geometry(BORDER_PATH)
    (
        raster,
        nodata_value,
        origin_lon,
        origin_lat,
        pixel_width,
        pixel_height,
    ) = _load_raster(RASTER_PATH)

    records = _extract_cells(
        raster=raster,
        border_geometry=border_geometry,
        nodata_value=nodata_value,
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
    )

    if not records:
        print(
            "Error: no raster cells were found inside Chiang Mai border.",
            file=sys.stderr,
        )
        return 1

    _write_csv(OUTPUT_CSV_PATH, records)
    density_map = _build_map(records, border_geometry)
    density_map.save(str(OUTPUT_HTML_PATH))

    print(
        f"Raster cells in Chiang Mai: {len(records)} | "
        f"Data output: {OUTPUT_CSV_PATH} | "
        f"Map output: {OUTPUT_HTML_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
