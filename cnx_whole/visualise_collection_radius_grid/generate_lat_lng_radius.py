#!/usr/bin/env python3
"""
Generate 500m square-grid collection circles for Chiang Mai Province.

Dependency:
    pip install shapely

Pipeline:
1) Load/calculate the Chiang Mai province border GeoJSON.
2) Project the border into local meter space.
3) Build an axis-aligned 500m x 500m square grid over the border bounds.
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
from urllib import error, parse, request

try:
    from shapely.geometry import box, shape
    from shapely.ops import transform, unary_union
    from shapely.prepared import prep
except ImportError:  # pragma: no cover
    print(
        "Error: 'shapely' is not installed. Install it with: pip install shapely",
        file=sys.stderr,
    )
    raise SystemExit(1)


REPO_ROOT = Path(__file__).resolve().parent
BORDER_PATH = REPO_ROOT / "chiang_mai_province_border.geojson"
OUTPUT_PATH = REPO_ROOT / "lat_lng_radius.json"

GRID_SIZE_M = 500.0
OUTPUT_RADIUS_M = 500.0
EPS = 1e-9
CHIANG_MAI_BORDER_NAME = "Chiang Mai Province"
CHIANG_MAI_PROVINCE_RELATION_ID = "R1908771"
NOMINATIM_USER_AGENT = "BrewScape-DataCollection/1.0 (+local-map-script)"


def _fatal(message: str) -> "None":
    """Abort execution with a consistent error format."""
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _validate_geojson_like(data: Any) -> bool:
    """Cheap structural check for cached FeatureCollection border data."""
    if not isinstance(data, dict):
        return False
    if data.get("type") != "FeatureCollection":
        return False
    features = data.get("features")
    if not isinstance(features, list) or not features:
        return False
    first_feature = features[0]
    if not isinstance(first_feature, dict):
        return False
    geometry = first_feature.get("geometry")
    if not isinstance(geometry, dict):
        return False
    return geometry.get("type") in {"Polygon", "MultiPolygon"}


def _feature_collection_from_relation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Wrap one Nominatim relation row into a local cacheable FeatureCollection."""
    geometry = row.get("geojson")
    if not isinstance(geometry, dict):
        _fatal("Chiang Mai province relation is missing geometry")
    if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        _fatal(
            "Chiang Mai province relation has unsupported geometry type "
            f"{geometry.get('type')!r}"
        )

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": CHIANG_MAI_BORDER_NAME,
                    "source": "OpenStreetMap Nominatim",
                    "source_relations": [CHIANG_MAI_PROVINCE_RELATION_ID],
                    "source_relation_names": [CHIANG_MAI_BORDER_NAME],
                    "source_display_name": row.get("display_name", ""),
                },
                "geometry": geometry,
            }
        ],
    }


def _fetch_chiang_mai_province_geojson() -> dict[str, Any]:
    """Fetch the Chiang Mai province administrative border from Nominatim."""
    query = parse.urlencode(
        {
            "format": "jsonv2",
            "osm_ids": CHIANG_MAI_PROVINCE_RELATION_ID,
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
        _fatal(f"unable to download Chiang Mai province border: {exc}")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        _fatal(f"invalid border response JSON: {exc}")

    if not isinstance(data, list) or len(data) != 1:
        _fatal("Chiang Mai province border service returned unexpected results")

    row = data[0]
    if not isinstance(row, dict) or row.get("osm_type") != "relation":
        _fatal("Chiang Mai province lookup did not return a relation")

    osm_id = row.get("osm_id")
    if f"R{osm_id}" != CHIANG_MAI_PROVINCE_RELATION_ID:
        _fatal(
            "Chiang Mai province lookup returned the wrong relation "
            f"({row.get('osm_type')} {osm_id})"
        )

    return _feature_collection_from_relation_row(row)


def _load_border_geojson(path: Path) -> dict[str, Any]:
    """Load cached border when compatible; otherwise re-fetch and refresh cache."""
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _fatal(f"invalid cached border JSON in {path}: {exc}")

        if _validate_geojson_like(cached):
            feature = cached["features"][0]
            properties = feature.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}

            cached_relations = tuple(properties.get("source_relations", []))
            if cached_relations == (CHIANG_MAI_PROVINCE_RELATION_ID,):
                return cached

    downloaded = _fetch_chiang_mai_province_geojson()
    try:
        path.write_text(json.dumps(downloaded, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(
            f"Warning: unable to cache Chiang Mai province border to {path}: {exc}",
            file=sys.stderr,
        )
    return downloaded


def _load_border_geometry(path: Path):
    """Load and validate Chiang Mai province border geometry from GeoJSON."""
    data = _load_border_geojson(path)

    feature_geometries = []
    for feature in data.get("features", []):
        if not isinstance(feature, dict):
            _fatal("invalid feature structure in border GeoJSON")

        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            _fatal("feature geometry missing in border GeoJSON")
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            _fatal("expected polygon geometry in border GeoJSON")

        feature_geometries.append(shape(geometry))

    if not feature_geometries:
        _fatal("border geometry is empty")

    border_geom = unary_union(feature_geometries)
    if border_geom.is_empty:
        _fatal("border geometry is empty")

    if border_geom.geom_type == "GeometryCollection":
        polygon_parts = [
            geom
            for geom in border_geom.geoms
            if geom.geom_type in {"Polygon", "MultiPolygon"} and not geom.is_empty
        ]
        if not polygon_parts:
            _fatal("border geometry has no polygonal components")
        border_geom = unary_union(polygon_parts)

    return border_geom


@dataclass(frozen=True)
class LocalProjection:
    """Local equirectangular-like projection centered on the border centroid."""

    lng0: float
    lat0: float
    cos_lat0: float

    def to_lng_lat(self, x: float, y: float) -> tuple[float, float]:
        """Convert local meter coordinates back to geographic lon/lat."""
        lng = (x / (self.cos_lat0 * 111320.0)) + self.lng0
        lat = (y / 111320.0) + self.lat0
        return lng, lat

    def geom_to_xy(self, geom):
        """Project a Shapely geometry from lon/lat into local meter coordinates."""
        return transform(
            lambda x, y, z=None: (
                (x - self.lng0) * self.cos_lat0 * 111320.0,
                (y - self.lat0) * 111320.0,
            ),
            geom,
        )


def _build_projection(border_geom_lng_lat) -> LocalProjection:
    """Build a local meter-space projection around the border centroid."""
    centroid = border_geom_lng_lat.centroid
    lng0 = float(centroid.x)
    lat0 = float(centroid.y)
    return LocalProjection(
        lng0=lng0,
        lat0=lat0,
        cos_lat0=math.cos(math.radians(lat0)),
    )


def _iter_intersecting_grid_centers(border_xy_geom):
    """Yield center points for 500m cells that intersect the border geometry."""
    min_x, min_y, max_x, max_y = border_xy_geom.bounds
    start_x = math.floor(min_x / GRID_SIZE_M) * GRID_SIZE_M
    start_y = math.floor(min_y / GRID_SIZE_M) * GRID_SIZE_M
    end_x = math.ceil(max_x / GRID_SIZE_M) * GRID_SIZE_M
    end_y = math.ceil(max_y / GRID_SIZE_M) * GRID_SIZE_M

    prepared_border = prep(border_xy_geom)
    y = start_y
    while y < end_y - EPS:
        x = start_x
        while x < end_x - EPS:
            if prepared_border.intersects(box(x, y, x + GRID_SIZE_M, y + GRID_SIZE_M)):
                yield x + (GRID_SIZE_M / 2.0), y + (GRID_SIZE_M / 2.0)
            x += GRID_SIZE_M
        y += GRID_SIZE_M


def _build_output_rows(border_geom_lng_lat) -> tuple[list[dict[str, Any]], int]:
    """Create sibling-compatible rows from the intersecting square grid."""
    projection = _build_projection(border_geom_lng_lat)
    border_xy_geom = projection.geom_to_xy(border_geom_lng_lat)

    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, float]] = set()
    cell_count = 0
    for center_x, center_y in _iter_intersecting_grid_centers(border_xy_geom):
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
    border_geom_lng_lat = _load_border_geometry(BORDER_PATH)
    rows, cell_count = _build_output_rows(border_geom_lng_lat)

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
