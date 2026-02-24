#!/usr/bin/env python3
"""
Generate a map of collection circles from lat_lng_radius.json.

Dependency:
    pip install folium
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any
from urllib import error, parse, request

try:
    import folium
except ImportError:  # pragma: no cover
    print(
        "Error: 'folium' is not installed. Install it with: pip install folium",
        file=sys.stderr,
    )
    raise SystemExit(1)


REPO_ROOT = Path(__file__).resolve().parent
INPUT_PATH = REPO_ROOT / "lat_lng_radius.json"
OUTPUT_PATH = REPO_ROOT / "collection_radius_map.html"
BORDER_CACHE_PATH = REPO_ROOT / "chiang_mai_province_border.geojson"

NOT_COLLECTED_COLOR = "#1f77b4"
COLLECTED_COLOR = "#2ca02c"
CHIANG_MAI_BORDER_COLOR = "#d62728"
CHIANG_MAI_OSM_RELATION_ID = "R1908771"
NOMINATIM_USER_AGENT = "BrewScape-DataCollection/1.0 (+local-map-script)"


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
    if isinstance(value, bool):
        return value
    return None


def _load_records(path: Path) -> list[Any]:
    if not path.exists():
        print(f"Error: input file not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not isinstance(data, list):
        print(f"Error: expected a JSON array in {path}.", file=sys.stderr)
        raise SystemExit(1)

    return data


def _validate_geojson_like(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    geo_type = data.get("type")
    return geo_type in {"Feature", "FeatureCollection", "Polygon", "MultiPolygon"}


def _fetch_chiang_mai_province_geojson() -> dict[str, Any]:
    query = parse.urlencode(
        {
            "format": "jsonv2",
            "osm_ids": CHIANG_MAI_OSM_RELATION_ID,
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
        print(f"Error: unable to download Chiang Mai border: {exc}", file=sys.stderr)
        raise SystemExit(1)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid border response JSON: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not isinstance(data, list) or not data:
        print("Error: Chiang Mai border service returned no results.", file=sys.stderr)
        raise SystemExit(1)

    border_row = data[0]
    if not isinstance(border_row, dict):
        print("Error: Chiang Mai border service returned an invalid record.", file=sys.stderr)
        raise SystemExit(1)

    geometry = border_row.get("geojson")
    if not isinstance(geometry, dict):
        print("Error: Chiang Mai border geometry is missing.", file=sys.stderr)
        raise SystemExit(1)

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Chiang Mai Province",
                    "osm_id": border_row.get("osm_id"),
                    "osm_type": border_row.get("osm_type"),
                    "source": "OpenStreetMap Nominatim",
                },
                "geometry": geometry,
            }
        ],
    }
    return feature_collection


def _load_chiang_mai_province_geojson(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as file:
                cached = json.load(file)
        except json.JSONDecodeError as exc:
            print(f"Error: invalid cached border JSON in {path}: {exc}", file=sys.stderr)
            raise SystemExit(1)
        if _validate_geojson_like(cached):
            return cached

    downloaded = _fetch_chiang_mai_province_geojson()
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(downloaded, file, ensure_ascii=True, indent=2)
    except OSError as exc:
        print(f"Warning: unable to cache Chiang Mai border to {path}: {exc}", file=sys.stderr)
    return downloaded


def _iter_lng_lat_pairs(value: Any):
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            yield float(value[0]), float(value[1])
        else:
            for item in value:
                yield from _iter_lng_lat_pairs(item)


def _iter_geojson_lng_lat_pairs(geojson: Any):
    if not isinstance(geojson, dict):
        return

    geo_type = geojson.get("type")

    if geo_type == "FeatureCollection":
        for feature in geojson.get("features", []):
            yield from _iter_geojson_lng_lat_pairs(feature)
        return

    if geo_type == "Feature":
        yield from _iter_geojson_lng_lat_pairs(geojson.get("geometry"))
        return

    if geo_type == "GeometryCollection":
        for geometry in geojson.get("geometries", []):
            yield from _iter_geojson_lng_lat_pairs(geometry)
        return

    if "coordinates" in geojson:
        yield from _iter_lng_lat_pairs(geojson.get("coordinates"))


def _geojson_bounds(geojson: Any) -> list[list[float]] | None:
    lng_lat_pairs = list(_iter_geojson_lng_lat_pairs(geojson))
    if not lng_lat_pairs:
        return None

    lats = []
    lngs = []
    for lng, lat in lng_lat_pairs:
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            lats.append(lat)
            lngs.append(lng)

    if not lats:
        return None

    return [[min(lats), min(lngs)], [max(lats), max(lngs)]]


def _build_popup_html(lat: float, lng: float, radius: float, collected: bool) -> str:
    collection_status = "Collected" if collected else "Not collected"
    return (
        "<div style='font-family:Arial,sans-serif; font-size:12px; line-height:1.35;'>"
        "<table>"
        "<tr><th style='text-align:left; padding:2px 8px 2px 0;'>Status</th>"
        f"<td style='padding:2px 0;'>{escape(collection_status)}</td></tr>"
        "<tr><th style='text-align:left; padding:2px 8px 2px 0;'>Radius (m)</th>"
        f"<td style='padding:2px 0;'>{radius:.2f}</td></tr>"
        "<tr><th style='text-align:left; padding:2px 8px 2px 0;'>Center</th>"
        f"<td style='padding:2px 0;'>{lat:.6f}, {lng:.6f}</td></tr>"
        "</table>"
        "</div>"
    )


def _extract_points(records: list[Any]) -> tuple[list[dict[str, Any]], int]:
    points: list[dict[str, Any]] = []
    skipped = 0

    for record in records:
        if not isinstance(record, dict):
            skipped += 1
            continue

        lat = _to_float(record.get("lat"))
        lng = _to_float(record.get("lng"))
        radius = _to_float(record.get("radius"))
        collected = _to_bool(record.get("collected"))

        if (
            lat is None
            or lng is None
            or radius is None
            or collected is None
            or not (-90 <= lat <= 90)
            or not (-180 <= lng <= 180)
            or radius <= 0
        ):
            skipped += 1
            continue

        color = COLLECTED_COLOR if collected else NOT_COLLECTED_COLOR

        points.append(
            {
                "lat": lat,
                "lng": lng,
                "radius": radius,
                "collected": collected,
                "color": color,
                "popup_html": _build_popup_html(lat, lng, radius, collected),
            }
        )

    return points, skipped


def _build_map(points: list[dict[str, Any]], chiang_mai_border: dict[str, Any]) -> folium.Map:
    lats = [point["lat"] for point in points]
    lngs = [point["lng"] for point in points]
    center = [sum(lats) / len(lats), sum(lngs) / len(lngs)]

    point_map = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=12)

    for point in points:
        folium.Circle(
            location=[point["lat"], point["lng"]],
            radius=point["radius"],
            color=point["color"],
            weight=2,
            fill=True,
            fill_color=point["color"],
            fill_opacity=0.25,
            popup=folium.Popup(point["popup_html"], max_width=360),
        ).add_to(point_map)

        # Small marker to clearly show each circle's center coordinate.
        folium.CircleMarker(
            location=[point["lat"], point["lng"]],
            radius=2,
            color=point["color"],
            weight=1,
            fill=True,
            fill_color=point["color"],
            fill_opacity=1.0,
        ).add_to(point_map)

    folium.GeoJson(
        chiang_mai_border,
        name="Chiang Mai Province Border",
        style_function=lambda _feature: {
            "color": CHIANG_MAI_BORDER_COLOR,
            "weight": 2,
            "fill": False,
        },
        tooltip="Chiang Mai Province Border",
    ).add_to(point_map)

    border_bounds = _geojson_bounds(chiang_mai_border)
    if border_bounds is not None:
        point_map.fit_bounds(border_bounds)
    else:
        point_map.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])

    return point_map


def main() -> int:
    records = _load_records(INPUT_PATH)
    points, skipped = _extract_points(records)
    chiang_mai_border = _load_chiang_mai_province_geojson(BORDER_CACHE_PATH)

    if not points:
        print(
            "Error: found zero valid entries in lat_lng_radius.json. "
            "Expected lat, lng, radius (> 0), and collected (boolean).",
            file=sys.stderr,
        )
        return 1

    point_map = _build_map(points, chiang_mai_border)
    point_map.save(str(OUTPUT_PATH))

    print(
        f"Total records: {len(records)} | "
        f"Plotted circles: {len(points)} | "
        f"Skipped rows: {skipped} | "
        f"Output: {OUTPUT_PATH} | "
        f"Border: {BORDER_CACHE_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
