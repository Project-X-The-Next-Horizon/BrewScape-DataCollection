#!/usr/bin/env python3
"""
Generate an interactive map from adaptive lat_lng_radius.json.

Circle color follows collected status:
    - collected = false -> blue
    - collected = true  -> green

Dependency:
    pip install folium
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any

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
BORDER_PATH = REPO_ROOT / "chiang_mai_main_area_merged_border.geojson"

NOT_COLLECTED_COLOR = "#1f77b4"
COLLECTED_COLOR = "#2ca02c"
CHIANG_MAI_BORDER_COLOR = "#ff4d4d"


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


def _load_json(path: Path) -> Any:
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)


def _validate_border_geojson(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        print("Error: border GeoJSON must be an object.", file=sys.stderr)
        raise SystemExit(1)
    if data.get("type") != "FeatureCollection":
        print("Error: border GeoJSON must be a FeatureCollection.", file=sys.stderr)
        raise SystemExit(1)
    features = data.get("features")
    if not isinstance(features, list) or not features:
        print("Error: border GeoJSON has no features.", file=sys.stderr)
        raise SystemExit(1)
    return data


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
    pairs = list(_iter_geojson_lng_lat_pairs(geojson))
    if not pairs:
        return None

    lats: list[float] = []
    lngs: list[float] = []
    for lng, lat in pairs:
        if -90 <= lat <= 90 and -180 <= lng <= 180:
            lats.append(lat)
            lngs.append(lng)

    if not lats or not lngs:
        return None
    return [[min(lats), min(lngs)], [max(lats), max(lngs)]]


def _build_popup_html(
    lat: float,
    lng: float,
    radius: float,
    collected: bool,
    population_density: float | None,
) -> str:
    collection_status = "Collected" if collected else "Not collected"
    density_text = "N/A" if population_density is None else f"{population_density:,.2f}"
    return (
        "<div style='font-family:Arial,sans-serif; font-size:12px; line-height:1.35;'>"
        "<table>"
        "<tr><th style='text-align:left; padding:2px 8px 2px 0;'>Status</th>"
        f"<td style='padding:2px 0;'>{escape(collection_status)}</td></tr>"
        "<tr><th style='text-align:left; padding:2px 8px 2px 0;'>Radius (m)</th>"
        f"<td style='padding:2px 0;'>{radius:.0f}</td></tr>"
        "<tr><th style='text-align:left; padding:2px 8px 2px 0;'>Population density</th>"
        f"<td style='padding:2px 0;'>{escape(density_text)} people/km^2</td></tr>"
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
        density = _to_float(record.get("population_density"))

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
                "population_density": density,
                "color": color,
            }
        )

    return points, skipped


def _build_map(points: list[dict[str, Any]], chiang_mai_border: dict[str, Any]) -> folium.Map:
    lats = [point["lat"] for point in points]
    lngs = [point["lng"] for point in points]
    center = [sum(lats) / len(lats), sum(lngs) / len(lngs)]

    point_map = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=12)

    for point in points:
        popup_html = _build_popup_html(
            lat=point["lat"],
            lng=point["lng"],
            radius=point["radius"],
            collected=point["collected"],
            population_density=point["population_density"],
        )

        folium.Circle(
            location=[point["lat"], point["lng"]],
            radius=point["radius"],
            color=point["color"],
            weight=2,
            fill=True,
            fill_color=point["color"],
            fill_opacity=0.25,
            popup=folium.Popup(popup_html, max_width=360),
        ).add_to(point_map)

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
        name="Chiang Mai Border Halo",
        style_function=lambda _feature: {
            "color": "#ffffff",
            "weight": 7,
            "fill": False,
            "opacity": 1.0,
        },
    ).add_to(point_map)

    folium.GeoJson(
        chiang_mai_border,
        name="Chiang Mai Border",
        style_function=lambda _feature: {
            "color": CHIANG_MAI_BORDER_COLOR,
            "weight": 3,
            "fill": False,
            "dashArray": "4, 4",
            "opacity": 1.0,
        },
        tooltip="Chiang Mai Border",
    ).add_to(point_map)

    bounds = _geojson_bounds(chiang_mai_border)
    if bounds is not None:
        point_map.fit_bounds(bounds)
    else:
        point_map.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])

    legend_html = (
        "<div style='"
        "position: fixed; bottom: 24px; left: 24px; z-index: 9999; "
        "background: white; border: 1px solid #bbb; border-radius: 6px; "
        "padding: 8px 10px; font-family: Arial, sans-serif; font-size: 12px;'>"
        "<div style='font-weight:600; margin-bottom:6px;'>Collection Status</div>"
        f"<div><span style='color:{NOT_COLLECTED_COLOR};'>&#9632;</span> Not collected</div>"
        f"<div><span style='color:{COLLECTED_COLOR};'>&#9632;</span> Collected</div>"
        "</div>"
    )
    point_map.get_root().html.add_child(folium.Element(legend_html))
    return point_map


def main() -> int:
    records = _load_json(INPUT_PATH)
    if not isinstance(records, list):
        print(f"Error: expected a JSON array in {INPUT_PATH}.", file=sys.stderr)
        return 1

    points, skipped = _extract_points(records)
    chiang_mai_border = _validate_border_geojson(_load_json(BORDER_PATH))

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
        f"Output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
