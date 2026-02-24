#!/usr/bin/env python3
"""
Generate a map of collection circles from lat_lng_radius.json.

Dependency:
    pip install folium shapely
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

try:
    from shapely.geometry import Point, mapping, shape
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover
    print(
        "Error: 'shapely' is not installed. Install it with: pip install shapely",
        file=sys.stderr,
    )
    raise SystemExit(1)


REPO_ROOT = Path(__file__).resolve().parent
INPUT_PATH = REPO_ROOT / "lat_lng_radius.json"
OUTPUT_PATH = REPO_ROOT / "collection_radius_map.html"
BORDER_CACHE_PATH = REPO_ROOT / "chiang_mai_main_area_merged_border.geojson"

NOT_COLLECTED_COLOR = "#1f77b4"
COLLECTED_COLOR = "#2ca02c"
CHIANG_MAI_BORDER_COLOR = "#ff4d4d"
CHIANG_MAI_MAIN_AREA_NAME = "Chiang Mai Main Area (City + Suthep + Chang Phueak)"
CHIANG_MAI_OSM_RELATION_IDS = ("R18271830", "R19975357", "R19033670")
CHIANG_MAI_SUPPLEMENT_RELATION_ID = "R19033670"
CHIANG_MAI_SUPPLEMENT_ANCHOR_POINTS = (
    # lng, lat inside Chang Phueak area to capture only the missing district piece.
    (98.9674, 18.8108),
)
CHIANG_MAI_SOURCE_NAMES = {
    "R18271830": "Chiang Mai City Municipality",
    "R19975357": "Suthep Town Municipality",
    "R19033670": "Mueang Chiang Mai District",
}
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


def _fetch_chiang_mai_main_area_geojson() -> dict[str, Any]:
    query = parse.urlencode(
        {
            "format": "jsonv2",
            "osm_ids": ",".join(CHIANG_MAI_OSM_RELATION_IDS),
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
        print(f"Error: unable to download Chiang Mai main-area border: {exc}", file=sys.stderr)
        raise SystemExit(1)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid border response JSON: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not isinstance(data, list) or not data:
        print("Error: Chiang Mai main-area border service returned no results.", file=sys.stderr)
        raise SystemExit(1)

    rows_by_relation: dict[str, dict[str, Any]] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        if row.get("osm_type") != "relation":
            continue
        osm_id = row.get("osm_id")
        if not isinstance(osm_id, int):
            continue
        relation_id = f"R{osm_id}"
        rows_by_relation[relation_id] = row

    missing_relations = [rid for rid in CHIANG_MAI_OSM_RELATION_IDS if rid not in rows_by_relation]
    if missing_relations:
        print(
            "Error: missing required Chiang Mai boundary relation(s): "
            f"{', '.join(missing_relations)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    geometries_by_relation: dict[str, Any] = {}
    for relation_id in CHIANG_MAI_OSM_RELATION_IDS:
        row = rows_by_relation[relation_id]
        geometry = row.get("geojson")
        if not isinstance(geometry, dict):
            print(
                f"Error: geometry missing for required relation {relation_id}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            print(
                f"Error: relation {relation_id} has unsupported geometry type "
                f"{geometry.get('type')!r}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        geometries_by_relation[relation_id] = shape(geometry)

    base_geometries = [
        geometries_by_relation["R18271830"],
        geometries_by_relation["R19975357"],
    ]
    dissolved = unary_union(base_geometries)

    supplement = geometries_by_relation[CHIANG_MAI_SUPPLEMENT_RELATION_ID]
    supplement_missing = supplement.difference(dissolved)
    supplement_parts = []
    if supplement_missing.geom_type == "Polygon":
        supplement_parts = [supplement_missing]
    elif supplement_missing.geom_type == "MultiPolygon":
        supplement_parts = list(supplement_missing.geoms)
    elif supplement_missing.geom_type == "GeometryCollection":
        supplement_parts = [
            geom for geom in supplement_missing.geoms if geom.geom_type in {"Polygon", "MultiPolygon"}
        ]

    selected_supplement_parts = []
    for part in supplement_parts:
        if part.is_empty:
            continue
        if any(part.covers(Point(lng, lat)) for lng, lat in CHIANG_MAI_SUPPLEMENT_ANCHOR_POINTS):
            selected_supplement_parts.append(part)

    if selected_supplement_parts:
        dissolved = unary_union([dissolved, *selected_supplement_parts])

    if dissolved.is_empty:
        print("Error: dissolved Chiang Mai main-area geometry is empty.", file=sys.stderr)
        raise SystemExit(1)

    if dissolved.geom_type == "GeometryCollection":
        polygon_parts = [
            geom
            for geom in dissolved.geoms
            if geom.geom_type in {"Polygon", "MultiPolygon"} and not geom.is_empty
        ]
        if not polygon_parts:
            print(
                "Error: dissolved Chiang Mai geometry has no polygonal components.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        dissolved = unary_union(polygon_parts)

    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": CHIANG_MAI_MAIN_AREA_NAME,
                    "source": "OpenStreetMap Nominatim",
                    "source_relations": list(CHIANG_MAI_OSM_RELATION_IDS),
                    "source_relation_names": [
                        CHIANG_MAI_SOURCE_NAMES[rid] for rid in CHIANG_MAI_OSM_RELATION_IDS
                    ],
                    "supplement_relation": CHIANG_MAI_SUPPLEMENT_RELATION_ID,
                    "supplement_anchor_points_lng_lat": list(CHIANG_MAI_SUPPLEMENT_ANCHOR_POINTS),
                },
                "geometry": mapping(dissolved),
            }
        ],
    }
    return feature_collection


def _load_chiang_mai_main_area_geojson(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as file:
                cached = json.load(file)
        except json.JSONDecodeError as exc:
            print(f"Error: invalid cached border JSON in {path}: {exc}", file=sys.stderr)
            raise SystemExit(1)
        if _validate_geojson_like(cached):
            feature = cached["features"][0]
            properties = feature.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}

            cached_relations = tuple(properties.get("source_relations", []))
            cached_supplement = properties.get("supplement_relation")

            if (
                cached_relations == CHIANG_MAI_OSM_RELATION_IDS
                and cached_supplement == CHIANG_MAI_SUPPLEMENT_RELATION_ID
            ):
                return cached

    downloaded = _fetch_chiang_mai_main_area_geojson()
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(downloaded, file, ensure_ascii=True, indent=2)
    except OSError as exc:
        print(
            f"Warning: unable to cache Chiang Mai main-area border to {path}: {exc}",
            file=sys.stderr,
        )
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

    # Draw a white halo first for contrast, then the red dotted border on top.
    folium.GeoJson(
        chiang_mai_border,
        name=f"{CHIANG_MAI_MAIN_AREA_NAME} Border Halo",
        style_function=lambda _feature: {
            "color": "#ffffff",
            "weight": 7,
            "fill": False,
            "opacity": 1.0,
        },
    ).add_to(point_map)

    folium.GeoJson(
        chiang_mai_border,
        name=f"{CHIANG_MAI_MAIN_AREA_NAME} Border",
        style_function=lambda _feature: {
            "color": CHIANG_MAI_BORDER_COLOR,
            "weight": 3,
            "fill": False,
            "dashArray": "4, 4",
            "opacity": 1.0,
        },
        tooltip=f"{CHIANG_MAI_MAIN_AREA_NAME} Border",
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
    chiang_mai_border = _load_chiang_mai_main_area_geojson(BORDER_CACHE_PATH)

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
