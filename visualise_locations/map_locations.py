#!/usr/bin/env python3
"""
Render an interactive map from the collected coffee-shop CSV dataset.

Data flow (high level):
1) Load CSV rows from the current folder or collect_location_data/coffee_shops_with_reviews.csv.
2) Validate and extract latitude/longitude for each row.
3) Build a popup per valid point with place metadata and up to 5 review snippets.
4) Render the points on a Folium map, including a place-id search box, and save
   locations_map.html.

Dependency:
    pip install folium
"""

from __future__ import annotations

import csv
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
INPUT_PATH_CANDIDATES = (
    REPO_ROOT / "coffee_shops_with_reviews.csv",
    REPO_ROOT.parent / "collect_location_data" / "coffee_shops_with_reviews.csv",
)
OUTPUT_PATH = REPO_ROOT / "locations_map.html"


def _to_float(value: Any) -> float | None:
    """Best-effort float coercion for CSV values.

    Returns None for empty strings, booleans, and non-numeric values so callers can
    treat those records as invalid coordinates or optional fields.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _to_text(value: Any) -> str | None:
    """Normalize values to user-facing text for popup display."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _load_records(path: Path) -> list[dict[str, str]]:
    """Load the input CSV and enforce required coordinate columns.

    Side effects:
    - Exits the script with a descriptive error message when the file is invalid.
    """
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                print(f"Error: CSV file has no header row: {path}", file=sys.stderr)
                raise SystemExit(1)
            rows = [dict(row) for row in reader]
    except csv.Error as exc:
        print(f"Error: invalid CSV in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    # Coordinates are the only strict requirement for mapping.
    required = {"lat", "lon"}
    missing = [column for column in required if column not in (reader.fieldnames or [])]
    if missing:
        print(
            f"Error: CSV is missing required columns: {', '.join(sorted(missing))}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    return rows


def _resolve_input_path(candidates: tuple[Path, ...]) -> Path | None:
    """Return the first existing CSV path from the configured candidates."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _build_popup_html(record: dict[str, Any], lat: float, lng: float) -> str:
    """Build compact table-style HTML for a marker popup."""
    # Keep primary metadata at the top for fast scanning on click.
    rows = [
        ("Name", _to_text(record.get("name"))),
        ("Place ID", _to_text(record.get("place_id"))),
        ("Average Rating", _to_text(record.get("average_rating"))),
        ("Total Review Count", _to_text(record.get("total_review_count"))),
        (
            "Earliest Available Review Date",
            _to_text(record.get("earliest_available_review_date")),
        ),
        ("Coordinates", f"{lat:.6f}, {lng:.6f}"),
    ]

    rendered_rows = []
    for label, value in rows:
        if value is None:
            continue
        rendered_rows.append(
            "<tr>"
            f"<th style='text-align:left; padding:2px 8px 2px 0; white-space:nowrap;'>{escape(label)}</th>"
            f"<td style='padding:2px 0;'>{escape(value)}</td>"
            "</tr>"
        )

    # Add review text rows only when present to avoid verbose empty popups.
    for i in range(1, 6):
        review_text = _to_text(record.get(f"review_{i}_text"))
        if review_text is None:
            continue
        rendered_rows.append(
            "<tr>"
            f"<th style='text-align:left; padding:2px 8px 2px 0; white-space:nowrap;'>Review {i}</th>"
            f"<td style='padding:2px 0;'>{escape(review_text)}</td>"
            "</tr>"
        )

    return (
        "<div style='font-family:Arial,sans-serif; font-size:12px; line-height:1.35;'>"
        "<table>"
        f"{''.join(rendered_rows)}"
        "</table>"
        "</div>"
    )


def _extract_points(records: list[Any]) -> tuple[list[dict[str, Any]], int]:
    """Extract valid map points and count skipped rows.

    A row is skipped when coordinates are missing or out of world bounds.
    """
    points: list[dict[str, Any]] = []
    skipped = 0

    for record in records:
        # CSV stores coordinates in `lat` and `lon` columns.
        lat = _to_float(record.get("lat"))
        lng = _to_float(record.get("lon"))

        if lat is None or lng is None:
            skipped += 1
            continue

        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            skipped += 1
            continue

        points.append(
            {
                "lat": lat,
                "lng": lng,
                "name": _to_text(record.get("name")) or "",
                "place_id": _to_text(record.get("place_id")) or "",
                "popup_html": _build_popup_html(record, lat, lng),
            }
        )

    return points, skipped


def _add_location_count_overlay(
    point_map: folium.Map, points: list[dict[str, Any]]
) -> None:
    """Inject a fixed overlay that shows plotted and in-view location counts."""
    overlay_id = "location-count-overlay"
    count_id = "location-count-overlay-text"
    overlay_html = (
        "<div "
        f"id='{overlay_id}' "
        "style='"
        "position: fixed; bottom: 24px; right: 24px; z-index: 9999; "
        "background: white; border: 1px solid #bbb; border-radius: 6px; "
        "padding: 8px 10px; font-family: Arial, sans-serif; font-size: 12px; "
        "line-height: 1.35; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15); "
        "pointer-events: none;'>"
        f"<div id='{count_id}'>Total plotted: {len(points)} | In view: {len(points)}</div>"
        "</div>"
    )
    point_map.get_root().html.add_child(folium.Element(overlay_html))

    plotted_locations = json.dumps(
        [[point["lat"], point["lng"]] for point in points], separators=(",", ":")
    )
    map_name = point_map.get_name()
    script = f"""
    const plottedLocations = {plotted_locations};
    const locationCountText = document.getElementById("{count_id}");

    window.addEventListener("load", function() {{
        function updateLocationCounts() {{
            if (!locationCountText) {{
                return;
            }}

            const bounds = {map_name}.getBounds();
            const inViewCount = plottedLocations.reduce((count, location) => {{
                return count + (bounds.contains(L.latLng(location[0], location[1])) ? 1 : 0);
            }}, 0);

            locationCountText.textContent =
                `Total plotted: ${{plottedLocations.length}} | In view: ${{inViewCount}}`;
        }}

        {map_name}.on("moveend", updateLocationCounts);
        updateLocationCounts();
    }});
    """
    point_map.get_root().script.add_child(folium.Element(script))


def _add_place_id_search_overlay(
    point_map: folium.Map, points: list[dict[str, Any]], marker_names: list[str]
) -> None:
    """Inject a search box that jumps to a marker by place_id."""
    search_box_id = "place-id-search-box"
    search_button_id = "place-id-search-button"
    status_id = "place-id-search-status"
    overlay_html = (
        "<div "
        "style='"
        "position: fixed; top: 24px; left: 24px; z-index: 9999; width: 320px; "
        "background: rgba(255, 255, 255, 0.96); border: 1px solid #bbb; "
        "border-radius: 8px; padding: 10px 12px; "
        "box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15); "
        "font-family: Arial, sans-serif; font-size: 12px; line-height: 1.35;'>"
        "<div style='font-weight: 600; margin-bottom: 6px;'>Search by Place ID</div>"
        f"<input id='{search_box_id}' type='text' placeholder='Enter place_id' "
        "style='width: 100%; box-sizing: border-box; padding: 6px 8px; "
        "border: 1px solid #bbb; border-radius: 4px; margin-bottom: 8px;'>"
        f"<button id='{search_button_id}' type='button' "
        "style='width: 100%; padding: 6px 8px; border: 1px solid #2c5aa0; "
        "border-radius: 4px; background: #2c5aa0; color: white; cursor: pointer;'>"
        "Find location"
        "</button>"
        f"<div id='{status_id}' style='margin-top: 8px; min-height: 16px; color: #444; "
        "padding: 8px 10px; border-radius: 6px; background: #f7f7f7; border: 1px solid #e0e0e0;'>"
        "Enter a place_id to center the map on that marker."
        "</div>"
        "</div>"
    )
    point_map.get_root().html.add_child(folium.Element(overlay_html))

    searchable_entries = []
    for point, marker_name in zip(points, marker_names, strict=True):
        if not point["place_id"]:
            continue
        searchable_entries.append(
            "{"
            f"placeId:{json.dumps(point['place_id'])},"
            f"name:{json.dumps(point['name'])},"
            f"normalizedPlaceId:{json.dumps(point['place_id'].strip().lower())},"
            f"lat:{point['lat']},"
            f"lng:{point['lng']},"
            f"markerName:{json.dumps(marker_name)}"
            "}"
        )

    searchable_locations = "[\n" + ",\n".join(searchable_entries) + "\n]"
    map_name = point_map.get_name()
    script = f"""
    const searchableLocations = {searchable_locations};
    const defaultMarkerStyle = {{
        radius: 3,
        color: "#1f77b4",
        weight: 1,
        fillColor: "#1f77b4",
        fillOpacity: 1.0,
    }};
    const highlightedMarkerStyle = {{
        radius: 7,
        color: "#c0392b",
        weight: 2,
        fillColor: "#f39c12",
        fillOpacity: 1.0,
    }};

    window.addEventListener("load", function() {{
        const searchInput = document.getElementById("{search_box_id}");
        const searchButton = document.getElementById("{search_button_id}");
        const statusText = document.getElementById("{status_id}");
        let activeMarker = null;
        let activeHalo = null;

        function resolveMarker(location) {{
            if (!location || !location.markerName) {{
                return null;
            }}
            return globalThis[location.markerName] || null;
        }}

        function setStatus(message, tone) {{
            if (!statusText) {{
                return;
            }}

            statusText.textContent = message;
            if (tone === "error") {{
                statusText.style.background = "#fff1f0";
                statusText.style.borderColor = "#f1b0aa";
                statusText.style.color = "#8a1f17";
                return;
            }}

            if (tone === "success") {{
                statusText.style.background = "#eef8ec";
                statusText.style.borderColor = "#9dcca1";
                statusText.style.color = "#1f5f28";
                return;
            }}

            statusText.style.background = "#f7f7f7";
            statusText.style.borderColor = "#e0e0e0";
            statusText.style.color = "#444";
        }}

        function resetMarkerHighlight() {{
            if (!activeMarker) {{
                if (activeHalo) {{
                    {map_name}.removeLayer(activeHalo);
                    activeHalo = null;
                }}
                return;
            }}
            activeMarker.setStyle(defaultMarkerStyle);
            activeMarker.setRadius(defaultMarkerStyle.radius);
            activeMarker = null;
            if (activeHalo) {{
                {map_name}.removeLayer(activeHalo);
                activeHalo = null;
            }}
        }}

        function highlightMarker(location, marker) {{
            resetMarkerHighlight();
            marker.setStyle(highlightedMarkerStyle);
            marker.setRadius(highlightedMarkerStyle.radius);
            activeMarker = marker;
            activeHalo = L.circle([location.lat, location.lng], {{
                radius: 45,
                color: "#d35400",
                weight: 2,
                fillColor: "#f1c40f",
                fillOpacity: 0.18,
                interactive: false,
            }}).addTo({map_name});
            marker.bringToFront();
        }}

        function runSearch() {{
            if (!searchInput || !statusText) {{
                return;
            }}

            const rawQuery = searchInput.value.trim();
            const query = rawQuery.toLowerCase();
            if (!query) {{
                resetMarkerHighlight();
                setStatus("Enter a place_id to search.", "neutral");
                return;
            }}

            const exactMatches = searchableLocations.filter(function(location) {{
                return location.normalizedPlaceId === query;
            }});
            const matches = exactMatches.length > 0
                ? exactMatches
                : searchableLocations.filter(function(location) {{
                    return location.normalizedPlaceId.includes(query);
                }});

            if (matches.length === 0) {{
                resetMarkerHighlight();
                setStatus(`No place_id match for "${{rawQuery}}".`, "error");
                return;
            }}

            const selected = matches[0];
            const marker = resolveMarker(selected);
            if (!marker) {{
                resetMarkerHighlight();
                setStatus(
                    "Matched a place_id, but the marker could not be loaded. Refresh the page and try again.",
                    "error",
                );
                return;
            }}

            highlightMarker(selected, marker);
            {map_name}.setView([selected.lat, selected.lng], Math.max({map_name}.getZoom(), 16), {{
                animate: true,
            }});
            marker.openPopup();

            if (matches.length === 1) {{
                setStatus(selected.name
                    ? `Showing ${{selected.placeId}} (${{selected.name}}).`
                    : `Showing ${{selected.placeId}}.`, "success");
                return;
            }}

            setStatus(selected.name
                ? `Showing first of ${{matches.length}} matches: ${{selected.placeId}} (${{selected.name}}).`
                : `Showing first of ${{matches.length}} matches: ${{selected.placeId}}.`, "success");
        }}

        if (searchButton) {{
            searchButton.addEventListener("click", runSearch);
        }}

        if (searchInput) {{
            searchInput.addEventListener("keydown", function(event) {{
                if (event.key !== "Enter") {{
                    return;
                }}
                event.preventDefault();
                runSearch();
            }});
        }}
    }});
    """
    point_map.get_root().script.add_child(folium.Element(script))


def _build_map(points: list[dict[str, Any]]) -> folium.Map:
    """Create a Folium map, place all markers, and fit camera bounds to data."""
    lats = [point["lat"] for point in points]
    lngs = [point["lng"] for point in points]
    center = [sum(lats) / len(lats), sum(lngs) / len(lngs)]

    # Use OpenStreetMap tiles for a lightweight default base layer.
    point_map = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=10)
    marker_names: list[str] = []

    for point in points:
        marker = folium.CircleMarker(
            location=[point["lat"], point["lng"]],
            radius=3,
            color="#1f77b4",
            weight=1,
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=1.0,
            popup=folium.Popup(point["popup_html"], max_width=420),
        )
        marker.add_to(point_map)
        marker_names.append(marker.get_name())

    point_map.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])
    _add_location_count_overlay(point_map, points)
    _add_place_id_search_overlay(point_map, points, marker_names)
    return point_map


def main() -> int:
    """Entrypoint for generating locations_map.html from the CSV input."""
    input_path = _resolve_input_path(INPUT_PATH_CANDIDATES)
    if input_path is None:
        checked_paths = ", ".join(str(path) for path in INPUT_PATH_CANDIDATES)
        print(
            f"Error: input file not found. Checked: {checked_paths}",
            file=sys.stderr,
        )
        return 1

    records = _load_records(input_path)
    points, skipped = _extract_points(records)

    if not points:
        print(
            "Error: found zero valid coordinates in coffee_shops_with_reviews.csv. "
            "Expected lat and lon columns.",
            file=sys.stderr,
        )
        return 1

    point_map = _build_map(points)
    point_map.save(str(OUTPUT_PATH))

    print(
        f"Input: {input_path} | "
        f"Total records: {len(records)} | "
        f"Plotted points: {len(points)} | "
        f"Skipped points: {skipped} | "
        f"Output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
