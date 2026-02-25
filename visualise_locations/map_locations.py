#!/usr/bin/env python3
"""
Render an interactive map from the collected coffee-shop CSV dataset.

Data flow (high level):
1) Load CSV rows from collect_location_data/coffee_shops_with_reviews.csv.
2) Validate and extract latitude/longitude for each row.
3) Build a popup per valid point with place metadata and up to 5 review snippets.
4) Render the points on a Folium map and save locations_map.html.

Dependency:
    pip install folium
"""

from __future__ import annotations

import csv
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
INPUT_PATH = REPO_ROOT.parent / "collect_location_data" / "coffee_shops_with_reviews.csv"
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
                "popup_html": _build_popup_html(record, lat, lng),
            }
        )

    return points, skipped


def _build_map(points: list[dict[str, Any]]) -> folium.Map:
    """Create a Folium map, place all markers, and fit camera bounds to data."""
    lats = [point["lat"] for point in points]
    lngs = [point["lng"] for point in points]
    center = [sum(lats) / len(lats), sum(lngs) / len(lngs)]

    # Use OpenStreetMap tiles for a lightweight default base layer.
    point_map = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=10)

    for point in points:
        folium.CircleMarker(
            location=[point["lat"], point["lng"]],
            radius=3,
            color="#1f77b4",
            weight=1,
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=1.0,
            popup=folium.Popup(point["popup_html"], max_width=420),
        ).add_to(point_map)

    point_map.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])
    return point_map


def main() -> int:
    """Entrypoint for generating locations_map.html from the CSV input."""
    if not INPUT_PATH.exists():
        print(f"Error: input file not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    records = _load_records(INPUT_PATH)
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
        f"Total records: {len(records)} | "
        f"Plotted points: {len(points)} | "
        f"Skipped points: {skipped} | "
        f"Output: {OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
