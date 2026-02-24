#!/usr/bin/env python3
"""
Plot all locations from data.json onto an interactive HTML map.

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
INPUT_PATH = REPO_ROOT / "data.json"
OUTPUT_PATH = REPO_ROOT / "locations_map.html"


def _to_float(value: Any) -> float | None:
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
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _load_records(path: Path) -> list[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if not isinstance(data, list):
        print(f"Error: expected a JSON array in {path}.", file=sys.stderr)
        raise SystemExit(1)

    return data


def _build_popup_html(record: dict[str, Any], lat: float, lng: float) -> str:
    address = _to_text(record.get("address"))
    if address is None:
        address_parts = [
            _to_text(record.get("street")),
            _to_text(record.get("city")),
            _to_text(record.get("state")),
            _to_text(record.get("postalCode")),
            _to_text(record.get("countryCode")),
        ]
        address = ", ".join(part for part in address_parts if part)

    rows = [
        ("Name", _to_text(record.get("title"))),
        ("Category", _to_text(record.get("categoryName"))),
        ("Address", address or None),
        ("Rating", _to_text(record.get("totalScore"))),
        ("Reviews", _to_text(record.get("reviewsCount"))),
        ("Price", _to_text(record.get("price"))),
        ("Phone", _to_text(record.get("phone"))),
        ("Website", _to_text(record.get("website"))),
        ("Place ID", _to_text(record.get("placeId"))),
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

    return (
        "<div style='font-family:Arial,sans-serif; font-size:12px; line-height:1.35;'>"
        "<table>"
        f"{''.join(rendered_rows)}"
        "</table>"
        "</div>"
    )


def _extract_points(records: list[Any]) -> tuple[list[dict[str, Any]], int]:
    points: list[dict[str, Any]] = []
    skipped = 0

    for record in records:
        lat = None
        lng = None

        if isinstance(record, dict):
            location = record.get("location")
            if isinstance(location, dict):
                lat = _to_float(location.get("lat"))
                lng = _to_float(location.get("lng"))

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
    lats = [point["lat"] for point in points]
    lngs = [point["lng"] for point in points]
    center = [sum(lats) / len(lats), sum(lngs) / len(lngs)]

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
    if not INPUT_PATH.exists():
        print(f"Error: input file not found: {INPUT_PATH}", file=sys.stderr)
        return 1

    records = _load_records(INPUT_PATH)
    points, skipped = _extract_points(records)

    if not points:
        print(
            "Error: found zero valid coordinates in data.json. "
            "Expected location.lat and location.lng values.",
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
