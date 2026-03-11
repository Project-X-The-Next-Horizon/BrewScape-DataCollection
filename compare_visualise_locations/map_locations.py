#!/usr/bin/env python3
"""
Render an interactive comparison map from two collected coffee-shop CSV datasets.

Data flow (high level):
1) Load two CSV files from the built-in defaults or from command-line overrides.
2) Validate required headers and extract unique, valid place_id/lat/lon rows per file.
3) Compare the union of place IDs across both files.
4) Render one Folium marker per place_id:
   - green when present in both files
   - red when present only in CSV 1
   - blue when present only in CSV 2
5) Save compare_locations_map.html in this folder.

Dependency:
    pip install folium
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
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
OUTPUT_PATH = REPO_ROOT / "compare_locations_map.html"
DEFAULT_INPUT_PATHS = (
    REPO_ROOT.parent
    / "collect_location_data"
    / "prev_data"
    / "coffee_shops_with_reviews3.csv",
    REPO_ROOT.parent
    / "collect_location_data"
    / "prev_data"
    / "coffee_shops_with_reviews4.csv",
)
REQUIRED_COLUMNS = {"place_id", "lat", "lon"}
STATUS_STYLES = {
    "shared": {"label": "Shared", "color": "#2ca02c"},
    "only_csv1": {"label": "Only in CSV 1", "color": "#d62728"},
    "only_csv2": {"label": "Only in CSV 2", "color": "#1f77b4"},
}


class CsvInputError(Exception):
    """Raised when a CSV file is missing or structurally invalid."""


@dataclass(frozen=True)
class DatasetRecord:
    """Normalized CSV row retained for mapping and popup rendering."""

    place_id: str
    lat: float
    lng: float
    values: dict[str, str]


@dataclass
class DatasetLoadResult:
    """Validated records and summary counts for one input CSV."""

    label: str
    path: Path
    records_by_id: dict[str, DatasetRecord] = field(default_factory=dict)
    total_rows: int = 0
    missing_place_id_rows: int = 0
    invalid_coordinate_rows: int = 0
    duplicate_place_id_rows: int = 0

    @property
    def file_name(self) -> str:
        return self.path.name

    @property
    def plotted_rows(self) -> int:
        return len(self.records_by_id)

    @property
    def skipped_rows(self) -> int:
        return (
            self.missing_place_id_rows
            + self.invalid_coordinate_rows
            + self.duplicate_place_id_rows
        )


@dataclass(frozen=True)
class ComparisonStats:
    """Counts describing the overlap between the two CSV inputs."""

    shared_count: int
    only_csv1_count: int
    only_csv2_count: int
    coordinate_mismatch_count: int


def _to_float(value: Any) -> float | None:
    """Best-effort float coercion for CSV values."""
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


def _to_text(value: Any) -> str | None:
    """Normalize a value for display in popup content."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _format_coordinates(lat: float, lng: float) -> str:
    """Render coordinates with a stable precision for popup display."""
    return f"{lat:.6f}, {lng:.6f}"


def _coordinates_differ(record1: DatasetRecord, record2: DatasetRecord) -> bool:
    """Return True when the two coordinates are materially different."""
    return record1.lat != record2.lat or record1.lng != record2.lng


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load and validate a CSV file before normalization."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                raise CsvInputError(f"CSV file has no header row: {path}")
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise CsvInputError(f"could not read CSV file {path}: {exc}") from exc
    except csv.Error as exc:
        raise CsvInputError(f"invalid CSV in {path}: {exc}") from exc

    missing_columns = sorted(REQUIRED_COLUMNS - set(fieldnames))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise CsvInputError(f"CSV is missing required columns ({missing}): {path}")

    return rows


def _load_dataset(path: Path, label: str) -> DatasetLoadResult:
    """Load one CSV and retain the first valid record for each place_id."""
    rows = _load_csv_rows(path)
    result = DatasetLoadResult(label=label, path=path)

    for row in rows:
        result.total_rows += 1

        place_id = _to_text(row.get("place_id"))
        if place_id is None:
            result.missing_place_id_rows += 1
            continue

        lat = _to_float(row.get("lat"))
        lng = _to_float(row.get("lon"))
        if lat is None or lng is None or not (-90 <= lat <= 90 and -180 <= lng <= 180):
            result.invalid_coordinate_rows += 1
            continue

        if place_id in result.records_by_id:
            result.duplicate_place_id_rows += 1
            continue

        result.records_by_id[place_id] = DatasetRecord(
            place_id=place_id,
            lat=lat,
            lng=lng,
            values=row,
        )

    return result


def _resolve_input_paths(args: list[str]) -> tuple[Path, Path]:
    """Resolve CSV inputs from defaults or a two-argument override."""
    if not args:
        return DEFAULT_INPUT_PATHS

    if len(args) != 2:
        raise CsvInputError(
            "expected either zero arguments to use the built-in CSV paths or "
            "exactly two CSV paths"
        )

    return Path(args[0]).expanduser(), Path(args[1]).expanduser()


def _render_metadata_table(record: DatasetRecord) -> str:
    """Build a compact table for one dataset's row metadata."""
    rows = [
        ("Name", _to_text(record.values.get("name"))),
        ("Place ID", record.place_id),
        ("Average Rating", _to_text(record.values.get("average_rating"))),
        ("Total Review Count", _to_text(record.values.get("total_review_count"))),
        (
            "Earliest Available Review Date",
            _to_text(record.values.get("earliest_available_review_date")),
        ),
        ("Coordinates", _format_coordinates(record.lat, record.lng)),
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

    return "<table>" + "".join(rendered_rows) + "</table>"


def _build_source_section(title: str, record: DatasetRecord) -> str:
    """Render one file-specific section inside a popup."""
    return (
        "<div style='margin-top:8px;'>"
        f"<div style='font-weight:600; margin-bottom:4px;'>{escape(title)}</div>"
        f"{_render_metadata_table(record)}"
        "</div>"
    )


def _build_popup_html(
    *,
    status_key: str,
    dataset1: DatasetLoadResult,
    dataset2: DatasetLoadResult,
    record1: DatasetRecord | None,
    record2: DatasetRecord | None,
) -> str:
    """Build popup HTML for one compared place_id."""
    status_style = STATUS_STYLES[status_key]
    status_badge = (
        "<div "
        "style='display:inline-block; padding:3px 8px; border-radius:999px; "
        f"background:{status_style['color']}; color:white; font-weight:600; font-size:11px;'>"
        f"{escape(status_style['label'])}"
        "</div>"
    )

    sections = []
    if record1 is not None:
        sections.append(_build_source_section(dataset1.file_name, record1))
    if record2 is not None:
        sections.append(_build_source_section(dataset2.file_name, record2))

    mismatch_note = ""
    if record1 is not None and record2 is not None and _coordinates_differ(record1, record2):
        mismatch_note = (
            "<div style='margin-top:8px; padding:6px 8px; border-radius:6px; "
            "background:#fff6df; border:1px solid #e6c46c; color:#6c5410;'>"
            "Coordinates differ between files. The marker uses CSV 1 coordinates."
            "</div>"
        )

    return (
        "<div style='font-family:Arial,sans-serif; font-size:12px; line-height:1.35;'>"
        f"{status_badge}"
        f"{mismatch_note}"
        f"{''.join(sections)}"
        "</div>"
    )


def _build_comparison_points(
    dataset1: DatasetLoadResult,
    dataset2: DatasetLoadResult,
) -> tuple[list[dict[str, Any]], ComparisonStats]:
    """Compare the two datasets by place_id and build map-ready points."""
    place_ids = sorted(set(dataset1.records_by_id) | set(dataset2.records_by_id))
    points: list[dict[str, Any]] = []
    shared_count = 0
    only_csv1_count = 0
    only_csv2_count = 0
    coordinate_mismatch_count = 0

    for place_id in place_ids:
        record1 = dataset1.records_by_id.get(place_id)
        record2 = dataset2.records_by_id.get(place_id)

        if record1 is not None and record2 is not None:
            status_key = "shared"
            lat = record1.lat
            lng = record1.lng
            shared_count += 1
            if _coordinates_differ(record1, record2):
                coordinate_mismatch_count += 1
        elif record1 is not None:
            status_key = "only_csv1"
            lat = record1.lat
            lng = record1.lng
            only_csv1_count += 1
        else:
            status_key = "only_csv2"
            lat = record2.lat
            lng = record2.lng
            only_csv2_count += 1

        status_style = STATUS_STYLES[status_key]
        chosen_record = record1 or record2
        name = ""
        if chosen_record is not None:
            name = _to_text(chosen_record.values.get("name")) or ""

        points.append(
            {
                "place_id": place_id,
                "name": name,
                "lat": lat,
                "lng": lng,
                "radius": 5,
                "color": status_style["color"],
                "status_key": status_key,
                "status_label": status_style["label"],
                "popup_html": _build_popup_html(
                    status_key=status_key,
                    dataset1=dataset1,
                    dataset2=dataset2,
                    record1=record1,
                    record2=record2,
                ),
            }
        )

    return points, ComparisonStats(
        shared_count=shared_count,
        only_csv1_count=only_csv1_count,
        only_csv2_count=only_csv2_count,
        coordinate_mismatch_count=coordinate_mismatch_count,
    )


def _legend_row(color: str, label: str, count: int) -> str:
    """Render one colored legend item."""
    return (
        "<div style='display:flex; align-items:center; gap:8px; margin-top:6px;'>"
        f"<span style='display:inline-block; width:10px; height:10px; border-radius:50%; background:{color};'></span>"
        f"<span>{escape(label)} ({count})</span>"
        "</div>"
    )


def _add_summary_overlay(
    point_map: folium.Map,
    dataset1: DatasetLoadResult,
    dataset2: DatasetLoadResult,
    stats: ComparisonStats,
) -> None:
    """Inject a fixed legend/count overlay for the compare map."""
    total_plotted = stats.shared_count + stats.only_csv1_count + stats.only_csv2_count
    overlay_html = (
        "<div style='"
        "position: fixed; bottom: 24px; right: 24px; z-index: 9999; width: 300px; "
        "background: rgba(255, 255, 255, 0.97); border: 1px solid #bbb; "
        "border-radius: 8px; padding: 10px 12px; "
        "box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15); "
        "font-family: Arial, sans-serif; font-size: 12px; line-height: 1.35;'>"
        "<div style='font-weight: 600; margin-bottom: 6px;'>Compare locations</div>"
        f"<div style='color:#444;'>CSV 1: {escape(dataset1.file_name)}</div>"
        f"<div style='color:#444;'>CSV 2: {escape(dataset2.file_name)}</div>"
        "<div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #e5e5e5;'>"
        f"{_legend_row(STATUS_STYLES['shared']['color'], 'Shared in both CSVs', stats.shared_count)}"
        f"{_legend_row(STATUS_STYLES['only_csv1']['color'], 'Only in CSV 1', stats.only_csv1_count)}"
        f"{_legend_row(STATUS_STYLES['only_csv2']['color'], 'Only in CSV 2', stats.only_csv2_count)}"
        "</div>"
        "<div style='margin-top: 8px; padding-top: 8px; border-top: 1px solid #e5e5e5; font-weight: 600;'>"
        f"Total plotted: {total_plotted}"
        "</div>"
        "</div>"
    )
    point_map.get_root().html.add_child(folium.Element(overlay_html))


def _add_place_id_search_overlay(
    point_map: folium.Map, points: list[dict[str, Any]], marker_names: list[str]
) -> None:
    """Inject a search box that jumps to a compared marker by place_id."""
    search_box_id = "place-id-search-box"
    search_button_id = "place-id-search-button"
    status_id = "place-id-search-status"
    overlay_html = (
        "<div style='"
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
        searchable_entries.append(
            "{"
            f"placeId:{json.dumps(point['place_id'])},"
            f"name:{json.dumps(point['name'])},"
            f"normalizedPlaceId:{json.dumps(point['place_id'].strip().lower())},"
            f"statusLabel:{json.dumps(point['status_label'])},"
            f"lat:{point['lat']},"
            f"lng:{point['lng']},"
            f"markerName:{json.dumps(marker_name)},"
            f"baseStyle:{{radius:{point['radius']},color:{json.dumps(point['color'])},weight:1,fillColor:{json.dumps(point['color'])},fillOpacity:1.0}}"
            "}"
        )

    searchable_locations = "[\n" + ",\n".join(searchable_entries) + "\n]"
    map_name = point_map.get_name()
    script = f"""
    const searchableLocations = {searchable_locations};

    window.addEventListener("load", function() {{
        const searchInput = document.getElementById("{search_box_id}");
        const searchButton = document.getElementById("{search_button_id}");
        const statusText = document.getElementById("{status_id}");
        let activeMarker = null;
        let activeBaseStyle = null;
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
            if (activeMarker && activeBaseStyle) {{
                activeMarker.setStyle(activeBaseStyle);
                activeMarker.setRadius(activeBaseStyle.radius);
            }}
            activeMarker = null;
            activeBaseStyle = null;
            if (activeHalo) {{
                {map_name}.removeLayer(activeHalo);
                activeHalo = null;
            }}
        }}

        function highlightMarker(location, marker) {{
            resetMarkerHighlight();
            activeMarker = marker;
            activeBaseStyle = location.baseStyle;
            marker.setStyle({{
                color: "#111111",
                weight: 2,
                fillColor: location.baseStyle.fillColor,
                fillOpacity: 1.0,
            }});
            marker.setRadius(Math.max(location.baseStyle.radius + 3, 8));
            activeHalo = L.circle([location.lat, location.lng], {{
                radius: 45,
                color: "#f39c12",
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
                setStatus(
                    selected.name
                        ? `Showing ${{selected.placeId}} (${{selected.name}}) [${{selected.statusLabel}}].`
                        : `Showing ${{selected.placeId}} [${{selected.statusLabel}}].`,
                    "success",
                );
                return;
            }}

            setStatus(
                selected.name
                    ? `Showing first of ${{matches.length}} matches: ${{selected.placeId}} (${{selected.name}}) [${{selected.statusLabel}}].`
                    : `Showing first of ${{matches.length}} matches: ${{selected.placeId}} [${{selected.statusLabel}}].`,
                "success",
            );
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


def _build_map(
    points: list[dict[str, Any]],
    dataset1: DatasetLoadResult,
    dataset2: DatasetLoadResult,
    stats: ComparisonStats,
) -> folium.Map:
    """Create the compare Folium map, add markers, and fit camera bounds."""
    lats = [point["lat"] for point in points]
    lngs = [point["lng"] for point in points]
    center = [sum(lats) / len(lats), sum(lngs) / len(lngs)]

    zoom_start = 16 if len(points) == 1 else 10
    point_map = folium.Map(location=center, tiles="OpenStreetMap", zoom_start=zoom_start)
    marker_names: list[str] = []

    for point in points:
        marker = folium.CircleMarker(
            location=[point["lat"], point["lng"]],
            radius=point["radius"],
            color=point["color"],
            weight=1,
            fill=True,
            fill_color=point["color"],
            fill_opacity=1.0,
            popup=folium.Popup(point["popup_html"], max_width=480),
        )
        marker.add_to(point_map)
        marker_names.append(marker.get_name())

    if len(points) > 1:
        point_map.fit_bounds([[min(lats), min(lngs)], [max(lats), max(lngs)]])

    _add_summary_overlay(point_map, dataset1, dataset2, stats)
    _add_place_id_search_overlay(point_map, points, marker_names)
    return point_map


def _print_dataset_summary(dataset: DatasetLoadResult) -> None:
    """Print normalization counts for one dataset."""
    print(
        f"{dataset.label}: {dataset.path} | "
        f"Total rows: {dataset.total_rows} | "
        f"Plotted points: {dataset.plotted_rows} | "
        f"Skipped rows: {dataset.skipped_rows} | "
        f"Blank place_id: {dataset.missing_place_id_rows} | "
        f"Invalid coordinates: {dataset.invalid_coordinate_rows} | "
        f"Duplicate place_id: {dataset.duplicate_place_id_rows}"
    )


def _print_overlap_summary(stats: ComparisonStats) -> None:
    """Print overlap counts for the combined comparison."""
    total_plotted = stats.shared_count + stats.only_csv1_count + stats.only_csv2_count
    print(
        "Overlap: "
        f"Shared: {stats.shared_count} | "
        f"Only in CSV 1: {stats.only_csv1_count} | "
        f"Only in CSV 2: {stats.only_csv2_count} | "
        f"Coordinate mismatches: {stats.coordinate_mismatch_count} | "
        f"Total plotted: {total_plotted} | "
        f"Output: {OUTPUT_PATH}"
    )


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for generating a compare map from two CSV inputs."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        csv1_path, csv2_path = _resolve_input_paths(args)
    except CsvInputError as exc:
        print(
            "Usage: python compare_visualise_locations/map_locations.py "
            "[<csv1_path> <csv2_path>]",
            file=sys.stderr,
        )
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    missing_paths = [path for path in (csv1_path, csv2_path) if not path.exists()]
    if missing_paths:
        print(
            "Error: input file not found: "
            + ", ".join(str(path) for path in missing_paths),
            file=sys.stderr,
        )
        return 1

    try:
        dataset1 = _load_dataset(csv1_path, "CSV 1")
        dataset2 = _load_dataset(csv2_path, "CSV 2")
    except CsvInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    points, stats = _build_comparison_points(dataset1, dataset2)
    if not points:
        print(
            "Error: found zero valid comparable locations across both CSV files.",
            file=sys.stderr,
        )
        return 1

    point_map = _build_map(points, dataset1, dataset2, stats)
    point_map.save(str(OUTPUT_PATH))

    _print_dataset_summary(dataset1)
    _print_dataset_summary(dataset2)
    _print_overlap_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
