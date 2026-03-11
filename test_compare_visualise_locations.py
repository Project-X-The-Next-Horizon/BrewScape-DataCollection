import contextlib
import csv
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "compare_visualise_locations"))

import map_locations as compare_map


FIELDNAMES = [
    "place_id",
    "name",
    "lat",
    "lon",
    "average_rating",
    "total_review_count",
    "earliest_available_review_date",
    "review_1_text",
]


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _row(place_id: str, lat: object, lon: object, *, name: str | None = None) -> dict[str, object]:
    return {
        "place_id": place_id,
        "name": name or f"Place {place_id}",
        "lat": lat,
        "lon": lon,
        "average_rating": 4.5,
        "total_review_count": 12,
        "earliest_available_review_date": "2024-01-01",
        "review_1_text": "unused in compare popup",
    }


class CompareVisualiseLocationsTests(unittest.TestCase):
    def test_load_dataset_skips_blank_invalid_and_duplicate_place_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "input.csv"
            _write_csv(
                csv_path,
                [
                    _row("pid-1", 18.1, 98.1),
                    _row("", 18.2, 98.2),
                    _row("pid-2", "not-a-number", 98.3),
                    _row("pid-1", 18.4, 98.4),
                    _row("pid-3", 18.5, 98.5),
                ],
            )

            dataset = compare_map._load_dataset(csv_path, "CSV 1")

        self.assertEqual(dataset.total_rows, 5)
        self.assertEqual(dataset.plotted_rows, 2)
        self.assertEqual(dataset.skipped_rows, 3)
        self.assertEqual(dataset.missing_place_id_rows, 1)
        self.assertEqual(dataset.invalid_coordinate_rows, 1)
        self.assertEqual(dataset.duplicate_place_id_rows, 1)
        self.assertEqual(sorted(dataset.records_by_id), ["pid-1", "pid-3"])

    def test_build_comparison_points_uses_csv1_coordinates_for_shared_mismatch(self) -> None:
        dataset1 = compare_map.DatasetLoadResult(
            label="CSV 1",
            path=Path("csv1.csv"),
            records_by_id={
                "shared": compare_map.DatasetRecord(
                    place_id="shared",
                    lat=18.111,
                    lng=98.111,
                    values={"place_id": "shared", "name": "Shared 1"},
                ),
                "only-1": compare_map.DatasetRecord(
                    place_id="only-1",
                    lat=18.222,
                    lng=98.222,
                    values={"place_id": "only-1", "name": "Only 1"},
                ),
            },
        )
        dataset2 = compare_map.DatasetLoadResult(
            label="CSV 2",
            path=Path("csv2.csv"),
            records_by_id={
                "shared": compare_map.DatasetRecord(
                    place_id="shared",
                    lat=19.999,
                    lng=99.999,
                    values={"place_id": "shared", "name": "Shared 2"},
                ),
                "only-2": compare_map.DatasetRecord(
                    place_id="only-2",
                    lat=18.333,
                    lng=98.333,
                    values={"place_id": "only-2", "name": "Only 2"},
                ),
            },
        )

        points, stats = compare_map._build_comparison_points(dataset1, dataset2)
        points_by_id = {point["place_id"]: point for point in points}

        self.assertEqual(stats.shared_count, 1)
        self.assertEqual(stats.only_csv1_count, 1)
        self.assertEqual(stats.only_csv2_count, 1)
        self.assertEqual(stats.coordinate_mismatch_count, 1)
        self.assertEqual(points_by_id["shared"]["status_key"], "shared")
        self.assertEqual(points_by_id["shared"]["lat"], 18.111)
        self.assertEqual(points_by_id["shared"]["lng"], 98.111)
        self.assertIn("Coordinates differ between files", points_by_id["shared"]["popup_html"])

    def test_main_requires_exactly_two_csv_paths(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            exit_code = compare_map.main([])

        self.assertEqual(exit_code, 1)
        self.assertIn("Usage: python compare_visualise_locations/map_locations.py", stderr.getvalue())

    def test_main_returns_error_when_required_headers_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv1 = Path(tmp_dir) / "csv1.csv"
            csv2 = Path(tmp_dir) / "csv2.csv"
            _write_csv(csv1, [_row("pid-1", 18.1, 98.1)], fieldnames=["place_id", "name", "lat"])
            _write_csv(csv2, [_row("pid-2", 18.2, 98.2)])

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = compare_map.main([str(csv1), str(csv2)])

        self.assertEqual(exit_code, 1)
        self.assertIn("missing required columns", stderr.getvalue())

    def test_main_generates_compare_map_and_reports_overlap_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            csv1 = temp_dir / "csv1.csv"
            csv2 = temp_dir / "csv2.csv"
            output_path = temp_dir / "compare_locations_map.html"

            _write_csv(
                csv1,
                [
                    _row("shared", 18.1, 98.1, name="Shared"),
                    _row("only-1", 18.2, 98.2, name="Only One"),
                    _row("", 18.9, 98.9, name="Blank"),
                    _row("shared", 18.3, 98.3, name="Duplicate Shared"),
                ],
            )
            _write_csv(
                csv2,
                [
                    _row("shared", 18.1, 98.1, name="Shared"),
                    _row("only-2", 18.4, 98.4, name="Only Two"),
                    _row("bad-coords", "oops", 98.5, name="Bad Coords"),
                ],
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch.object(compare_map, "OUTPUT_PATH", output_path),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = compare_map.main([str(csv1), str(csv2)])

            self.assertEqual(exit_code, 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue(output_path.exists())

            html = output_path.read_text(encoding="utf-8")
            output = stdout.getvalue()
            self.assertIn("CSV 1: ", output)
            self.assertIn("Shared: 1", output)
            self.assertIn("Only in CSV 1: 1", output)
            self.assertIn("Only in CSV 2: 1", output)
            self.assertIn("csv1.csv", html)
            self.assertIn("csv2.csv", html)
            self.assertIn("Search by Place ID", html)
            self.assertIn("Shared in both CSVs (1)", html)
            self.assertIn("Only in CSV 1 (1)", html)
            self.assertIn("Only in CSV 2 (1)", html)


if __name__ == "__main__":
    unittest.main()
