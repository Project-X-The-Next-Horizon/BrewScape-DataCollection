import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "collect_location_data"))

import collect_coffee_shops_with_reviews as collector


def _make_record(*, collected: bool = False) -> dict[str, object]:
    return {
        "lat": 18.758972,
        "lng": 98.932816,
        "radius": 1000,
        "collected": collected,
        "population_density": None,
    }


def _make_row(place_id: str) -> dict[str, object]:
    return {
        "place_id": place_id,
        "name": f"Place {place_id}",
        "lat": 18.758972,
        "lon": 98.932816,
        "average_rating": 4.5,
        "total_review_count": 12,
        "earliest_available_review_date": "2024-01-01",
        "review_1_text": "review 1",
        "review_2_text": "",
        "review_3_text": "",
        "review_4_text": "",
        "review_5_text": "",
    }


class CollectCoffeeShopsWithReviewsTests(unittest.TestCase):
    def _run_main(
        self,
        records: list[dict[str, object]],
        collect_result: collector.LocationCollectionResult,
        detail_side_effect,
    ) -> tuple[int, list[dict[str, object]], list[dict[str, str]], int, int]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir = Path(tmp_dir)
            input_path = temp_dir / "lat_lng_radius.json"
            output_path = temp_dir / "coffee_shops_with_reviews.csv"
            input_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

            with (
                patch.object(collector, "INPUT_PATH", input_path),
                patch.object(collector, "OUTPUT_PATH", output_path),
                patch.object(collector, "API_KEY", "test-key"),
                patch.object(
                    collector,
                    "_collect_place_ids_for_location",
                    return_value=collect_result,
                ) as collect_mock,
                patch.object(
                    collector,
                    "_fetch_place_details",
                    side_effect=detail_side_effect,
                ) as details_mock,
            ):
                exit_code = collector.main()

            updated_records = json.loads(input_path.read_text(encoding="utf-8"))
            if output_path.exists():
                with output_path.open("r", encoding="utf-8", newline="") as handle:
                    output_rows = list(csv.DictReader(handle))
            else:
                output_rows = []

            return (
                exit_code,
                updated_records,
                output_rows,
                collect_mock.call_count,
                details_mock.call_count,
            )

    def test_collect_place_ids_uses_nearby_payload_without_pagination(self) -> None:
        stats = collector.Stats()
        location = collector.Location(
            lat=18.758972,
            lng=98.932816,
            radius=1000,
            collected=False,
            record_index=0,
            input_index=1,
        )

        with patch.object(collector, "_request_json", return_value={"places": []}) as request_mock:
            result = collector._collect_place_ids_for_location(
                location=location,
                location_seq=1,
                total_locations=1,
                stats=stats,
                persisted_place_ids=set(),
            )

        self.assertTrue(result.success)
        self.assertEqual(result.place_ids, [])
        self.assertEqual(request_mock.call_count, 1)

        kwargs = request_mock.call_args.kwargs
        self.assertEqual(kwargs["field_mask"], collector.NEARBY_FIELD_MASK)
        self.assertNotIn("nextPageToken", kwargs["field_mask"])
        self.assertNotIn("pageToken", kwargs["payload"])
        self.assertEqual(kwargs["payload"]["includedTypes"], ["cafe"])
        self.assertEqual(
            kwargs["payload"]["locationRestriction"]["circle"]["center"],
            {"latitude": 18.758972, "longitude": 98.932816},
        )

    def test_main_keeps_location_uncollected_when_nearby_search_fails(self) -> None:
        exit_code, updated_records, output_rows, collect_calls, details_calls = self._run_main(
            records=[_make_record(collected=False)],
            collect_result=collector.LocationCollectionResult(place_ids=[], success=False),
            detail_side_effect=AssertionError("details should not be fetched"),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(collect_calls, 1)
        self.assertEqual(details_calls, 0)
        self.assertFalse(updated_records[0]["collected"])
        self.assertEqual(output_rows, [])

    def test_main_marks_location_collected_when_nearby_search_returns_zero_places(self) -> None:
        exit_code, updated_records, output_rows, collect_calls, details_calls = self._run_main(
            records=[_make_record(collected=False)],
            collect_result=collector.LocationCollectionResult(place_ids=[], success=True),
            detail_side_effect=AssertionError("details should not be fetched"),
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(collect_calls, 1)
        self.assertEqual(details_calls, 0)
        self.assertTrue(updated_records[0]["collected"])
        self.assertEqual(output_rows, [])

    def test_main_keeps_location_uncollected_when_any_detail_fetch_fails(self) -> None:
        exit_code, updated_records, output_rows, collect_calls, details_calls = self._run_main(
            records=[_make_record(collected=False)],
            collect_result=collector.LocationCollectionResult(
                place_ids=["pid-1", "pid-2"],
                success=True,
            ),
            detail_side_effect=[_make_row("pid-1"), None],
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(collect_calls, 1)
        self.assertEqual(details_calls, 2)
        self.assertFalse(updated_records[0]["collected"])
        self.assertEqual(output_rows, [])


if __name__ == "__main__":
    unittest.main()
