import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "collect_location_data" / "collect_coffee_shops_with_reviews.py"


def load_collector_module():
    module_name = "collector_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_place(
    place_id: str,
    name: str = "Cafe",
    *,
    lat: float = 18.75,
    lng: float = 98.95,
) -> dict[str, object]:
    return {
        "id": place_id,
        "displayName": {"text": name},
        "location": {"latitude": lat, "longitude": lng},
        "rating": 4.5,
        "userRatingCount": 12,
        "reviews": [
            {
                "text": {"text": f"Review for {name}"},
                "publishTime": "2024-01-01T00:00:00Z",
            }
        ],
    }


def read_csv_place_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["place_id"] for row in reader if row.get("place_id")]


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def make_urlopen_side_effect(events: list[dict[str, object] | Exception]):
    iterator = iter(events)

    def side_effect(*args, **kwargs):
        event = next(iterator)
        if isinstance(event, Exception):
            raise event
        return FakeHTTPResponse(event)

    return side_effect


class ResumableCollectorTests(unittest.TestCase):
    def run_main_with_temp_paths(
        self,
        module,
        temp_root: Path,
        responses: list[dict[str, object] | None],
        *,
        nearby_rank_preferences: list[str] | None = None,
        text_strategies: tuple[object, ...] = (),
    ) -> tuple[int, object]:
        input_path = temp_root / "lat_lng_radius.json"
        output_path = temp_root / "coffee_shops_with_reviews.csv"

        if nearby_rank_preferences is None:
            nearby_rank_preferences = ["DISTANCE"]

        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "INPUT_PATH", input_path))
            stack.enter_context(patch.object(module, "OUTPUT_PATH", output_path))
            stack.enter_context(patch.object(module, "API_KEY", "test-key"))
            stack.enter_context(patch.object(module, "SLEEP_SECONDS", 0))
            stack.enter_context(
                patch.object(module, "NEARBY_RANK_PREFERENCES", nearby_rank_preferences)
            )
            stack.enter_context(
                patch.object(module, "TEXT_SEARCH_STRATEGIES", text_strategies)
            )
            request_mock = stack.enter_context(
                patch.object(module, "_request_json", side_effect=responses)
            )
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                result = module.main()

        return result, request_mock

    def run_main_with_urlopen(
        self,
        module,
        temp_root: Path,
        events: list[dict[str, object] | Exception],
        *,
        nearby_rank_preferences: list[str] | None = None,
        text_strategies: tuple[object, ...] = (),
    ) -> tuple[int, object, object, str, str]:
        input_path = temp_root / "lat_lng_radius.json"
        output_path = temp_root / "coffee_shops_with_reviews.csv"

        if nearby_rank_preferences is None:
            nearby_rank_preferences = ["DISTANCE"]

        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "INPUT_PATH", input_path))
            stack.enter_context(patch.object(module, "OUTPUT_PATH", output_path))
            stack.enter_context(patch.object(module, "API_KEY", "test-key"))
            stack.enter_context(patch.object(module, "SLEEP_SECONDS", 0))
            stack.enter_context(
                patch.object(module, "NEARBY_RANK_PREFERENCES", nearby_rank_preferences)
            )
            stack.enter_context(
                patch.object(module, "TEXT_SEARCH_STRATEGIES", text_strategies)
            )
            sleep_mock = stack.enter_context(patch.object(module.time, "sleep"))
            urlopen_mock = stack.enter_context(
                patch.object(
                    module.request,
                    "urlopen",
                    side_effect=make_urlopen_side_effect(events),
                )
            )
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                result = module.main()

        return (
            result,
            urlopen_mock,
            sleep_mock,
            stdout_buffer.getvalue(),
            stderr_buffer.getvalue(),
        )

    def test_existing_csv_and_collected_circles_are_respected(self):
        module = load_collector_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            input_path = temp_root / "lat_lng_radius.json"
            output_path = temp_root / "coffee_shops_with_reviews.csv"

            input_path.write_text(
                json.dumps(
                    [
                        {
                            "circle_id": "done",
                            "lat": 18.75,
                            "lng": 98.95,
                            "radius": 500,
                            "collected": True,
                        },
                        {
                            "circle_id": "pending",
                            "lat": 18.76,
                            "lng": 98.96,
                            "radius": 500,
                            "collected": False,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=module.CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "place_id": "existing",
                        "name": "Existing Cafe",
                        "lat": 18.75,
                        "lon": 98.95,
                        "average_rating": 4.4,
                        "total_review_count": 9,
                        "earliest_available_review_date": "2024-01-01",
                        "review_1_text": "",
                        "review_2_text": "",
                        "review_3_text": "",
                        "review_4_text": "",
                        "review_5_text": "",
                    }
                )

            responses = [
                {
                    "places": [
                        make_place("existing", lat=18.76, lng=98.96),
                        make_place("new-one", lat=18.76, lng=98.96),
                    ]
                },
                {
                    "places": [
                        make_place("new-one", lat=18.76, lng=98.96),
                        make_place("new-two", lat=18.76, lng=98.96),
                    ]
                },
            ]

            result, request_mock = self.run_main_with_temp_paths(
                module,
                temp_root,
                responses,
                nearby_rank_preferences=["DISTANCE", "POPULARITY"],
                text_strategies=(),
            )

            self.assertEqual(result, 0)
            self.assertEqual(request_mock.call_count, 2)
            self.assertEqual(
                read_csv_place_ids(output_path),
                ["existing", "new-one", "new-two"],
            )

            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertTrue(payload[0]["collected"])
            self.assertTrue(payload[1]["collected"])

    def test_api_failure_leaves_circle_pending_but_keeps_appended_rows(self):
        module = load_collector_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            input_path = temp_root / "lat_lng_radius.json"
            output_path = temp_root / "coffee_shops_with_reviews.csv"

            input_path.write_text(
                json.dumps(
                    [
                        {
                            "circle_id": "pending",
                            "lat": 18.75,
                            "lng": 98.95,
                            "radius": 500,
                            "collected": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            responses = [
                {"places": [make_place("alpha")]},
                None,
            ]

            strategy = module.TextSearchStrategy(
                label="cafe",
                text_query="cafe",
                included_type="cafe",
            )
            result, request_mock = self.run_main_with_temp_paths(
                module,
                temp_root,
                responses,
                nearby_rank_preferences=["DISTANCE"],
                text_strategies=(strategy,),
            )

            self.assertEqual(result, 0)
            self.assertEqual(request_mock.call_count, 2)
            self.assertEqual(read_csv_place_ids(output_path), ["alpha"])

            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertFalse(payload[0]["collected"])

    def test_missing_main_csv_is_created_without_loading_prev_data(self):
        module = load_collector_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            input_path = temp_root / "lat_lng_radius.json"
            output_path = temp_root / "coffee_shops_with_reviews.csv"
            prev_data_dir = temp_root / "prev_data"
            prev_data_dir.mkdir()

            input_path.write_text(
                json.dumps(
                    [
                        {
                            "circle_id": "pending",
                            "lat": 18.75,
                            "lng": 98.95,
                            "radius": 500,
                            "collected": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with (prev_data_dir / "coffee_shops_with_reviews.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=module.CSV_COLUMNS)
                writer.writeheader()
                writer.writerow(
                    {
                        "place_id": "legacy",
                        "name": "Legacy Cafe",
                        "lat": 18.75,
                        "lon": 98.95,
                        "average_rating": 4.0,
                        "total_review_count": 1,
                        "earliest_available_review_date": "2024-01-01",
                        "review_1_text": "",
                        "review_2_text": "",
                        "review_3_text": "",
                        "review_4_text": "",
                        "review_5_text": "",
                    }
                )

            result, _ = self.run_main_with_temp_paths(
                module,
                temp_root,
                [{"places": [make_place("legacy")]}],
                nearby_rank_preferences=["DISTANCE"],
                text_strategies=(),
            )

            self.assertEqual(result, 0)
            self.assertTrue(output_path.exists())
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                lines = handle.read().splitlines()
            self.assertEqual(lines[0], ",".join(module.CSV_COLUMNS))
            self.assertEqual(read_csv_place_ids(output_path), ["legacy"])

    def test_invalid_existing_csv_without_place_id_fails_fast(self):
        module = load_collector_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            output_path = temp_root / "coffee_shops_with_reviews.csv"
            output_path.write_text("name,lat,lon\nCafe,18.75,98.95\n", encoding="utf-8")

            stats = module.Stats()
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module._load_existing_place_ids(output_path, stats)

    def test_request_retries_then_succeeds_and_completes_circle(self):
        module = load_collector_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            input_path = temp_root / "lat_lng_radius.json"
            output_path = temp_root / "coffee_shops_with_reviews.csv"

            input_path.write_text(
                json.dumps(
                    [
                        {
                            "circle_id": "retry-success",
                            "lat": 18.75,
                            "lng": 98.95,
                            "radius": 500,
                            "collected": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            result, urlopen_mock, sleep_mock, stdout_text, stderr_text = (
                self.run_main_with_urlopen(
                    module,
                    temp_root,
                    [
                        module.error.URLError("temporary failure"),
                        {"places": [make_place("alpha")]},
                    ],
                    nearby_rank_preferences=["DISTANCE"],
                    text_strategies=(),
                )
            )

            self.assertEqual(result, 0)
            self.assertEqual(urlopen_mock.call_count, 2)
            self.assertEqual(read_csv_place_ids(output_path), ["alpha"])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertTrue(payload[0]["collected"])
            self.assertIn("Retry attempts performed: 1", stdout_text)
            self.assertIn("Requests succeeded on retry: 1", stdout_text)
            self.assertIn("Requests that exhausted all attempts: 0", stdout_text)
            self.assertIn("[WARN] Retrying", stderr_text)
            sleep_calls = [call.args[0] for call in sleep_mock.call_args_list]
            self.assertIn(1.0, sleep_calls)

    def test_request_exhausts_all_attempts_keeps_circle_pending_and_moves_on(self):
        module = load_collector_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            input_path = temp_root / "lat_lng_radius.json"
            output_path = temp_root / "coffee_shops_with_reviews.csv"

            input_path.write_text(
                json.dumps(
                    [
                        {
                            "circle_id": "will-fail",
                            "lat": 18.75,
                            "lng": 98.95,
                            "radius": 500,
                            "collected": False,
                        },
                        {
                            "circle_id": "will-succeed",
                            "lat": 18.76,
                            "lng": 98.96,
                            "radius": 500,
                            "collected": False,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            result, urlopen_mock, sleep_mock, stdout_text, _ = self.run_main_with_urlopen(
                module,
                temp_root,
                [
                    module.error.URLError("failure one"),
                    module.error.URLError("failure two"),
                    module.error.URLError("failure three"),
                    {"places": [make_place("beta", lat=18.76, lng=98.96)]},
                ],
                nearby_rank_preferences=["DISTANCE"],
                text_strategies=(),
            )

            self.assertEqual(result, 0)
            self.assertEqual(urlopen_mock.call_count, 4)
            self.assertEqual(read_csv_place_ids(output_path), ["beta"])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertFalse(payload[0]["collected"])
            self.assertTrue(payload[1]["collected"])
            self.assertIn("Requests that exhausted all attempts: 1", stdout_text)
            self.assertIn("Circles left pending because of API errors: 1", stdout_text)
            sleep_calls = [call.args[0] for call in sleep_mock.call_args_list]
            self.assertIn(1.0, sleep_calls)
            self.assertIn(2.0, sleep_calls)

    def test_partial_csv_write_survives_when_later_request_exhausts_retries(self):
        module = load_collector_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            input_path = temp_root / "lat_lng_radius.json"
            output_path = temp_root / "coffee_shops_with_reviews.csv"

            input_path.write_text(
                json.dumps(
                    [
                        {
                            "circle_id": "partial-save",
                            "lat": 18.75,
                            "lng": 98.95,
                            "radius": 500,
                            "collected": False,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            strategy = module.TextSearchStrategy(
                label="cafe",
                text_query="cafe",
                included_type="cafe",
            )
            result, urlopen_mock, _, stdout_text, _ = self.run_main_with_urlopen(
                module,
                temp_root,
                [
                    {"places": [make_place("alpha")]},
                    module.error.URLError("failure one"),
                    module.error.URLError("failure two"),
                    module.error.URLError("failure three"),
                ],
                nearby_rank_preferences=["DISTANCE"],
                text_strategies=(strategy,),
            )

            self.assertEqual(result, 0)
            self.assertEqual(urlopen_mock.call_count, 4)
            self.assertEqual(read_csv_place_ids(output_path), ["alpha"])
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            self.assertFalse(payload[0]["collected"])
            self.assertIn("Rows appended this run: 1", stdout_text)
            self.assertIn("Requests that exhausted all attempts: 1", stdout_text)


if __name__ == "__main__":
    unittest.main()
