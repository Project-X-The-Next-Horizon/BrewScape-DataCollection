import math
import sys
import unittest
from pathlib import Path

from shapely.geometry import Point, box


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "cnx_main" / "visualise_collection_radius_baseon_pd"))

import generate_lat_lng_radius as generator


def _assert_area_close(test_case: unittest.TestCase, actual: float, expected: float) -> None:
    test_case.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))


class GenerateLatLngRadiusTests(unittest.TestCase):
    def test_exact_undercovered_area_is_nonzero_when_border_only_has_single_cover(self) -> None:
        border = box(0.0, 0.0, 100.0, 100.0)
        circles = [Point(50.0, 50.0).buffer(200.0).intersection(border)]

        state = generator._exact_coverage_state(
            border_xy=border,
            circle_geometries=circles,
            min_cover_count=2,
        )

        _assert_area_close(self, state.undercovered_area, border.area)
        _assert_area_close(self, state.deficit_area, border.area)
        _assert_area_close(self, state.geoms_by_count[1].area, border.area)

    def test_exact_undercovered_area_is_zero_when_border_has_double_cover_everywhere(self) -> None:
        border = box(0.0, 0.0, 100.0, 100.0)
        circles = [
            Point(50.0, 50.0).buffer(200.0).intersection(border),
            Point(50.0, 50.0).buffer(200.0).intersection(border),
        ]

        state = generator._exact_coverage_state(
            border_xy=border,
            circle_geometries=circles,
            min_cover_count=2,
        )

        _assert_area_close(self, state.undercovered_area, 0.0)
        _assert_area_close(self, state.deficit_area, 0.0)
        _assert_area_close(self, state.geoms_by_count[2].area, border.area)

    def test_soft_max_reporting_marks_five_fold_overlap_without_invalidating_min_cover(self) -> None:
        border = box(0.0, 0.0, 100.0, 100.0)
        circles = [Point(50.0, 50.0).buffer(200.0).intersection(border) for _ in range(5)]

        state = generator._exact_coverage_state(
            border_xy=border,
            circle_geometries=circles,
            min_cover_count=2,
        )
        coverage_ratio, mean_mult, min_sample_cover, samples_below_min, samples_above_soft_max = (
            generator._coverage_metrics([5, 5, 4, 2], min_cover_count=2, soft_max_cover_count=4)
        )

        _assert_area_close(self, state.undercovered_area, 0.0)
        self.assertEqual(samples_below_min, 0)
        self.assertEqual(samples_above_soft_max, 2)
        self.assertEqual(min_sample_cover, 2)
        self.assertEqual(coverage_ratio, 1.0)
        self.assertEqual(mean_mult, 4.0)


if __name__ == "__main__":
    unittest.main()
