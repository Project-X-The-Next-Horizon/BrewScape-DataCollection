import math
import sys
import time
import unittest
from pathlib import Path

from shapely.geometry import Point, box


REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "cnx_main" / "visualise_collection_radius_baseon_pd"))

import generate_lat_lng_radius as generator


def _assert_area_close(test_case: unittest.TestCase, actual: float, expected: float) -> None:
    test_case.assertTrue(math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6))


def _build_calibration_model(
    points: list[tuple[float, float]],
    *,
    shop_soft_cap: int = 45,
    shop_hard_cap: int = 60,
) -> generator.CalibrationModel:
    shops = [
        generator.CalibrationShop(place_id=f"shop-{idx}", x=point[0], y=point[1])
        for idx, point in enumerate(points)
    ]
    return generator.CalibrationModel(
        shops=shops,
        shop_soft_cap=shop_soft_cap,
        shop_hard_cap=shop_hard_cap,
    )


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

    def test_adaptive_radius_uses_calibration_cap_in_dense_shop_cluster(self) -> None:
        calibration_model = _build_calibration_model(
            [(float(idx * 10), 0.0) for idx in range(46)],
            shop_soft_cap=45,
            shop_hard_cap=60,
        )

        uncapped_radius = generator._adaptive_radius_for_center(
            x=0.0,
            y=0.0,
            density=0.0,
            min_radius=500,
            max_radius=1000,
            low_log=0.0,
            high_log=1.0,
            calibration_model=None,
        )
        capped_radius = generator._adaptive_radius_for_center(
            x=0.0,
            y=0.0,
            density=0.0,
            min_radius=500,
            max_radius=1000,
            low_log=0.0,
            high_log=1.0,
            calibration_model=calibration_model,
        )

        self.assertEqual(uncapped_radius, 1000)
        self.assertEqual(capped_radius, 500)

    def test_shop_repair_adds_circle_for_uncovered_calibration_shop(self) -> None:
        border = box(0.0, 0.0, 2000.0, 2000.0)
        calibration_model = _build_calibration_model([(1500.0, 1000.0)])
        circles = [
            generator.CirclePlacement(
                x=500.0,
                y=1000.0,
                radius=500,
                density=0.0,
                band_idx=0,
                row=0,
                col=0,
                lattice_error=0.0,
                estimated_shop_load=0,
            )
        ]
        density_points = [generator.DensityPoint(x=1500.0, y=1000.0, density=1000.0)]
        lattice_bands = generator._build_lattice_bands(
            border_xy=border,
            min_radius=500,
            max_radius=500,
            band_step=100,
            spacing_scale=0.60,
            phase_x=0.0,
            phase_y=0.0,
        )

        repaired = generator._repair_calibration_coverage(
            border_xy=border,
            lattice_bands=lattice_bands,
            circles=circles,
            density_points=density_points,
            calibration_model=calibration_model,
            min_radius=500,
            max_radius=500,
            low_log=0.0,
            high_log=1.0,
            max_patch_limit=10,
            lattice_patch_radius_factor=1.35,
            lattice_neighbor_ring=2,
            deadline=time.perf_counter() + 5.0,
        )

        self.assertIsNotNone(repaired)
        added, calibration_state = repaired
        self.assertGreaterEqual(added, 1)
        self.assertEqual(calibration_state.uncovered_shops, 0)

    def test_prune_preserves_non_overloaded_calibration_cover(self) -> None:
        border = box(-5.0, -5.0, 5.0, 5.0)
        samples = [(0.0, 0.0)]
        calibration_model = _build_calibration_model(
            [(0.0, 0.0), (40.0, 0.0)],
            shop_soft_cap=1,
            shop_hard_cap=2,
        )
        circles = [
            generator.CirclePlacement(
                x=0.0,
                y=0.0,
                radius=30,
                density=0.0,
                band_idx=0,
                row=0,
                col=0,
                lattice_error=0.0,
                estimated_shop_load=1,
            ),
            generator.CirclePlacement(
                x=20.0,
                y=0.0,
                radius=60,
                density=0.0,
                band_idx=0,
                row=0,
                col=1,
                lattice_error=0.0,
                estimated_shop_load=2,
            ),
        ]

        pruned = generator._prune_circles(
            border_xy=border,
            samples=samples,
            circles=circles,
            calibration_model=calibration_model,
            exact_tolerance_m2=1.0,
            prune_max_passes=1,
            prune_min_cover=1,
            deadline=time.perf_counter() + 5.0,
        )

        self.assertIsNotNone(pruned)
        self.assertEqual(len(pruned[0]), 2)

    def test_dataset_regression_current_output_meets_recall_and_overload_targets(self) -> None:
        border_lng_lat = generator._load_border_geometry(generator.BORDER_PATH)
        projection = generator._build_projection(border_lng_lat)
        calibration_rows = generator._load_calibration_rows(generator.DEFAULT_CALIBRATION_CSV_PATH)
        calibration_model = generator._build_calibration_model(
            calibration_rows=calibration_rows,
            projection=projection,
            shop_soft_cap=generator.DEFAULT_SHOP_SOFT_CAP,
            shop_hard_cap=generator.DEFAULT_SHOP_HARD_CAP,
        )
        rows = generator._load_existing_rows(generator.OUTPUT_PATH)
        circles = generator._rows_to_circles(rows, projection, calibration_model)
        calibration_state = generator._build_calibration_coverage_state(circles, calibration_model)

        self.assertGreaterEqual(
            len(calibration_model.shops) - calibration_state.uncovered_shops,
            2095,
        )
        self.assertLessEqual(calibration_state.hard_overloaded_circles, 3)
        self.assertLess(len(rows), 556)


if __name__ == "__main__":
    unittest.main()
