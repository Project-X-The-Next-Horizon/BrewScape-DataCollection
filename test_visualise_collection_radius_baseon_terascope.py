import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from shapely.geometry import Point, Polygon

from cnx_main.visualise_collection_radius_baseon_terascope import (
    generate_lat_lng_radius as terascope_gen,
)
from cnx_main.visualise_collection_radius_baseon_terascope import (
    visualise_collection_radius as terascope_vis,
)


def _border_polygon() -> Polygon:
    return Polygon(
        [
            (-0.03, -0.03),
            (0.03, -0.03),
            (0.03, 0.03),
            (-0.03, 0.03),
            (-0.03, -0.03),
        ]
    )


def _make_terascope_row(
    projection: terascope_gen.LocalProjection,
    *,
    grid_row: int,
    grid_col: int,
    land_cover_label: str,
    built_up_share_pct: float,
) -> dict[str, float | int | str]:
    x = (grid_col + 0.5) * terascope_gen.TERASCOPE_GRID_SIZE_M
    y = (grid_row + 0.5) * terascope_gen.TERASCOPE_GRID_SIZE_M
    lng, lat = projection.to_lng_lat(x, y)
    radius_rule, radius = terascope_gen._radius_rule_for_land_cover(
        land_cover_label,
        built_up_share_pct,
    )
    return {
        "grid_row": grid_row,
        "grid_col": grid_col,
        "lng": lng,
        "lat": lat,
        "land_cover_label": land_cover_label,
        "built_up_share_pct": built_up_share_pct,
        "radius_rule": radius_rule,
        "radius": radius,
    }


def _synthetic_rows(projection: terascope_gen.LocalProjection) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for grid_row in range(0, 30):
        for grid_col in range(0, 30):
            if 8 <= grid_row <= 21 and 8 <= grid_col <= 21:
                label = "Built-up"
                built_up_share = 100.0
            elif grid_col < 8:
                label = "Cropland"
                built_up_share = 0.0
            else:
                label = "Tree cover"
                built_up_share = 0.0
            rows.append(
                _make_terascope_row(
                    projection,
                    grid_row=grid_row,
                    grid_col=grid_col,
                    land_cover_label=label,
                    built_up_share_pct=built_up_share,
                )
            )
    return rows


class TerascopeAdaptiveRadiusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.border = _border_polygon()
        self.projection = terascope_gen._build_projection(self.border)

    def test_radius_rule_mapping_uses_stage_rules(self) -> None:
        self.assertEqual(
            terascope_gen._radius_rule_for_land_cover("Built-up", 25.0),
            ("built_up_dominant", 500),
        )
        self.assertEqual(
            terascope_gen._radius_rule_for_land_cover("Cropland", 5.0),
            ("open_non_builtup", 1500),
        )
        self.assertEqual(
            terascope_gen._radius_rule_for_land_cover("Tree cover", 90.0),
            ("sparse_non_builtup", 2000),
        )

    def test_query_density_uses_exact_grid_then_nearest_fallback(self) -> None:
        rows = [
            _make_terascope_row(
                self.projection,
                grid_row=0,
                grid_col=0,
                land_cover_label="Built-up",
                built_up_share_pct=100.0,
            ),
            _make_terascope_row(
                self.projection,
                grid_row=0,
                grid_col=1,
                land_cover_label="Cropland",
                built_up_share_pct=0.0,
            ),
            _make_terascope_row(
                self.projection,
                grid_row=1,
                grid_col=1,
                land_cover_label="Tree cover",
                built_up_share_pct=0.0,
            ),
        ]
        model = terascope_gen._build_terascope_model(rows, self.projection)

        exact = terascope_gen._query_density(50.0, 50.0, model)
        self.assertEqual((exact.grid_row, exact.grid_col), (0, 0))
        self.assertEqual(exact.radius, 500)

        fallback = terascope_gen._query_density(-40.0, 40.0, model)
        self.assertEqual((fallback.grid_row, fallback.grid_col), (0, 0))

        outside = terascope_gen._query_density(190.0, 60.0, model)
        self.assertEqual((outside.grid_row, outside.grid_col), (0, 1))

    def test_build_stage_masks_selects_dominant_built_up_cells(self) -> None:
        rows = [
            _make_terascope_row(self.projection, grid_row=0, grid_col=0, land_cover_label="Built-up", built_up_share_pct=40.0),
            _make_terascope_row(self.projection, grid_row=0, grid_col=1, land_cover_label="Cropland", built_up_share_pct=70.0),
            _make_terascope_row(self.projection, grid_row=1, grid_col=0, land_cover_label="Tree cover", built_up_share_pct=0.0),
        ]
        model = terascope_gen._build_terascope_model(rows, self.projection)
        built_target, _non_built_target, built_center, non_built_center = terascope_gen._build_stage_masks(
            model,
            self.projection.geom_to_xy(self.border),
        )
        built_cell = model.cells_by_key[(0, 0)]
        cropland_cell = model.cells_by_key[(0, 1)]
        self.assertTrue(built_target.covers(Point(built_cell.x, built_cell.y)))
        self.assertFalse(built_center.covers(Point(cropland_cell.x, cropland_cell.y)))
        self.assertTrue(non_built_center.covers(Point(cropland_cell.x, cropland_cell.y)))

    def test_square_grid_spacing_matches_circle_radius(self) -> None:
        origins = terascope_gen._build_square_grid_origins((0.0, 0.0, 2000.0, 2000.0), 500, ((0.0, 0.0),))
        phase_label, origin_x, origin_y = origins[0]
        self.assertEqual(phase_label, "0.0,0.0")
        nodes = list(terascope_gen._iter_square_nodes_in_bounds((0.0, 0.0, 1000.0, 1000.0), 500, origin_x, origin_y))
        x0, y0 = nodes[0][2], nodes[0][3]
        same_row = next(node for node in nodes if node[1] == nodes[0][1] + 1 and node[0] == nodes[0][0])
        same_col = next(node for node in nodes if node[0] == nodes[0][0] + 1 and node[1] == nodes[0][1])
        self.assertAlmostEqual(((same_row[2] - x0) ** 2 + (same_row[3] - y0) ** 2) ** 0.5, 500.0, places=6)
        self.assertAlmostEqual(((same_col[2] - x0) ** 2 + (same_col[3] - y0) ** 2) ** 0.5, 500.0, places=6)

    def test_stage_generation_respects_two_stage_constraints(self) -> None:
        rows = _synthetic_rows(self.projection)
        model = terascope_gen._build_terascope_model(rows, self.projection)
        border_xy = self.projection.geom_to_xy(self.border)
        built_target, non_built_target, built_center, non_built_center = terascope_gen._build_stage_masks(model, border_xy)

        built_result = terascope_gen._generate_stage(
            stage_name="built_up",
            target_mask=built_target,
            center_mask=built_center,
            density_points=model,
            calibration_model=None,
            support_circles=[],
            radius_order=(500,),
            min_cover_count=2,
            exact_tolerance_m2=1.0,
            exact_patch_limit=200,
            deadline=time.perf_counter() + 10.0,
        )
        self.assertIsNotNone(built_result)
        assert built_result is not None
        self.assertTrue(all(circle.radius == 500 for circle in built_result.circles))
        built_cells = [cell for cell in model.cells if cell.land_cover_label == "Built-up"]
        built_counts = terascope_gen._coverage_counts_for_cells(built_cells, built_result.circles)
        self.assertTrue(all(count >= 2 for count in built_counts))

        non_built_result = terascope_gen._generate_stage(
            stage_name="non_built_up",
            target_mask=non_built_target,
            center_mask=non_built_center,
            density_points=model,
            calibration_model=None,
            support_circles=built_result.circles,
            radius_order=(1500, 2000),
            min_cover_count=2,
            exact_tolerance_m2=1.0,
            exact_patch_limit=300,
            deadline=time.perf_counter() + 10.0,
        )
        self.assertIsNotNone(non_built_result)
        assert non_built_result is not None
        self.assertTrue(all(circle.radius in {1500, 2000} for circle in non_built_result.circles))
        non_built_cells = [cell for cell in model.cells if cell.land_cover_label != "Built-up"]
        non_built_counts = terascope_gen._coverage_counts_for_cells(
            non_built_cells,
            built_result.circles + non_built_result.circles,
        )
        self.assertTrue(all(count >= 2 for count in non_built_counts))
        self.assertTrue(all(not built_center.covers(Point(circle.x, circle.y)) for circle in non_built_result.circles))

    def test_main_preserves_collected_and_writes_stage_metadata(self) -> None:
        rows = _synthetic_rows(self.projection)
        output_border = self.border
        first_built_row = next(row for row in rows if row["land_cover_label"] == "Built-up")
        first_lng = float(first_built_row["lng"])
        first_lat = float(first_built_row["lat"])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "lat_lng_radius.json"
            output_path.write_text(
                json.dumps(
                    [
                        {
                            "lat": round(first_lat, 6),
                            "lng": round(first_lng, 6),
                            "radius": 500,
                            "collected": True,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch.object(terascope_gen, "OUTPUT_PATH", output_path),
                patch.object(terascope_gen, "_load_border_geometry", return_value=output_border),
                patch.object(terascope_gen, "_load_terascope_rows", return_value=rows),
            ):
                rc = terascope_gen.main(
                    [
                        "--disable-calibration",
                        "--preserve-distance",
                        "1000",
                        "--prune-max-passes",
                        "0",
                    ]
                )

            self.assertEqual(rc, 0)
            written = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(written)
            self.assertTrue(all(row["radius"] in {500, 1500, 2000} for row in written))
            self.assertTrue(any(row["radius_rule"] == "built_up_dominant" for row in written))
            self.assertTrue(any(row["radius_rule"] in {"open_non_builtup", "sparse_non_builtup"} for row in written))
            self.assertTrue(any(bool(row["collected"]) for row in written))

    def test_visualizer_main_writes_terascope_popup_content(self) -> None:
        border_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [98.90, 18.75],
                                [99.05, 18.75],
                                [99.05, 18.90],
                                [98.90, 18.90],
                                [98.90, 18.75],
                            ]
                        ],
                    },
                }
            ],
        }
        records = [
            {
                "lat": 18.80,
                "lng": 98.98,
                "radius": 500,
                "collected": True,
                "land_cover_label": "Built-up",
                "built_up_share_pct": 60.0,
                "radius_rule": "built_up_dominant",
            },
            {
                "lat": 18.81,
                "lng": 98.99,
                "radius": 2000,
                "collected": False,
                "land_cover_label": "Tree cover",
                "built_up_share_pct": 0.0,
                "radius_rule": "sparse_non_builtup",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "lat_lng_radius.json"
            output_path = Path(temp_dir) / "collection_radius_map.html"
            border_path = Path(temp_dir) / "border.geojson"
            input_path.write_text(json.dumps(records), encoding="utf-8")
            border_path.write_text(json.dumps(border_geojson), encoding="utf-8")

            with (
                patch.object(terascope_vis, "INPUT_PATH", input_path),
                patch.object(terascope_vis, "OUTPUT_PATH", output_path),
                patch.object(terascope_vis, "BORDER_PATH", border_path),
            ):
                rc = terascope_vis.main()

            self.assertEqual(rc, 0)
            html = output_path.read_text(encoding="utf-8")
            self.assertIn("Land cover", html)
            self.assertIn("Built-up share", html)
            self.assertIn("Radius rule", html)
            self.assertIn("Chiang Mai Border", html)
            self.assertIn("Collection Status", html)
            self.assertIn(terascope_vis.COLLECTED_COLOR, html)
            self.assertIn(terascope_vis.NOT_COLLECTED_COLOR, html)


if __name__ == "__main__":
    unittest.main()
