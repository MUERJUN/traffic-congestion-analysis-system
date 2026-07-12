from __future__ import annotations

import unittest

from app.dashboard_utils import (
    ROOT,
    load_congestion_heatmap,
    load_sensor_edges,
    load_sensor_history,
    load_sensor_locations,
    load_train_reference,
    risk_level,
)


class DashboardUtilityTests(unittest.TestCase):
    def test_risk_level_boundaries(self) -> None:
        self.assertEqual(risk_level(0.0), "低风险")
        self.assertEqual(risk_level(0.3999), "低风险")
        self.assertEqual(risk_level(0.40), "中风险")
        self.assertEqual(risk_level(0.6999), "中风险")
        self.assertEqual(risk_level(0.70), "高风险")


    def test_sensor_locations_available_for_map(self) -> None:
        locations = load_sensor_locations()
        self.assertEqual(len(locations), 207)
        self.assertTrue({"sensor_id", "latitude", "longitude"}.issubset(locations.columns))

    def test_sensor_edges_available_for_corridor_map(self) -> None:
        edges = load_sensor_edges()
        self.assertGreater(len(edges), 300)
        self.assertTrue({"from_sensor_id", "to_sensor_id", "distance"}.issubset(edges.columns))

    def test_compact_replay_data_supports_prediction_page(self) -> None:
        self.assertTrue((ROOT / "data/dashboard/test_replay.parquet").exists())
        history = load_sensor_history("716339")
        self.assertGreaterEqual(len(history), 12)
        self.assertIn("target_congestion_30m", history.columns)

    def test_compact_heatmap_and_train_reference_are_available(self) -> None:
        self.assertTrue((ROOT / "data/dashboard/congestion_heatmap.parquet").exists())
        self.assertTrue((ROOT / "data/dashboard/train_reference.parquet").exists())
        heatmap = load_congestion_heatmap()
        reference = load_train_reference()
        self.assertFalse(heatmap.empty)
        self.assertTrue({"date", "sensor_id", "hour", "congestion_rate"}.issubset(heatmap.columns))
        self.assertFalse(reference.empty)


if __name__ == "__main__":
    unittest.main()
