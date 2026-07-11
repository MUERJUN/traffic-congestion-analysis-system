from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from traffic_congestion.data_pipeline import TrafficMatrix
from traffic_congestion.eda import (
    congestion_state,
    day_type_hour_patterns,
    descriptive_free_flow,
    finite_positive_speeds,
)


class EdaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = TrafficMatrix(
            timestamps=pd.date_range("2012-03-01", periods=4, freq="5min"),
            sensor_ids=np.array(["A", "B"]),
            speeds=np.array(
                [
                    [0.0, 10.0],
                    [20.0, 20.0],
                    [40.0, 30.0],
                    [60.0, 40.0],
                ]
            ),
        )

    def test_zero_is_missing_only_in_main_eda_view(self) -> None:
        valid = finite_positive_speeds(self.matrix)
        self.assertTrue(np.isnan(valid[0, 0]))
        self.assertEqual(self.matrix.speeds[0, 0], 0)

    def test_descriptive_congestion_uses_relative_sensor_speed(self) -> None:
        valid = finite_positive_speeds(self.matrix)
        free_flow = descriptive_free_flow(valid, quantile=0.75)
        state = congestion_state(valid, free_flow, threshold=0.60)
        self.assertFalse(state[0, 0])
        self.assertTrue(state[1, 0])
        self.assertTrue(state[0, 1])

    def test_day_type_aggregation_supports_current_pandas_index_behavior(self) -> None:
        valid = finite_positive_speeds(self.matrix)
        result = day_type_hour_patterns(self.matrix, valid)
        self.assertEqual(len(result), 48)
        self.assertEqual(set(result["day_type"]), {"Weekday", "Weekend"})


if __name__ == "__main__":
    unittest.main()
