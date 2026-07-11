from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from traffic_congestion.data_pipeline import (
    TrafficMatrix,
    align_sensor_locations,
    build_quality_masks,
    mark_long_true_runs,
    validate_traffic_matrix,
    write_clean_long_parquet,
)


def sample_matrix() -> TrafficMatrix:
    timestamps = pd.date_range("2012-03-01", periods=6, freq="5min")
    sensor_ids = np.array(["A", "B", "C"])
    speeds = np.array(
        [
            [60.0, 55.0, 50.0],
            [0.0, 0.0, 0.0],
            [0.0, 45.0, 0.0],
            [20.0, 44.0, 0.0],
            [65.0, 43.0, 48.0],
            [64.0, 42.0, 47.0],
        ]
    )
    return TrafficMatrix(timestamps=timestamps, sensor_ids=sensor_ids, speeds=speeds)


class DataPipelineTests(unittest.TestCase):
    def test_validate_accepts_continuous_matrix(self) -> None:
        result = validate_traffic_matrix(sample_matrix())
        self.assertEqual(result["rows"], 6)
        self.assertEqual(result["sensors"], 3)
        self.assertEqual(result["observations"], 18)

    def test_validate_rejects_time_gap(self) -> None:
        matrix = sample_matrix()
        bad_timestamps = matrix.timestamps.delete(2).append(
            pd.DatetimeIndex([matrix.timestamps[-1] + pd.Timedelta(minutes=5)])
        )
        bad = TrafficMatrix(
            timestamps=bad_timestamps,
            sensor_ids=matrix.sensor_ids,
            speeds=matrix.speeds,
        )
        with self.assertRaisesRegex(ValueError, "Non-five-minute"):
            validate_traffic_matrix(bad)

    def test_long_run_mask_marks_complete_runs(self) -> None:
        mask = np.array(
            [
                [False, True],
                [True, True],
                [True, False],
                [False, True],
            ]
        )
        result = mark_long_true_runs(mask, min_run_steps=2)
        expected = np.array(
            [
                [False, True],
                [True, True],
                [True, False],
                [False, False],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_quality_masks_keep_quality_concepts_separate(self) -> None:
        quality = build_quality_masks(
            sample_matrix(),
            majority_threshold=0.5,
            long_zero_run_steps=2,
            large_jump_mph=40,
        )
        self.assertTrue(quality.is_systemwide_zero[1])
        self.assertTrue(quality.is_majority_zero[2])
        self.assertTrue(quality.is_long_sensor_zero_run[1, 0])
        self.assertTrue(quality.is_long_sensor_zero_run[3, 2])
        self.assertTrue(quality.is_large_jump_from_zero[1, 0])
        self.assertTrue(quality.is_large_jump[4, 0])
        self.assertFalse(quality.is_large_jump_from_zero[4, 0])

    def test_parquet_export_preserves_all_rows_and_raw_zero(self) -> None:
        matrix = sample_matrix()
        quality = build_quality_masks(matrix, long_zero_run_steps=2)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            locations_path = root / "locations.csv"
            pd.DataFrame(
                {
                    "sensor_id": ["C", "A", "B"],
                    "latitude": [3.0, 1.0, 2.0],
                    "longitude": [-3.0, -1.0, -2.0],
                }
            ).to_csv(locations_path, index=False)
            locations = align_sensor_locations(matrix.sensor_ids, locations_path)
            output_path = root / "clean.parquet"
            result = write_clean_long_parquet(
                matrix,
                quality,
                locations,
                output_path,
                chunk_time_steps=2,
            )
            table = pq.read_table(output_path).to_pandas()

        self.assertEqual(result["rows"], matrix.speeds.size)
        self.assertEqual(len(table), matrix.speeds.size)
        zero_row = table.loc[
            (table["timestamp"] == matrix.timestamps[1]) & (table["sensor_id"] == "A")
        ].iloc[0]
        self.assertEqual(zero_row["speed_raw"], 0)
        self.assertTrue(pd.isna(zero_row["speed_valid"]))
        self.assertTrue(zero_row["is_systemwide_zero"])


if __name__ == "__main__":
    unittest.main()
