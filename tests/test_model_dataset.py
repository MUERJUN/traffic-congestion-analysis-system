from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from traffic_congestion.data_pipeline import TrafficMatrix
from traffic_congestion.model_dataset import (
    FEATURE_COLUMNS,
    TimeSegment,
    build_sensor_samples,
    create_time_segments,
    fit_training_statistics,
    future_window_counts,
    validate_model_parquets,
    validate_segment_windows,
    write_model_datasets,
)


def constant_matrix(rows: int = 40, sensors: int = 2) -> TrafficMatrix:
    return TrafficMatrix(
        timestamps=pd.date_range("2012-03-01", periods=rows, freq="5min"),
        sensor_ids=np.array([f"S{index}" for index in range(sensors)]),
        speeds=np.full((rows, sensors), 60.0, dtype=float),
    )


class ModelDatasetTests(unittest.TestCase):
    def test_time_segments_are_contiguous_and_isolated(self) -> None:
        segments = create_time_segments(100)
        self.assertEqual([segment.observation_count for segment in segments], [70, 10, 20])
        validate_segment_windows(segments, history_steps=2, future_steps=2)
        self.assertEqual(segments[0].sample_end_index(2), 67)
        self.assertEqual(segments[1].sample_start_index(2), 72)

    def test_training_statistics_ignore_future_period_changes(self) -> None:
        matrix = constant_matrix()
        train = TimeSegment("train", 0, 27)
        baseline = fit_training_statistics(matrix, train)
        changed_speeds = matrix.speeds.copy()
        changed_speeds[28:] = 999.0
        changed = TrafficMatrix(matrix.timestamps, matrix.sensor_ids, changed_speeds)
        changed_statistics = fit_training_statistics(changed, train)
        np.testing.assert_allclose(
            baseline.free_flow_speed,
            changed_statistics.free_flow_speed,
        )
        np.testing.assert_allclose(
            baseline.period_median_speed,
            changed_statistics.period_median_speed,
        )

    def test_future_window_count_uses_offsets_after_current_only(self) -> None:
        state = np.array([False, True, True, True, False, False])
        counts = future_window_counts(state, future_steps=3)
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[2], 1)
        self.assertEqual(counts[3], -1)

    def test_future_changes_affect_label_but_not_same_time_features(self) -> None:
        matrix = constant_matrix(rows=45, sensors=1)
        statistics = fit_training_statistics(matrix, TimeSegment("train", 0, 19))
        full_segment = TimeSegment("test", 0, 34)
        baseline, _ = build_sensor_samples(
            matrix,
            0,
            full_segment,
            statistics,
            history_steps=12,
            future_steps=3,
            congestion_threshold=0.6,
            sensitivity_thresholds=[0.5, 0.6, 0.7],
            minimum_congested_future_steps=2,
        )

        changed_speeds = matrix.speeds.copy()
        changed_speeds[16:19, 0] = 20.0
        changed_matrix = TrafficMatrix(matrix.timestamps, matrix.sensor_ids, changed_speeds)
        changed, _ = build_sensor_samples(
            changed_matrix,
            0,
            full_segment,
            statistics,
            history_steps=12,
            future_steps=3,
            congestion_threshold=0.6,
            sensitivity_thresholds=[0.5, 0.6, 0.7],
            minimum_congested_future_steps=2,
        )

        timestamp = matrix.timestamps[15]
        baseline_row = baseline.loc[baseline["timestamp"] == timestamp].iloc[0]
        changed_row = changed.loc[changed["timestamp"] == timestamp].iloc[0]
        np.testing.assert_allclose(
            baseline_row[FEATURE_COLUMNS].to_numpy(dtype=float),
            changed_row[FEATURE_COLUMNS].to_numpy(dtype=float),
        )
        self.assertEqual(baseline_row["target_congestion_30m"], 0)
        self.assertEqual(changed_row["target_congestion_30m"], 1)

    def test_split_parquets_have_no_timestamp_overlap(self) -> None:
        matrix = constant_matrix(rows=120, sensors=2)
        segments = create_time_segments(matrix.speeds.shape[0], train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2)
        statistics = fit_training_statistics(matrix, segments[0])
        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = write_model_datasets(
                matrix,
                segments,
                statistics,
                Path(temporary_directory),
                history_steps=12,
                future_steps=2,
                congestion_threshold=0.6,
                sensitivity_thresholds=[0.5, 0.6, 0.7],
                minimum_congested_future_steps=1,
            )
            checks = validate_model_parquets(outputs, expected_sensor_count=2)
            ranges = []
            for name in ("train", "validation", "test"):
                table = pq.read_table(outputs[name]["path"], columns=["timestamp"]).to_pandas()
                ranges.append((table["timestamp"].min(), table["timestamp"].max()))

        self.assertLess(ranges[0][1], ranges[1][0])
        self.assertLess(ranges[1][1], ranges[2][0])
        self.assertTrue(all(item["status"] == "PASS" for item in checks.values()))


if __name__ == "__main__":
    unittest.main()
