"""Leakage-safe time splits, labels, and historical features for model input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data_pipeline import TrafficMatrix, file_sha256


@dataclass(frozen=True)
class TimeSegment:
    name: str
    start_index: int
    end_index: int

    @property
    def observation_count(self) -> int:
        return self.end_index - self.start_index + 1

    def sample_start_index(self, history_steps: int) -> int:
        return self.start_index + history_steps

    def sample_end_index(self, future_steps: int) -> int:
        return self.end_index - future_steps


@dataclass(frozen=True)
class TrainingStatistics:
    free_flow_speed: np.ndarray
    overall_median_speed: np.ndarray
    overall_congestion_rate: np.ndarray
    period_median_speed: np.ndarray
    period_congestion_rate: np.ndarray


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "current_speed_mph",
    "free_flow_speed_train_mph",
    "current_speed_ratio",
    "lag_speed_ratio_5m",
    "lag_speed_ratio_10m",
    "lag_speed_ratio_15m",
    "lag_speed_ratio_30m",
    "lag_speed_ratio_60m",
    "rolling_speed_ratio_mean_15m",
    "rolling_speed_ratio_min_15m",
    "rolling_speed_ratio_std_15m",
    "rolling_speed_ratio_mean_30m",
    "rolling_speed_ratio_min_30m",
    "rolling_speed_ratio_std_30m",
    "rolling_speed_ratio_mean_60m",
    "rolling_speed_ratio_min_60m",
    "rolling_speed_ratio_std_60m",
    "trend_speed_mph_per_min_10m",
    "trend_speed_mph_per_min_15m",
    "trend_speed_mph_per_min_30m",
    "speed_change_5m",
    "consecutive_decline_steps",
    "historical_period_median_speed_mph_train",
    "historical_period_congestion_rate_train",
    "deviation_from_historical_median_mph",
]


def create_time_segments(
    observation_count: int,
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> tuple[TimeSegment, TimeSegment, TimeSegment]:
    """Create deterministic contiguous segments covering the complete time axis."""

    ratios = np.array([train_ratio, validation_ratio, test_ratio], dtype=float)
    if observation_count < 3:
        raise ValueError("At least three observations are required")
    if (ratios <= 0).any() or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Split ratios must be positive and sum to 1")

    train_count = int(observation_count * train_ratio)
    validation_count = int(observation_count * validation_ratio)
    test_count = observation_count - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("A split would be empty")

    train = TimeSegment("train", 0, train_count - 1)
    validation = TimeSegment(
        "validation",
        train.end_index + 1,
        train.end_index + validation_count,
    )
    test = TimeSegment("test", validation.end_index + 1, observation_count - 1)
    return train, validation, test


def validate_segment_windows(
    segments: Iterable[TimeSegment],
    *,
    history_steps: int,
    future_steps: int,
) -> None:
    """Ensure every segment can contain isolated history and target windows."""

    previous_end = -1
    for segment in segments:
        if segment.start_index != previous_end + 1:
            raise ValueError("Time segments are not contiguous")
        if segment.observation_count <= history_steps + future_steps:
            raise ValueError(f"Segment {segment.name} is too short for isolated windows")
        if segment.sample_start_index(history_steps) - history_steps < segment.start_index:
            raise ValueError("History window crosses a segment boundary")
        if segment.sample_end_index(future_steps) + future_steps > segment.end_index:
            raise ValueError("Future target window crosses a segment boundary")
        previous_end = segment.end_index


def valid_speed_matrix(matrix: TrafficMatrix) -> np.ndarray:
    return np.where(
        np.isfinite(matrix.speeds) & (matrix.speeds > 0),
        matrix.speeds,
        np.nan,
    )


def fit_training_statistics(
    matrix: TrafficMatrix,
    train_segment: TimeSegment,
    *,
    free_flow_quantile: float = 0.85,
    congestion_threshold: float = 0.60,
) -> TrainingStatistics:
    """Fit every reusable statistic from training observations only."""

    if train_segment.start_index != 0:
        raise ValueError("The training segment must start at index 0")
    train_slice = slice(train_segment.start_index, train_segment.end_index + 1)
    train_speed = valid_speed_matrix(matrix)[train_slice]
    free_flow = np.nanquantile(train_speed, free_flow_quantile, axis=0)
    if (~np.isfinite(free_flow) | (free_flow <= 0)).any():
        raise ValueError("One or more sensors have no valid training free-flow estimate")

    overall_median = np.nanmedian(train_speed, axis=0)
    train_ratio = train_speed / free_flow[np.newaxis, :]
    valid_train_ratio = np.isfinite(train_ratio)
    overall_congestion = np.divide(
        (valid_train_ratio & (train_ratio < congestion_threshold)).sum(axis=0),
        valid_train_ratio.sum(axis=0),
    )

    train_timestamps = matrix.timestamps[train_slice]
    day_type = np.asarray(train_timestamps.dayofweek >= 5, dtype=np.int8)
    slot = train_timestamps.hour.to_numpy() * 12 + train_timestamps.minute.to_numpy() // 5
    sensor_count = train_speed.shape[1]
    period_median = np.full((sensor_count, 2, 288), np.nan, dtype=np.float64)
    period_congestion = np.full((sensor_count, 2, 288), np.nan, dtype=np.float64)

    for is_weekend in (0, 1):
        for time_slot in range(288):
            selected = (day_type == is_weekend) & (slot == time_slot)
            if not selected.any():
                continue
            group_speed = train_speed[selected]
            group_ratio = train_ratio[selected]
            period_median[:, is_weekend, time_slot] = np.nanmedian(group_speed, axis=0)
            valid_group = np.isfinite(group_ratio)
            congestion_group = valid_group & (group_ratio < congestion_threshold)
            valid_counts = valid_group.sum(axis=0)
            rates = np.divide(
                congestion_group.sum(axis=0),
                valid_counts,
                out=np.full(sensor_count, np.nan, dtype=float),
                where=valid_counts > 0,
            )
            period_congestion[:, is_weekend, time_slot] = rates

    for sensor_index in range(sensor_count):
        median_missing = ~np.isfinite(period_median[sensor_index])
        rate_missing = ~np.isfinite(period_congestion[sensor_index])
        period_median[sensor_index][median_missing] = overall_median[sensor_index]
        period_congestion[sensor_index][rate_missing] = overall_congestion[sensor_index]

    return TrainingStatistics(
        free_flow_speed=free_flow,
        overall_median_speed=overall_median,
        overall_congestion_rate=overall_congestion,
        period_median_speed=period_median,
        period_congestion_rate=period_congestion,
    )


def future_window_counts(state: np.ndarray, future_steps: int) -> np.ndarray:
    """Count True values in offsets +1 through +future_steps for every index."""

    values = state.astype(np.int64)
    prefix = np.concatenate(([0], np.cumsum(values)))
    counts = np.full(len(values), -1, dtype=np.int16)
    indices = np.arange(0, len(values) - future_steps)
    counts[indices] = prefix[indices + future_steps + 1] - prefix[indices + 1]
    return counts


def complete_future_mask(valid: np.ndarray, future_steps: int) -> np.ndarray:
    counts = future_window_counts(valid.astype(bool), future_steps)
    return counts == future_steps


def complete_history_mask(valid: np.ndarray, history_steps: int) -> np.ndarray:
    """Require offsets -history_steps through 0 to be valid."""

    window = history_steps + 1
    counts = (
        pd.Series(valid.astype(np.int8))
        .rolling(window=window, min_periods=window)
        .sum()
        .to_numpy()
    )
    return counts == window


def _shift(values: np.ndarray, steps: int) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    result[steps:] = values[:-steps]
    return result


def _consecutive_decline_steps(speed: np.ndarray) -> np.ndarray:
    result = np.zeros(len(speed), dtype=np.int16)
    for index in range(1, len(speed)):
        if (
            np.isfinite(speed[index])
            and np.isfinite(speed[index - 1])
            and speed[index] < speed[index - 1]
        ):
            result[index] = result[index - 1] + 1
    return result


def build_sensor_samples(
    matrix: TrafficMatrix,
    sensor_index: int,
    segment: TimeSegment,
    statistics: TrainingStatistics,
    *,
    history_steps: int,
    future_steps: int,
    congestion_threshold: float,
    sensitivity_thresholds: Iterable[float],
    minimum_congested_future_steps: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one sensor's samples using only current and past inputs."""

    if history_steps < 12:
        raise ValueError("The configured feature set requires at least 12 history steps")

    raw_speed = matrix.speeds[:, sensor_index].astype(np.float64)
    speed = np.where(np.isfinite(raw_speed) & (raw_speed > 0), raw_speed, np.nan)
    free_flow = statistics.free_flow_speed[sensor_index]
    ratio = speed / free_flow
    valid = np.isfinite(speed)
    history_complete = complete_history_mask(valid, history_steps)
    future_complete = complete_future_mask(valid, future_steps)

    candidate = np.zeros(len(speed), dtype=bool)
    sample_start = segment.sample_start_index(history_steps)
    sample_end = segment.sample_end_index(future_steps)
    candidate[sample_start : sample_end + 1] = True
    after_history = candidate & history_complete
    usable = after_history & future_complete
    sample_indices = np.flatnonzero(usable)

    labels_by_threshold: dict[float, np.ndarray] = {}
    for threshold in sensitivity_thresholds:
        state = np.isfinite(ratio) & (ratio < threshold)
        future_congested = future_window_counts(state, future_steps)
        labels_by_threshold[float(threshold)] = (
            future_congested >= minimum_congested_future_steps
        )
    main_labels = labels_by_threshold[float(congestion_threshold)]

    series = pd.Series(ratio)
    rolling: dict[tuple[int, str], np.ndarray] = {}
    for minutes, steps in ((15, 3), (30, 6), (60, 12)):
        window = series.rolling(window=steps, min_periods=steps)
        rolling[(minutes, "mean")] = window.mean().to_numpy()
        rolling[(minutes, "min")] = window.min().to_numpy()
        rolling[(minutes, "std")] = window.std(ddof=0).to_numpy()

    timestamps = matrix.timestamps
    hours = timestamps.hour.to_numpy()
    weekdays = timestamps.dayofweek.to_numpy()
    weekends = np.asarray(weekdays >= 5, dtype=np.int8)
    slots = hours * 12 + timestamps.minute.to_numpy() // 5
    historical_median = statistics.period_median_speed[
        sensor_index, weekends, slots
    ]
    historical_rate = statistics.period_congestion_rate[
        sensor_index, weekends, slots
    ]

    frame = pd.DataFrame(
        {
            "timestamp": timestamps[sample_indices],
            "sensor_id": str(matrix.sensor_ids[sensor_index]),
            "split": segment.name,
            "target_congestion_30m": main_labels[sample_indices].astype(np.int8),
            "hour": hours[sample_indices].astype(np.int8),
            "day_of_week": weekdays[sample_indices].astype(np.int8),
            "is_weekend": weekends[sample_indices].astype(np.int8),
            "hour_sin": np.sin(2 * np.pi * hours[sample_indices] / 24),
            "hour_cos": np.cos(2 * np.pi * hours[sample_indices] / 24),
            "day_of_week_sin": np.sin(2 * np.pi * weekdays[sample_indices] / 7),
            "day_of_week_cos": np.cos(2 * np.pi * weekdays[sample_indices] / 7),
            "current_speed_mph": speed[sample_indices],
            "free_flow_speed_train_mph": free_flow,
            "current_speed_ratio": ratio[sample_indices],
            "lag_speed_ratio_5m": _shift(ratio, 1)[sample_indices],
            "lag_speed_ratio_10m": _shift(ratio, 2)[sample_indices],
            "lag_speed_ratio_15m": _shift(ratio, 3)[sample_indices],
            "lag_speed_ratio_30m": _shift(ratio, 6)[sample_indices],
            "lag_speed_ratio_60m": _shift(ratio, 12)[sample_indices],
            "rolling_speed_ratio_mean_15m": rolling[(15, "mean")][sample_indices],
            "rolling_speed_ratio_min_15m": rolling[(15, "min")][sample_indices],
            "rolling_speed_ratio_std_15m": rolling[(15, "std")][sample_indices],
            "rolling_speed_ratio_mean_30m": rolling[(30, "mean")][sample_indices],
            "rolling_speed_ratio_min_30m": rolling[(30, "min")][sample_indices],
            "rolling_speed_ratio_std_30m": rolling[(30, "std")][sample_indices],
            "rolling_speed_ratio_mean_60m": rolling[(60, "mean")][sample_indices],
            "rolling_speed_ratio_min_60m": rolling[(60, "min")][sample_indices],
            "rolling_speed_ratio_std_60m": rolling[(60, "std")][sample_indices],
            "trend_speed_mph_per_min_10m": (
                speed - _shift(speed, 2)
            )[sample_indices]
            / 10,
            "trend_speed_mph_per_min_15m": (
                speed - _shift(speed, 3)
            )[sample_indices]
            / 15,
            "trend_speed_mph_per_min_30m": (
                speed - _shift(speed, 6)
            )[sample_indices]
            / 30,
            "speed_change_5m": (speed - _shift(speed, 1))[sample_indices],
            "consecutive_decline_steps": _consecutive_decline_steps(speed)[sample_indices],
            "historical_period_median_speed_mph_train": historical_median[sample_indices],
            "historical_period_congestion_rate_train": historical_rate[sample_indices],
            "deviation_from_historical_median_mph": (
                speed - historical_median
            )[sample_indices],
        }
    )

    if frame[FEATURE_COLUMNS].isna().any(axis=None):
        missing_columns = frame[FEATURE_COLUMNS].columns[
            frame[FEATURE_COLUMNS].isna().any()
        ].tolist()
        raise RuntimeError(f"Unexpected missing model features: {missing_columns}")

    summary = {
        "candidate_after_boundary": int(candidate.sum()),
        "excluded_incomplete_history": int((candidate & ~history_complete).sum()),
        "excluded_incomplete_future": int((after_history & ~future_complete).sum()),
        "usable_samples": int(len(frame)),
        "positive_labels": {
            str(threshold): int(labels[sample_indices].sum())
            for threshold, labels in labels_by_threshold.items()
        },
    }
    return frame, summary


def _statistics_tables(
    matrix: TrafficMatrix,
    statistics: TrainingStatistics,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensor_table = pd.DataFrame(
        {
            "sensor_id": matrix.sensor_ids,
            "free_flow_speed_train_mph": statistics.free_flow_speed,
            "overall_median_speed_train_mph": statistics.overall_median_speed,
            "overall_congestion_rate_train": statistics.overall_congestion_rate,
        }
    )
    period_rows: list[pd.DataFrame] = []
    for sensor_index, sensor_id in enumerate(matrix.sensor_ids):
        weekend, slot = np.indices((2, 288))
        period_rows.append(
            pd.DataFrame(
                {
                    "sensor_id": str(sensor_id),
                    "is_weekend": weekend.ravel().astype(np.int8),
                    "time_slot": slot.ravel().astype(np.int16),
                    "median_speed_train_mph": statistics.period_median_speed[
                        sensor_index
                    ].ravel(),
                    "congestion_rate_train": statistics.period_congestion_rate[
                        sensor_index
                    ].ravel(),
                }
            )
        )
    return sensor_table, pd.concat(period_rows, ignore_index=True)


def write_model_datasets(
    matrix: TrafficMatrix,
    segments: Iterable[TimeSegment],
    statistics: TrainingStatistics,
    output_dir: Path,
    *,
    history_steps: int,
    future_steps: int,
    congestion_threshold: float,
    sensitivity_thresholds: Iterable[float],
    minimum_congested_future_steps: int,
) -> dict[str, Any]:
    """Write one Parquet dataset per split, chunked by sensor."""

    output_dir.mkdir(parents=True, exist_ok=True)
    segments = tuple(segments)
    writers: dict[str, pq.ParquetWriter] = {}
    temporary_paths = {
        segment.name: output_dir / f"{segment.name}_features.parquet.tmp"
        for segment in segments
    }
    final_paths = {
        segment.name: output_dir / f"{segment.name}_features.parquet"
        for segment in segments
    }
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)

    summaries = {
        segment.name: {
            "candidate_after_boundary": 0,
            "excluded_incomplete_history": 0,
            "excluded_incomplete_future": 0,
            "usable_samples": 0,
            "positive_labels": {str(float(value)): 0 for value in sensitivity_thresholds},
            "first_timestamp": None,
            "last_timestamp": None,
        }
        for segment in segments
    }

    try:
        for sensor_index in range(len(matrix.sensor_ids)):
            for segment in segments:
                frame, sensor_summary = build_sensor_samples(
                    matrix,
                    sensor_index,
                    segment,
                    statistics,
                    history_steps=history_steps,
                    future_steps=future_steps,
                    congestion_threshold=congestion_threshold,
                    sensitivity_thresholds=sensitivity_thresholds,
                    minimum_congested_future_steps=minimum_congested_future_steps,
                )
                summary = summaries[segment.name]
                for key in (
                    "candidate_after_boundary",
                    "excluded_incomplete_history",
                    "excluded_incomplete_future",
                    "usable_samples",
                ):
                    summary[key] += sensor_summary[key]
                for threshold, count in sensor_summary["positive_labels"].items():
                    summary["positive_labels"][threshold] += count

                if frame.empty:
                    continue
                first = frame["timestamp"].min()
                last = frame["timestamp"].max()
                if summary["first_timestamp"] is None or first < summary["first_timestamp"]:
                    summary["first_timestamp"] = first
                if summary["last_timestamp"] is None or last > summary["last_timestamp"]:
                    summary["last_timestamp"] = last

                table = pa.Table.from_pandas(frame, preserve_index=False)
                if segment.name not in writers:
                    writers[segment.name] = pq.ParquetWriter(
                        temporary_paths[segment.name],
                        table.schema,
                        compression="zstd",
                        use_dictionary=["sensor_id", "split"],
                    )
                writers[segment.name].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()

    output_info: dict[str, Any] = {}
    for segment in segments:
        name = segment.name
        if name not in writers:
            raise RuntimeError(f"No samples were written for split {name}")
        temporary_paths[name].replace(final_paths[name])
        metadata = pq.ParquetFile(final_paths[name]).metadata
        if metadata.num_rows != summaries[name]["usable_samples"]:
            raise RuntimeError(f"Parquet row mismatch for split {name}")
        summary = summaries[name]
        summary["first_timestamp"] = str(summary["first_timestamp"])
        summary["last_timestamp"] = str(summary["last_timestamp"])
        for threshold, count in summary["positive_labels"].items():
            summary.setdefault("positive_rates", {})[threshold] = (
                count / summary["usable_samples"]
            )
        output_info[name] = {
            **summary,
            "path": str(final_paths[name]),
            "bytes": int(final_paths[name].stat().st_size),
            "sha256": file_sha256(final_paths[name]),
            "columns": int(metadata.num_columns),
        }
    return output_info


def write_training_statistics(
    matrix: TrafficMatrix,
    statistics: TrainingStatistics,
    output_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    sensor_table, period_table = _statistics_tables(matrix, statistics)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    sensor_path = report_dir / "training_sensor_statistics.csv"
    period_path = output_dir / "training_period_statistics.parquet"
    sensor_table.to_csv(sensor_path, index=False, encoding="utf-8-sig")
    period_table.to_parquet(period_path, index=False, compression="zstd")
    return {
        "sensor_statistics_path": str(sensor_path),
        "sensor_rows": int(len(sensor_table)),
        "period_statistics_path": str(period_path),
        "period_rows": int(len(period_table)),
        "period_file_sha256": file_sha256(period_path),
    }


def validate_model_parquets(
    split_outputs: dict[str, Any],
    *,
    expected_sensor_count: int,
) -> dict[str, Any]:
    """Validate stored split files without loading every feature into memory."""

    required_columns = {
        "timestamp",
        "sensor_id",
        "split",
        "target_congestion_30m",
        *FEATURE_COLUMNS,
    }
    results: dict[str, Any] = {}
    for split_name, output in split_outputs.items():
        parquet_file = pq.ParquetFile(output["path"])
        schema_names = parquet_file.schema_arrow.names
        missing_columns = sorted(required_columns - set(schema_names))
        null_count = 0
        for row_group_index in range(parquet_file.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                statistics = row_group.column(column_index).statistics
                if statistics is not None and statistics.null_count is not None:
                    null_count += statistics.null_count

        key_unique = True
        sensors_seen: set[str] = set()
        target_domain: set[int] = set()
        for row_group_index in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(
                row_group_index,
                columns=["timestamp", "sensor_id", "target_congestion_30m"],
            )
            sensors = set(table["sensor_id"].to_pylist())
            if len(sensors) != 1:
                key_unique = False
                continue
            sensor = str(next(iter(sensors)))
            if sensor in sensors_seen:
                key_unique = False
            sensors_seen.add(sensor)
            timestamps = table["timestamp"].to_pylist()
            if len(timestamps) != len(set(timestamps)):
                key_unique = False
            target_domain.update(int(value) for value in table["target_congestion_30m"].to_pylist())

        parquet_file.close()

        checks = {
            "rows_match_summary": parquet_file.metadata.num_rows == output["usable_samples"],
            "column_count": len(schema_names),
            "missing_required_columns": missing_columns,
            "total_null_values": int(null_count),
            "timestamp_sensor_keys_unique": key_unique,
            "sensor_count": len(sensors_seen),
            "target_domain": sorted(target_domain),
            "contains_both_target_classes": target_domain == {0, 1},
        }
        checks["status"] = "PASS" if (
            checks["rows_match_summary"]
            and not missing_columns
            and null_count == 0
            and key_unique
            and len(sensors_seen) == expected_sensor_count
            and bool(target_domain)
            and target_domain.issubset({0, 1})
        ) else "FAIL"
        results[split_name] = checks

    if any(result["status"] != "PASS" for result in results.values()):
        raise RuntimeError(f"Stored model dataset validation failed: {results}")
    return results
