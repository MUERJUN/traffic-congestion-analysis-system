"""Read-only METR-LA loading, validation, quality flags, and long-table export."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrafficMatrix:
    """In-memory representation of the original time-by-sensor speed matrix."""

    timestamps: pd.DatetimeIndex
    sensor_ids: np.ndarray
    speeds: np.ndarray


@dataclass(frozen=True)
class QualityMasks:
    """Quality flags derived without modifying the original values."""

    is_zero: np.ndarray
    is_systemwide_zero: np.ndarray
    is_majority_zero: np.ndarray
    is_long_sensor_zero_run: np.ndarray
    is_large_jump: np.ndarray
    is_large_jump_from_zero: np.ndarray


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest for a local file."""

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_metr_la_h5(path: Path) -> TrafficMatrix:
    """Load the fixed-format Pandas HDF5 file with h5py.

    This avoids a PyTables runtime dependency and keeps the original file read-only.
    """

    with h5py.File(path, "r") as h5_file:
        required = {
            "df/axis0",
            "df/axis1",
            "df/block0_items",
            "df/block0_values",
        }
        missing: list[str] = []
        for node in required:
            if node not in h5_file:
                missing.append(node)
        if missing:
            raise ValueError(f"HDF5 is missing required nodes: {sorted(set(missing))}")

        sensor_ids = np.asarray(
            [value.decode("utf-8") for value in h5_file["df/axis0"][...]],
            dtype=str,
        )
        block_items = np.asarray(
            [value.decode("utf-8") for value in h5_file["df/block0_items"][...]],
            dtype=str,
        )
        if not np.array_equal(sensor_ids, block_items):
            raise ValueError("HDF5 axis0 and block0_items sensor order differs")

        timestamps = pd.DatetimeIndex(pd.to_datetime(h5_file["df/axis1"][...]))
        speeds = np.asarray(h5_file["df/block0_values"][...], dtype=np.float64)

    matrix = TrafficMatrix(timestamps=timestamps, sensor_ids=sensor_ids, speeds=speeds)
    validate_traffic_matrix(matrix)
    return matrix


def validate_traffic_matrix(matrix: TrafficMatrix) -> dict[str, Any]:
    """Validate structural invariants and return a compact audit dictionary."""

    rows, columns = matrix.speeds.shape
    if rows != len(matrix.timestamps) or columns != len(matrix.sensor_ids):
        raise ValueError("Speed matrix shape does not match timestamp/sensor axes")
    if matrix.timestamps.has_duplicates:
        raise ValueError("Duplicate timestamps detected")
    if not matrix.timestamps.is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonically increasing")
    if pd.Index(matrix.sensor_ids).has_duplicates:
        raise ValueError("Duplicate sensor IDs detected")

    intervals = matrix.timestamps.to_series().diff().dropna()
    expected_interval = pd.Timedelta(minutes=5)
    if not intervals.eq(expected_interval).all():
        bad = intervals.loc[~intervals.eq(expected_interval)]
        raise ValueError(f"Non-five-minute intervals detected: {bad.head().to_dict()}")

    return {
        "rows": int(rows),
        "sensors": int(columns),
        "observations": int(matrix.speeds.size),
        "start": str(matrix.timestamps.min()),
        "end": str(matrix.timestamps.max()),
        "interval_minutes": 5,
        "duplicate_timestamps": 0,
        "duplicate_sensor_ids": 0,
    }


def mark_long_true_runs(mask: np.ndarray, min_run_steps: int) -> np.ndarray:
    """Mark every cell belonging to a True run at least ``min_run_steps`` long."""

    if mask.ndim != 2:
        raise ValueError("Run mask must be a two-dimensional time-by-sensor array")
    if min_run_steps < 1:
        raise ValueError("min_run_steps must be positive")

    result = np.zeros_like(mask, dtype=bool)
    for sensor_index in range(mask.shape[1]):
        column = mask[:, sensor_index]
        padded = np.concatenate(([False], column, [False]))
        boundaries = np.flatnonzero(padded[1:] != padded[:-1])
        starts = boundaries[0::2]
        ends = boundaries[1::2]
        for start, end in zip(starts, ends, strict=True):
            if end - start >= min_run_steps:
                result[start:end, sensor_index] = True
    return result


def build_quality_masks(
    matrix: TrafficMatrix,
    *,
    majority_threshold: float = 0.5,
    long_zero_run_steps: int = 12,
    large_jump_mph: float = 40.0,
) -> QualityMasks:
    """Build explicit quality flags while leaving raw values unchanged."""

    if not 0 < majority_threshold <= 1:
        raise ValueError("majority_threshold must be in (0, 1]")

    is_zero = matrix.speeds == 0
    zero_counts = is_zero.sum(axis=1)
    majority_count = int(np.ceil(matrix.speeds.shape[1] * majority_threshold))
    is_systemwide_zero = zero_counts == matrix.speeds.shape[1]
    is_majority_zero = zero_counts >= majority_count
    is_long_sensor_zero_run = mark_long_true_runs(is_zero, long_zero_run_steps)

    is_large_jump = np.zeros_like(is_zero, dtype=bool)
    is_large_jump[1:] = np.abs(np.diff(matrix.speeds, axis=0)) > large_jump_mph
    is_large_jump_from_zero = np.zeros_like(is_zero, dtype=bool)
    is_large_jump_from_zero[1:] = is_large_jump[1:] & (
        is_zero[1:] | is_zero[:-1]
    )

    return QualityMasks(
        is_zero=is_zero,
        is_systemwide_zero=is_systemwide_zero,
        is_majority_zero=is_majority_zero,
        is_long_sensor_zero_run=is_long_sensor_zero_run,
        is_large_jump=is_large_jump,
        is_large_jump_from_zero=is_large_jump_from_zero,
    )


def align_sensor_locations(sensor_ids: np.ndarray, locations_path: Path) -> pd.DataFrame:
    """Return sensor locations in exactly the same order as the HDF5 columns."""

    locations = pd.read_csv(locations_path, dtype={"sensor_id": str})
    required = {"sensor_id", "latitude", "longitude"}
    if not required.issubset(locations.columns):
        raise ValueError(f"Location file missing columns: {sorted(required - set(locations.columns))}")
    if locations["sensor_id"].duplicated().any():
        raise ValueError("Location file contains duplicate sensor IDs")

    aligned = locations.set_index("sensor_id").reindex(sensor_ids)
    aligned.index.name = "sensor_id"
    return aligned.reset_index()


def write_clean_long_parquet(
    matrix: TrafficMatrix,
    quality: QualityMasks,
    locations: pd.DataFrame,
    output_path: Path,
    *,
    chunk_time_steps: int = 2016,
) -> dict[str, Any]:
    """Write an EDA-ready long table in bounded-memory Parquet chunks."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if chunk_time_steps < 1:
        raise ValueError("chunk_time_steps must be positive")

    location_index = locations.set_index("sensor_id").reindex(matrix.sensor_ids)
    latitude = location_index["latitude"].to_numpy(dtype=np.float32)
    longitude = location_index["longitude"].to_numpy(dtype=np.float32)
    has_location = np.isfinite(latitude) & np.isfinite(longitude)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0

    try:
        sensor_count = len(matrix.sensor_ids)
        for start in range(0, len(matrix.timestamps), chunk_time_steps):
            end = min(start + chunk_time_steps, len(matrix.timestamps))
            timestamps = matrix.timestamps[start:end]
            raw = matrix.speeds[start:end]
            valid = np.where(np.isfinite(raw) & (raw > 0), raw, np.nan).astype(np.float32)
            repeated_timestamps = np.repeat(timestamps.to_numpy(), sensor_count)
            repeated_sensor_ids = np.tile(matrix.sensor_ids, len(timestamps))

            table = pa.table(
                {
                    "timestamp": pa.array(repeated_timestamps, type=pa.timestamp("ns")),
                    "sensor_id": pa.array(repeated_sensor_ids).dictionary_encode(),
                    "speed_raw": pa.array(raw.astype(np.float32).ravel(order="C")),
                    "speed_valid": pa.array(valid.ravel(order="C"), from_pandas=True),
                    "hour": pa.array(
                        np.repeat(timestamps.hour.to_numpy(dtype=np.int8), sensor_count)
                    ),
                    "day_of_week": pa.array(
                        np.repeat(timestamps.dayofweek.to_numpy(dtype=np.int8), sensor_count)
                    ),
                    "is_weekend": pa.array(
                        np.repeat((timestamps.dayofweek >= 5), sensor_count)
                    ),
                    "latitude": pa.array(np.tile(latitude, len(timestamps)), from_pandas=True),
                    "longitude": pa.array(np.tile(longitude, len(timestamps)), from_pandas=True),
                    "is_zero_observation": pa.array(
                        quality.is_zero[start:end].ravel(order="C")
                    ),
                    "is_systemwide_zero": pa.array(
                        np.repeat(quality.is_systemwide_zero[start:end], sensor_count)
                    ),
                    "is_majority_zero": pa.array(
                        np.repeat(quality.is_majority_zero[start:end], sensor_count)
                    ),
                    "is_long_sensor_zero_run": pa.array(
                        quality.is_long_sensor_zero_run[start:end].ravel(order="C")
                    ),
                    "is_large_jump": pa.array(
                        quality.is_large_jump[start:end].ravel(order="C")
                    ),
                    "is_large_jump_from_zero": pa.array(
                        quality.is_large_jump_from_zero[start:end].ravel(order="C")
                    ),
                    "has_valid_location": pa.array(np.tile(has_location, len(timestamps))),
                }
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression="zstd",
                    use_dictionary=["sensor_id"],
                )
            writer.write_table(table)
            total_rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if total_rows != matrix.speeds.size:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Long table row mismatch: expected {matrix.speeds.size}, wrote {total_rows}"
        )
    temporary_path.replace(output_path)

    return {
        "path": str(output_path),
        "rows": int(total_rows),
        "columns": 16,
        "bytes": int(output_path.stat().st_size),
        "sha256": file_sha256(output_path),
    }


def quality_summary(
    matrix: TrafficMatrix,
    quality: QualityMasks,
    *,
    source_sha256: str,
    parquet_info: dict[str, Any],
    long_zero_run_steps: int,
    large_jump_mph: float,
) -> dict[str, Any]:
    """Create a serializable summary of the Phase 3 cleaning view."""

    positive = np.isfinite(matrix.speeds) & (matrix.speeds > 0)
    return {
        "source_sha256": source_sha256,
        "source_shape": [int(value) for value in matrix.speeds.shape],
        "total_observations": int(matrix.speeds.size),
        "positive_valid_observations": int(positive.sum()),
        "positive_valid_rate": float(positive.mean()),
        "zero_observations": int(quality.is_zero.sum()),
        "zero_rate": float(quality.is_zero.mean()),
        "systemwide_zero_timestamps": int(quality.is_systemwide_zero.sum()),
        "majority_zero_timestamps": int(quality.is_majority_zero.sum()),
        "long_zero_run_threshold_steps": int(long_zero_run_steps),
        "long_zero_run_threshold_minutes": int(long_zero_run_steps * 5),
        "observations_in_long_zero_runs": int(quality.is_long_sensor_zero_run.sum()),
        "large_jump_threshold_mph": float(large_jump_mph),
        "large_jump_observations": int(quality.is_large_jump.sum()),
        "large_jumps_adjacent_to_zero": int(quality.is_large_jump_from_zero.sum()),
        "large_jumps_between_positive_values": int(
            (quality.is_large_jump & ~quality.is_large_jump_from_zero).sum()
        ),
        "cleaning_rule": "speed_valid keeps finite values > 0; raw values remain unchanged",
        "imputation_applied": False,
        "parquet": parquet_info,
    }
