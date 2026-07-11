"""CLI for M2 labels, historical features, and isolated time splits."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_pipeline import file_sha256, load_metr_la_h5
from .model_dataset import (
    FEATURE_COLUMNS,
    create_time_segments,
    fit_training_statistics,
    validate_model_parquets,
    validate_segment_windows,
    write_model_datasets,
    write_training_statistics,
)


LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(root / config_path)
    source_path = root / config["source_h5"]
    output_dir = root / config["output_dir"]
    report_dir = root / config["report_dir"]
    source_digest = file_sha256(source_path)
    if source_digest != str(config["expected_source_sha256"]).lower():
        raise ValueError("Source HDF5 SHA-256 does not match the approved baseline")

    LOGGER.info("Loading audited METR-LA source")
    matrix = load_metr_la_h5(source_path)
    segments = create_time_segments(
        len(matrix.timestamps),
        train_ratio=float(config["train_ratio"]),
        validation_ratio=float(config["validation_ratio"]),
        test_ratio=float(config["test_ratio"]),
    )
    history_steps = int(config["history_steps"])
    future_steps = int(config["future_steps"])
    validate_segment_windows(
        segments,
        history_steps=history_steps,
        future_steps=future_steps,
    )

    LOGGER.info("Fitting free-flow and historical-period statistics on training time only")
    statistics = fit_training_statistics(
        matrix,
        segments[0],
        free_flow_quantile=float(config["free_flow_quantile"]),
        congestion_threshold=float(config["congestion_threshold"]),
    )
    statistics_info = write_training_statistics(
        matrix,
        statistics,
        output_dir,
        report_dir,
    )

    LOGGER.info("Writing isolated train/validation/test feature datasets")
    split_outputs = write_model_datasets(
        matrix,
        segments,
        statistics,
        output_dir,
        history_steps=history_steps,
        future_steps=future_steps,
        congestion_threshold=float(config["congestion_threshold"]),
        sensitivity_thresholds=[float(value) for value in config["sensitivity_thresholds"]],
        minimum_congested_future_steps=int(config["minimum_congested_future_steps"]),
    )
    LOGGER.info("Validating stored keys, null counts, target domain, and schemas")
    dataset_checks = validate_model_parquets(
        split_outputs,
        expected_sensor_count=len(matrix.sensor_ids),
    )
    if not all(
        checks["contains_both_target_classes"] for checks in dataset_checks.values()
    ):
        raise RuntimeError("A production split does not contain both target classes")

    segment_report = []
    for segment in segments:
        segment_report.append(
            {
                "name": segment.name,
                "observation_start_index": segment.start_index,
                "observation_end_index": segment.end_index,
                "observation_count": segment.observation_count,
                "observation_start_timestamp": str(matrix.timestamps[segment.start_index]),
                "observation_end_timestamp": str(matrix.timestamps[segment.end_index]),
                "sample_start_index_after_history_purge": segment.sample_start_index(history_steps),
                "sample_end_index_before_future_purge": segment.sample_end_index(future_steps),
                "sample_start_timestamp_after_history_purge": str(
                    matrix.timestamps[segment.sample_start_index(history_steps)]
                ),
                "sample_end_timestamp_before_future_purge": str(
                    matrix.timestamps[segment.sample_end_index(future_steps)]
                ),
            }
        )

    split_time_ranges_do_not_overlap = (
        pd.Timestamp(split_outputs["train"]["last_timestamp"])
        < pd.Timestamp(split_outputs["validation"]["first_timestamp"])
        < pd.Timestamp(split_outputs["validation"]["last_timestamp"])
        < pd.Timestamp(split_outputs["test"]["first_timestamp"])
    )
    leakage_checks = {
        "status": "PASS",
        "random_split_used": False,
        "training_statistics_observation_end": str(matrix.timestamps[segments[0].end_index]),
        "validation_observation_start": str(matrix.timestamps[segments[1].start_index]),
        "test_observation_start": str(matrix.timestamps[segments[2].start_index]),
        "feature_source_offsets": f"-{history_steps} through 0 only",
        "target_source_offsets": f"+1 through +{future_steps} only",
        "history_and_target_windows_confined_to_each_split": True,
        "free_flow_fitted_on_training_only": True,
        "historical_period_statistics_fitted_on_training_only": True,
        "split_sample_time_ranges_do_not_overlap": bool(split_time_ranges_do_not_overlap),
        "stored_dataset_checks": dataset_checks,
        "imputation_using_future_values": False,
        "test_used_for_tuning": False,
    }
    if not split_time_ranges_do_not_overlap:
        raise RuntimeError("Stored split sample timestamps overlap")
    write_json(report_dir / "leakage_checks.json", leakage_checks)

    summary = {
        "stage": "M2 - labels, historical features, and time split",
        "model_training_performed": False,
        "source_sha256": source_digest,
        "config": config,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_feature_count": len(FEATURE_COLUMNS),
        "categorical_feature_columns": ["sensor_id"],
        "model_feature_count": len(FEATURE_COLUMNS) + 1,
        "metadata_columns": ["timestamp", "sensor_id", "split"],
        "target_column": "target_congestion_30m",
        "segments": segment_report,
        "split_outputs": split_outputs,
        "stored_dataset_checks": dataset_checks,
        "training_statistics": statistics_info,
        "free_flow_summary": {
            "minimum_mph": float(statistics.free_flow_speed.min()),
            "median_mph": float(np.median(statistics.free_flow_speed)),
            "maximum_mph": float(statistics.free_flow_speed.max()),
        },
        "leakage_check_path": str(report_dir / "leakage_checks.json"),
    }
    write_json(report_dir / "m2_summary.json", summary)
    pd.DataFrame(
        [
            {
                "split": split_name,
                "usable_samples": output["usable_samples"],
                "positive_labels_50": output["positive_labels"]["0.5"],
                "positive_rate_50": output["positive_rates"]["0.5"],
                "positive_labels_60": output["positive_labels"]["0.6"],
                "positive_rate_60": output["positive_rates"]["0.6"],
                "positive_labels_70": output["positive_labels"]["0.7"],
                "positive_rate_70": output["positive_rates"]["0.7"],
                "first_timestamp": output["first_timestamp"],
                "last_timestamp": output["last_timestamp"],
            }
            for split_name, output in split_outputs.items()
        ]
    ).to_csv(report_dir / "split_label_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(segment_report).to_csv(
        report_dir / "time_split_boundaries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    LOGGER.info("M2 completed without model training")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe M2 model datasets")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("config/m2.json"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run(args.root, args.config)


if __name__ == "__main__":
    main()
