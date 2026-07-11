"""Command-line entry point for Phase 3 cleaning and formal EDA."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .data_pipeline import (
    align_sensor_locations,
    build_quality_masks,
    file_sha256,
    load_metr_la_h5,
    quality_summary,
    validate_traffic_matrix,
    write_clean_long_parquet,
)
from .eda import run_eda


LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "source_h5",
        "locations_csv",
        "expected_source_sha256",
        "clean_long_parquet",
        "cleaning_report_dir",
        "eda_table_dir",
        "eda_figure_dir",
        "majority_zero_threshold",
        "long_zero_run_steps",
        "large_jump_mph",
        "parquet_chunk_time_steps",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Phase 3 config is missing keys: {missing}")
    return config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(root / config_path)
    source_path = root / config["source_h5"]
    locations_path = root / config["locations_csv"]
    parquet_path = root / config["clean_long_parquet"]
    cleaning_dir = root / config["cleaning_report_dir"]
    eda_table_dir = root / config["eda_table_dir"]
    eda_figure_dir = root / config["eda_figure_dir"]

    LOGGER.info("Verifying source file: %s", source_path)
    source_digest = file_sha256(source_path)
    expected_digest = str(config["expected_source_sha256"]).lower()
    if source_digest != expected_digest:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {expected_digest}, got {source_digest}"
        )

    LOGGER.info("Loading and validating METR-LA")
    matrix = load_metr_la_h5(source_path)
    structure = validate_traffic_matrix(matrix)
    locations = align_sensor_locations(matrix.sensor_ids, locations_path)
    if locations[["latitude", "longitude"]].isna().any(axis=None):
        raise ValueError("One or more core sensors do not have valid locations")

    LOGGER.info("Building non-destructive quality flags")
    quality = build_quality_masks(
        matrix,
        majority_threshold=float(config["majority_zero_threshold"]),
        long_zero_run_steps=int(config["long_zero_run_steps"]),
        large_jump_mph=float(config["large_jump_mph"]),
    )

    LOGGER.info("Writing EDA-ready long Parquet table")
    parquet_info = write_clean_long_parquet(
        matrix,
        quality,
        locations,
        parquet_path,
        chunk_time_steps=int(config["parquet_chunk_time_steps"]),
    )
    metadata = pq.ParquetFile(parquet_path).metadata
    if metadata.num_rows != matrix.speeds.size:
        raise RuntimeError(
            f"Parquet metadata row mismatch: {metadata.num_rows} != {matrix.speeds.size}"
        )

    cleaning = quality_summary(
        matrix,
        quality,
        source_sha256=source_digest,
        parquet_info=parquet_info,
        long_zero_run_steps=int(config["long_zero_run_steps"]),
        large_jump_mph=float(config["large_jump_mph"]),
    )
    cleaning["structure_validation"] = structure
    cleaning["location_match_count"] = int(
        locations[["latitude", "longitude"]].notna().all(axis=1).sum()
    )
    cleaning["config"] = config
    write_json(cleaning_dir / "cleaning_summary.json", cleaning)

    LOGGER.info("Running required EDA tables and figures")
    eda_summary = run_eda(matrix, quality, locations, eda_table_dir, eda_figure_dir)
    write_json(root / "reports/eda/eda_summary.json", eda_summary)

    manifest = {
        "phase": "Phase 3A - cleaning and formal EDA",
        "model_training_performed": False,
        "future_supervised_label_generated": False,
        "source_sha256": source_digest,
        "cleaning_summary": str(cleaning_dir / "cleaning_summary.json"),
        "eda_summary": str(root / "reports/eda/eda_summary.json"),
        "parquet": parquet_info,
    }
    write_json(root / "reports/phase3_manifest.json", manifest)
    LOGGER.info("Phase 3A completed without model training")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the audited METR-LA cleaning view and formal EDA outputs."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/phase3.json"),
        help="Config path relative to project root",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run(args.root, args.config)


if __name__ == "__main__":
    main()

