"""Build compact runtime data required by the Streamlit dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic_congestion.modeling import MODEL_FEATURE_COLUMNS  # noqa: E402


TEST_SOURCE = ROOT / "data/processed/model_dataset/test_features.parquet"
TRAIN_SOURCE = ROOT / "data/processed/model_dataset/train_features.parquet"
OUT = ROOT / "data/dashboard"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    test_file = pq.ParquetFile(TEST_SOURCE)
    replay_parts = []
    heatmap_parts = []
    for index in range(test_file.metadata.num_row_groups):
        frame = test_file.read_row_group(index).to_pandas()
        replay_parts.append(frame.tail(288))
        frame["date"] = pd.to_datetime(frame["timestamp"]).dt.strftime("%Y-%m-%d")
        heatmap_parts.append(
            frame.groupby(["date", "sensor_id", "hour"], as_index=False)["target_congestion_30m"]
            .mean()
            .rename(columns={"target_congestion_30m": "congestion_rate"})
        )
    test_file.close()

    replay = pd.concat(replay_parts, ignore_index=True).sort_values(["sensor_id", "timestamp"])
    replay.to_parquet(OUT / "test_replay.parquet", index=False, compression="zstd")
    heatmap = pd.concat(heatmap_parts, ignore_index=True)
    heatmap.to_parquet(OUT / "congestion_heatmap.parquet", index=False, compression="zstd")

    train_file = pq.ParquetFile(TRAIN_SOURCE)
    reference = train_file.read_row_group(0, columns=MODEL_FEATURE_COLUMNS).to_pandas()
    train_file.close()
    reference.to_parquet(OUT / "train_reference.parquet", index=False, compression="zstd")

    for path in sorted(OUT.glob("*.parquet")):
        print(f"{path.name}: {path.stat().st_size / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
