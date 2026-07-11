"""Small, testable data-access helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from traffic_congestion.modeling import (
    MODEL_FEATURE_COLUMNS,
    local_counterfactual_explanation,
)


ROOT = Path(__file__).resolve().parents[1]

# 面向展示的字段说明：保留原始字段名，便于和数据字典、报告互相核对。
FEATURE_LABELS = {
    "hour": "小时（hour）",
    "day_of_week": "星期（day_of_week）",
    "is_weekend": "是否周末（is_weekend）",
    "hour_sin": "小时周期正弦（hour_sin）",
    "hour_cos": "小时周期余弦（hour_cos）",
    "day_of_week_sin": "星期周期正弦（day_of_week_sin）",
    "day_of_week_cos": "星期周期余弦（day_of_week_cos）",
    "current_speed_mph": "当前速度（mph）",
    "free_flow_speed_train_mph": "训练期自由流速度（mph）",
    "current_speed_ratio": "当前速度/自由流速度（current_speed_ratio）",
    "lag_speed_ratio_5m": "5分钟前速度比（lag_speed_ratio_5m）",
    "lag_speed_ratio_10m": "10分钟前速度比（lag_speed_ratio_10m）",
    "lag_speed_ratio_15m": "15分钟前速度比（lag_speed_ratio_15m）",
    "lag_speed_ratio_30m": "30分钟前速度比（lag_speed_ratio_30m）",
    "lag_speed_ratio_60m": "60分钟前速度比（lag_speed_ratio_60m）",
    "rolling_speed_ratio_mean_15m": "15分钟平均速度比",
    "rolling_speed_ratio_min_15m": "15分钟最低速度比",
    "rolling_speed_ratio_std_15m": "15分钟速度比波动",
    "rolling_speed_ratio_mean_30m": "30分钟平均速度比",
    "rolling_speed_ratio_min_30m": "30分钟最低速度比",
    "rolling_speed_ratio_std_30m": "30分钟速度比波动",
    "rolling_speed_ratio_mean_60m": "60分钟平均速度比",
    "rolling_speed_ratio_min_60m": "60分钟最低速度比",
    "rolling_speed_ratio_std_60m": "60分钟速度比波动",
    "trend_speed_mph_per_min_10m": "近10分钟速度趋势（mph/分钟）",
    "trend_speed_mph_per_min_15m": "近15分钟速度趋势（mph/分钟）",
    "trend_speed_mph_per_min_30m": "近30分钟速度趋势（mph/分钟）",
    "speed_change_5m": "5分钟速度变化（mph）",
    "consecutive_decline_steps": "连续降速步数",
    "historical_period_median_speed_mph_train": "历史同期速度中位数（mph）",
    "historical_period_congestion_rate_train": "历史同期拥堵率",
    "deviation_from_historical_median_mph": "相对历史同期速度偏差（mph）",
    "sensor_id": "道路传感器编号（sensor_id）",
}


def feature_label(name: str) -> str:
    """Return a user-facing Chinese label while retaining the source field."""
    return FEATURE_LABELS.get(name, name)


def localize_feature_column(frame: pd.DataFrame, column: str = "feature") -> pd.DataFrame:
    result = frame.copy()
    if column in result.columns:
        result[column] = result[column].map(feature_label)
    return result


@st.cache_data(show_spinner=False)
def load_csv(relative_path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative_path)


@st.cache_data(show_spinner=False)
def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_image_bytes(relative_path: str) -> bytes:
    return (ROOT / relative_path).read_bytes()


@st.cache_resource(show_spinner=False)
def load_model(relative_path: str) -> Any:
    return joblib.load(ROOT / relative_path)


@st.cache_data(show_spinner=False)
def load_sensor_history(sensor_id: str) -> pd.DataFrame:
    path = ROOT / "data/processed/model_dataset/test_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path, filters=[("sensor_id", "==", str(sensor_id))])
    if frame.empty:
        return frame
    return frame.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_congestion_heatmap() -> pd.DataFrame:
    """Build an hour-by-sensor congestion-rate matrix from the held-out replay set."""
    path = ROOT / "data/processed/model_dataset/test_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(
        path,
        columns=["sensor_id", "hour", "target_congestion_30m"],
    )
    grouped = (
        frame.groupby(["sensor_id", "hour"], as_index=False)["target_congestion_30m"]
        .mean()
        .rename(columns={"target_congestion_30m": "congestion_rate"})
    )
    return grouped.pivot(index="sensor_id", columns="hour", values="congestion_rate").fillna(0)


@st.cache_data(show_spinner=False)
def load_train_reference() -> pd.DataFrame:
    path = ROOT / "data/processed/model_dataset/train_features.parquet"
    if not path.exists():
        return pd.DataFrame()
    columns = MODEL_FEATURE_COLUMNS
    # A deterministic early row-group sample is sufficient for reference medians
    # and keeps page startup bounded; it is never used to retrain the model.
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read_row_group(0, columns=columns)
    parquet_file.close()
    return table.to_pandas()


def risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "高风险"
    if probability >= 0.40:
        return "中风险"
    return "低风险"


def predict_history_row(model: Any, row: pd.DataFrame) -> float:
    return float(model.predict_proba(row[MODEL_FEATURE_COLUMNS])[:, 1][0])


def explain_history_row(model: Any, row: pd.DataFrame) -> dict[str, Any]:
    reference = load_train_reference()
    if reference.empty:
        return {"top_factors": [], "method": "训练参考数据不可用"}
    return local_counterfactual_explanation(model, row, reference)
