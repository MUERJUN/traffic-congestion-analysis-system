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
