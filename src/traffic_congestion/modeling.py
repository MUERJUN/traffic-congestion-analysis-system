"""M3 model training, validation thresholding, evaluation, and explanations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data_pipeline import file_sha256
from .model_dataset import FEATURE_COLUMNS


MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + ["sensor_id"]
TARGET_COLUMN = "target_congestion_30m"


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    x: pd.DataFrame
    y: np.ndarray


def load_split(path: Path) -> DatasetBundle:
    columns = ["timestamp"] + MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]
    frame = pd.read_parquet(path, columns=columns)
    x = frame[MODEL_FEATURE_COLUMNS].copy()
    y = frame[TARGET_COLUMN].to_numpy(dtype=np.int8)
    return DatasetBundle(frame=frame, x=x, y=y)


def deterministic_train_sample(
    train: DatasetBundle,
    max_rows: int,
    random_state: int,
) -> DatasetBundle:
    if max_rows <= 0 or len(train.frame) <= max_rows:
        return train
    sampled = train.frame.sample(n=max_rows, random_state=random_state, replace=False)
    sampled = sampled.sort_index()
    return DatasetBundle(
        frame=sampled,
        x=sampled[MODEL_FEATURE_COLUMNS].copy(),
        y=sampled[TARGET_COLUMN].to_numpy(dtype=np.int8),
    )


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),
        ]
    )
    categorical = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, FEATURE_COLUMNS),
            ("sensor", categorical, ["sensor_id"]),
        ],
        sparse_threshold=0.3,
        remainder="drop",
    )


def make_models(config: dict[str, Any], positive_weight: float) -> dict[str, Any]:
    random_state = int(config["random_state"])
    return {
        "Logistic Regression": LogisticRegression(
            solver="saga",
            max_iter=int(config["logistic_max_iter"]),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=4,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=int(config["random_forest_estimators"]),
            max_depth=int(config["random_forest_max_depth"]),
            min_samples_leaf=int(config["random_forest_min_samples_leaf"]),
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=int(config["xgboost_estimators"]),
            max_depth=int(config["xgboost_max_depth"]),
            learning_rate=float(config["xgboost_learning_rate"]),
            subsample=float(config["xgboost_subsample"]),
            colsample_bytree=float(config["xgboost_colsample_bytree"]),
            min_child_weight=int(config["xgboost_min_child_weight"]),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=int(config["xgboost_n_jobs"]),
            random_state=random_state,
            scale_pos_weight=positive_weight,
        ),
    }


def choose_validation_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    start: float = 0.10,
    stop: float = 0.90,
    step: float = 0.01,
) -> dict[str, Any]:
    thresholds = np.arange(start, stop + step / 2, step)
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        candidates.append(
            {
                "threshold": float(threshold),
                "f1": float(f1_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
            }
        )
    chosen = max(candidates, key=lambda item: (item["f1"], item["recall"], item["precision"]))
    return {"chosen": chosen, "grid": candidates}


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "positive_rate": float(np.mean(predictions)),
        "actual_positive_rate": float(np.mean(y_true)),
    }


def feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    return preprocessor.get_feature_names_out()


def aggregate_feature_importance(
    names: np.ndarray,
    importance: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, value in zip(names, importance, strict=True):
        raw_name = str(name)
        if raw_name.startswith("sensor__sensor_id_"):
            display_name = "sensor_id (one-hot)"
        elif raw_name.startswith("numeric__"):
            display_name = raw_name.removeprefix("numeric__")
        else:
            display_name = raw_name
        rows.append(
            {
                "feature": display_name,
                "model_feature": raw_name,
                "importance": float(value),
            }
        )
    table = pd.DataFrame(rows)
    return (
        table.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def extract_importance(model: Any, preprocessor: ColumnTransformer) -> pd.DataFrame:
    names = feature_names(preprocessor)
    if isinstance(model, LogisticRegression):
        importance = np.abs(model.coef_[0])
    else:
        importance = np.asarray(model.feature_importances_)
    return aggregate_feature_importance(names, importance)


def local_counterfactual_explanation(
    pipeline: Pipeline,
    sample: pd.DataFrame,
    train_reference: pd.DataFrame,
) -> dict[str, Any]:
    """Explain one prediction by replacing each source feature with a train reference."""

    base_probability = float(pipeline.predict_proba(sample)[0, 1])
    reference = {}
    for column in FEATURE_COLUMNS:
        reference[column] = float(train_reference[column].median())
    reference["sensor_id"] = str(train_reference["sensor_id"].mode().iloc[0])

    rows: list[dict[str, Any]] = []
    for column in MODEL_FEATURE_COLUMNS:
        counterfactual = sample.copy()
        original = counterfactual.iloc[0][column]
        counterfactual.loc[counterfactual.index[0], column] = reference[column]
        cf_probability = float(pipeline.predict_proba(counterfactual)[0, 1])
        rows.append(
            {
                "feature": column,
                "observed_value": str(original),
                "reference_value": str(reference[column]),
                "base_probability": base_probability,
                "counterfactual_probability": cf_probability,
                "risk_support_delta": base_probability - cf_probability,
            }
        )
    rows.sort(key=lambda row: abs(row["risk_support_delta"]), reverse=True)
    return {
        "base_probability": base_probability,
        "observed_timestamp": str(sample.iloc[0].get("timestamp", "not included")),
        "observed_sensor_id": str(sample.iloc[0]["sensor_id"]),
        "top_factors": rows[:10],
        "method": "one-feature-at-a-time train-reference counterfactual; not causal",
    }


def plot_roc_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curves")
    ax.grid(alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_model_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    plot_data = metrics.set_index("model")[metric_columns]
    ax = plot_data.plot(kind="bar", figsize=(10, 6), ylim=(0, 1), rot=0)
    ax.set(title="Test model comparison", ylabel="Score", xlabel="Model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig = ax.get_figure()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, output_path: Path, title: str) -> None:
    top = importance.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set(title=title, xlabel="Importance", ylabel="Feature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value
