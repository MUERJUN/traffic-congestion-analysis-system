"""CLI for M3 model comparison, evaluation, and feature explanation."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from .data_pipeline import file_sha256
from .model_dataset import FEATURE_COLUMNS
from .modeling import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_preprocessor,
    choose_validation_threshold,
    deterministic_train_sample,
    evaluate_predictions,
    extract_importance,
    load_split,
    local_counterfactual_explanation,
    make_models,
    plot_importance,
    plot_model_metrics,
    plot_roc_curves,
    json_safe,
)


LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def run(root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(root / config_path)
    dataset_dir = root / config["model_dataset_dir"]
    report_dir = root / config["report_dir"]
    artifact_dir = root / config["artifact_dir"]
    figure_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading full validation and test splits")
    train = load_split(dataset_dir / "train_features.parquet")
    validation = load_split(dataset_dir / "validation_features.parquet")
    test = load_split(dataset_dir / "test_features.parquet")
    train_sample = deterministic_train_sample(
        train,
        int(config["train_sample_max_rows"]),
        int(config["random_state"]),
    )
    positive = int(train_sample.y.sum())
    negative = int(len(train_sample.y) - positive)
    positive_weight = negative / positive if positive else 1.0

    LOGGER.info(
        "Using deterministic train sample: %s rows (full train: %s)",
        len(train_sample.frame),
        len(train.frame),
    )
    models = make_models(config, positive_weight)
    model_results: list[dict[str, Any]] = []
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    importance_tables: dict[str, pd.DataFrame] = {}
    fitted_pipelines: dict[str, Pipeline] = {}
    test_probabilities: dict[str, np.ndarray] = {}

    for model_name, estimator in models.items():
        LOGGER.info("Training %s", model_name)
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(train_sample.x, train_sample.y)
        validation_probability = pipeline.predict_proba(validation.x)[:, 1]
        threshold_info = choose_validation_threshold(
            validation.y,
            validation_probability,
            start=float(config["validation_threshold_grid_start"]),
            stop=float(config["validation_threshold_grid_stop"]),
            step=float(config["validation_threshold_grid_step"]),
        )
        threshold = threshold_info["chosen"]["threshold"]
        validation_metrics = evaluate_predictions(
            validation.y,
            validation_probability,
            threshold,
        )
        test_probability = pipeline.predict_proba(test.x)[:, 1]
        test_metrics = evaluate_predictions(test.y, test_probability, threshold)
        fpr, tpr, _ = roc_curve(test.y, test_probability)
        roc_auc = float(roc_auc_score(test.y, test_probability))
        roc_curves[model_name] = (fpr, tpr, roc_auc)
        test_probabilities[model_name] = test_probability
        fitted_pipelines[model_name] = pipeline

        importance = extract_importance(
            pipeline.named_steps["model"],
            pipeline.named_steps["preprocessor"],
        )
        importance_tables[model_name] = importance
        importance_path = report_dir / f"feature_importance_{slug(model_name)}.csv"
        importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
        plot_importance(
            importance,
            figure_dir / f"feature_importance_{slug(model_name)}.png",
            f"{model_name} global feature importance",
        )

        artifact_path = artifact_dir / f"{slug(model_name)}.joblib"
        temporary_path = artifact_path.with_suffix(".joblib.tmp")
        joblib.dump(pipeline, temporary_path, compress=3)
        temporary_path.replace(artifact_path)
        model_results.append(
            {
                "model": model_name,
                "train_rows_used": int(len(train_sample.frame)),
                "train_rows_available": int(len(train.frame)),
                "validation": validation_metrics,
                "test": test_metrics,
                "validation_threshold": threshold_info["chosen"],
                "artifact_path": str(artifact_path),
                "artifact_sha256": file_sha256(artifact_path),
                "top_global_features": importance.head(10).to_dict(orient="records"),
            }
        )

    metrics_table = pd.DataFrame(
        [
            {
                "model": result["model"],
                **{
                    metric: result["test"][metric]
                    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
                },
                "threshold": result["test"]["threshold"],
                "validation_f1": result["validation"]["f1"],
                "validation_recall": result["validation"]["recall"],
                "validation_roc_auc": result["validation"]["roc_auc"],
            }
            for result in model_results
        ]
    )
    metrics_table.to_csv(report_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    plot_model_metrics(metrics_table, figure_dir / "model_comparison.png")
    plot_roc_curves(roc_curves, figure_dir / "test_roc_curves.png")

    selected = max(
        model_results,
        key=lambda result: (
            result["validation"]["f1"],
            result["validation"]["recall"],
            result["validation"]["roc_auc"],
        ),
    )
    selected_name = selected["model"]
    selected_pipeline = fitted_pipelines[selected_name]
    selected_probability = test_probabilities[selected_name]
    high_risk_index = int(np.argmax(selected_probability))
    local_sample = test.frame.iloc[[high_risk_index]].copy()
    local_explanation = local_counterfactual_explanation(
        selected_pipeline,
        local_sample,
        train_sample.frame,
    )
    local_explanation["actual_target"] = int(local_sample.iloc[0][TARGET_COLUMN])
    local_explanation["selected_model"] = selected_name
    local_explanation["risk_level"] = (
        "high" if local_explanation["base_probability"] >= 0.70
        else "medium" if local_explanation["base_probability"] >= 0.40
        else "low"
    )
    write_json(report_dir / "local_explanation.json", local_explanation)
    write_json(report_dir / "validation_thresholds.json", {
        result["model"]: result["validation_threshold"] for result in model_results
    })

    summary = {
        "stage": "M3 - model training, comparison, and explanation",
        "model_training_performed": True,
        "test_used_for_tuning": False,
        "train_rows_available": int(len(train.frame)),
        "train_rows_used": int(len(train_sample.frame)),
        "validation_rows": int(len(validation.frame)),
        "test_rows": int(len(test.frame)),
        "train_sampling": {
            "max_rows": int(config["train_sample_max_rows"]),
            "random_state": int(config["random_state"]),
            "positive_rows": positive,
            "negative_rows": negative,
            "xgboost_scale_pos_weight": positive_weight,
        },
        "model_results": model_results,
        "selected_model": selected_name,
        "selection_rule": "validation congestion-class F1, then Recall, then ROC-AUC",
        "local_explanation_path": str(report_dir / "local_explanation.json"),
        "metrics_path": str(report_dir / "model_comparison.csv"),
    }
    write_json(report_dir / "m3_summary.json", summary)
    LOGGER.info("Selected model: %s", selected_name)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare M3 traffic risk models")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("config/m3.json"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run(args.root, args.config)


if __name__ == "__main__":
    main()

