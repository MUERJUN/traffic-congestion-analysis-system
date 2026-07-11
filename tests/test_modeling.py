from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from traffic_congestion.modeling import (
    choose_validation_threshold,
    evaluate_predictions,
)


class ModelingTests(unittest.TestCase):
    def test_validation_threshold_prefers_f1_then_recall(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1, 0])
        probabilities = np.array([0.05, 0.20, 0.60, 0.70, 0.95, 0.80])
        result = choose_validation_threshold(
            y_true,
            probabilities,
            start=0.1,
            stop=0.9,
            step=0.1,
        )
        self.assertGreaterEqual(result["chosen"]["f1"], 0.8)
        self.assertIn("recall", result["chosen"])
        self.assertEqual(len(result["grid"]), 9)

    def test_evaluation_contains_required_metrics_and_confusion_matrix(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.8, 0.9])
        result = evaluate_predictions(y_true, probabilities, threshold=0.5)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            self.assertIn(key, result)
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 2]])


if __name__ == "__main__":
    unittest.main()

