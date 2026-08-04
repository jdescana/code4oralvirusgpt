from __future__ import annotations

import numpy as np

from oral_virus_gpt.metrics.classification import accuracy, auc, macro_f1, sensitivity, specificity


def test_accuracy_perfect_predictions() -> None:
    preds = np.array([0, 1, 2, 1, 0])
    labels = preds.copy()
    assert accuracy(preds, labels) == 1.0


def test_macro_f1_is_unit_for_correct() -> None:
    preds = np.array([0, 1, 1, 2, 2])
    labels = np.array([0, 1, 1, 2, 2])
    assert macro_f1(preds, labels) == 1.0


def test_sensitivity_specificity_per_class() -> None:
    preds = np.array([0, 1, 2, 0, 1, 2])
    labels = np.array([0, 1, 2, 0, 1, 2])
    sens = sensitivity(preds, labels, num_classes=3)
    spec = specificity(preds, labels, num_classes=3)
    assert np.allclose(sens, 1.0)
    assert np.allclose(spec, 1.0)


def test_auc_binary() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])
    labels = np.array([0, 1, 1, 0])
    value = auc(probs, labels)
    assert 0.5 <= value <= 1.0
