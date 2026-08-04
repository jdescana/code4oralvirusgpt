from __future__ import annotations

import numpy as np

from oral_virus_gpt.metrics.calibration import brier, ece, reliability_diagram


def test_ece_zero_on_calibrated_distribution() -> None:
    n = 1000
    rng = np.random.default_rng(0)
    confidences = rng.uniform(0.5, 1.0, size=n)
    correct = rng.uniform(size=n) < confidences
    probs = np.zeros((n, 2))
    probs[:, 0] = confidences
    probs[:, 1] = 1.0 - confidences
    labels = np.where(correct, 0, 1)
    value = ece(probs, labels)
    assert value < 0.05


def test_brier_zero_on_perfect_predictions() -> None:
    probs = np.zeros((4, 3))
    probs[:, 0] = 1.0
    labels = np.zeros(4, dtype=np.int64)
    assert brier(probs, labels) == 0.0


def test_reliability_diagram_shape() -> None:
    probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
    labels = np.array([0, 1, 1])
    diag = reliability_diagram(probs, labels, num_bins=5)
    assert diag.bin_confidences.shape == (5,)
    assert diag.bin_accuracies.shape == (5,)
    assert diag.bin_counts.sum() == 3
