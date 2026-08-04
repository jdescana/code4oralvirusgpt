from __future__ import annotations

import numpy as np

from oral_virus_gpt.metrics.bootstrap import bootstrap_ci
from oral_virus_gpt.metrics.classification import accuracy


def test_bootstrap_ci_brackets_estimate() -> None:
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 2, size=n)
    preds = labels.copy()
    flip = rng.random(size=n) < 0.05
    preds = np.where(flip, 1 - preds, preds)
    res = bootstrap_ci(accuracy, preds, labels, iterations=200, confidence=0.95, seed=1)
    assert res.lower <= res.estimate <= res.upper
    assert res.samples.shape == (200,)


def test_bootstrap_ci_widens_with_higher_confidence() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200)
    preds = labels.copy()
    narrow = bootstrap_ci(accuracy, preds, labels, iterations=200, confidence=0.5, seed=2)
    wide = bootstrap_ci(accuracy, preds, labels, iterations=200, confidence=0.99, seed=2)
    assert (wide.upper - wide.lower) >= (narrow.upper - narrow.lower)
