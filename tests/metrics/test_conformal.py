from __future__ import annotations

import numpy as np

from oral_virus_gpt.metrics.conformal import (
    conditional_coverage,
    marginal_coverage,
    mean_set_size,
)


def test_marginal_coverage_full_when_set_contains_label() -> None:
    sets = [[0, 1], [1, 2], [0, 2]]
    labels = np.array([1, 1, 2])
    assert marginal_coverage(sets, labels) == 1.0


def test_marginal_coverage_zero_when_no_overlap() -> None:
    sets = [[0], [1], [2]]
    labels = np.array([2, 0, 1])
    assert marginal_coverage(sets, labels) == 0.0


def test_conditional_coverage_per_class() -> None:
    sets = [[0], [0, 1], [1], [1, 2]]
    labels = np.array([0, 0, 1, 2])
    cov = conditional_coverage(sets, labels, num_classes=3)
    assert cov[0] == 1.0
    assert cov[1] == 1.0
    assert cov[2] == 1.0


def test_mean_set_size_arithmetic_mean() -> None:
    assert mean_set_size([[0, 1], [0], [0, 1, 2]]) == 2.0
