from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    samples: NDArray[np.floating[Any]]


def bootstrap_ci(
    statistic: Callable[[NDArray[np.integer[Any]], NDArray[np.integer[Any]]], float],
    predictions: NDArray[np.integer[Any]],
    labels: NDArray[np.integer[Any]],
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    p = np.asarray(predictions)
    y = np.asarray(labels)
    if p.shape[0] != y.shape[0]:
        raise ValueError("predictions and labels must share batch dimension")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    n = p.shape[0]
    samples = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic(p[idx], y[idx])
    estimate = statistic(p, y)
    lo = float(np.quantile(samples, (1.0 - confidence) / 2.0))
    hi = float(np.quantile(samples, 1.0 - (1.0 - confidence) / 2.0))
    return BootstrapResult(estimate=float(estimate), lower=lo, upper=hi, samples=samples)
