from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def paired_bootstrap_pvalue(
    metric_a: NDArray[np.floating[Any]],
    metric_b: NDArray[np.floating[Any]],
    iterations: int = 1000,
    seed: int = 0,
) -> float:
    a = np.asarray(metric_a, dtype=np.float64)
    b = np.asarray(metric_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("metric arrays must share shape")
    if a.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    diffs = a - b
    observed = float(diffs.mean())
    n = diffs.shape[0]
    centred = diffs - observed
    extreme = 0
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        boot_mean = float(centred[idx].mean())
        if abs(boot_mean) >= abs(observed):
            extreme += 1
    return (1 + extreme) / (iterations + 1)


def holm_bonferroni(pvalues: Sequence[float]) -> NDArray[np.floating[Any]]:
    p = np.asarray(list(pvalues), dtype=np.float64)
    if p.size == 0:
        return p
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p[idx]
        running_max = max(running_max, min(adj, 1.0))
        adjusted[idx] = running_max
    return adjusted


def bh_fdr(pvalues: Sequence[float]) -> NDArray[np.floating[Any]]:
    p = np.asarray(list(pvalues), dtype=np.float64)
    if p.size == 0:
        return p
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        adj = m * p[idx] / (rank + 1)
        running_min = min(running_min, min(adj, 1.0))
        adjusted[idx] = running_min
    return adjusted
