from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def marginal_coverage(sets: list[list[int]], labels: NDArray[np.integer[Any]]) -> float:
    y = np.asarray(labels, dtype=np.int64).ravel()
    if len(sets) != y.shape[0]:
        raise ValueError("sets and labels disagree on length")
    hits = sum(1 for s, c in zip(sets, y, strict=True) if int(c) in s)
    return hits / len(sets) if sets else 0.0


def conditional_coverage(
    sets: list[list[int]],
    labels: NDArray[np.integer[Any]],
    num_classes: int,
) -> NDArray[np.floating[Any]]:
    y = np.asarray(labels, dtype=np.int64).ravel()
    out = np.zeros(num_classes, dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for s, c in zip(sets, y, strict=True):
        counts[c] += 1
        if int(c) in s:
            out[c] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = np.where(counts > 0, out / np.maximum(counts, 1), np.nan)
    return cov


def mean_set_size(sets: list[list[int]]) -> float:
    if not sets:
        return float("nan")
    return float(np.mean([len(s) for s in sets]))
