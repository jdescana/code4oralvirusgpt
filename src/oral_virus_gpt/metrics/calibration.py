from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class ReliabilityDiagram:
    bin_edges: NDArray[np.floating[Any]]
    bin_confidences: NDArray[np.floating[Any]]
    bin_accuracies: NDArray[np.floating[Any]]
    bin_counts: NDArray[np.int64]


def ece(
    probs: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    num_bins: int = 15,
) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).ravel()
    if p.ndim != 2:
        raise ValueError("probs must be 2D")
    if p.shape[0] != y.shape[0]:
        raise ValueError("probs and labels disagree on batch")
    if p.size == 0:
        return float("nan")
    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    total = float(p.shape[0])
    e = 0.0
    for lo, hi in pairwise(bin_edges):
        in_bin = (
            (confidences > lo) & (confidences <= hi)
            if lo > 0
            else (confidences >= lo) & (confidences <= hi)
        )
        n = float(in_bin.sum())
        if n == 0:
            continue
        avg_conf = float(confidences[in_bin].mean())
        avg_acc = float(correct[in_bin].mean())
        e += (n / total) * abs(avg_acc - avg_conf)
    return e


def brier(probs: NDArray[np.floating[Any]], labels: NDArray[np.integer[Any]]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).ravel()
    if p.ndim != 2:
        raise ValueError("probs must be 2D")
    one_hot = np.zeros_like(p)
    one_hot[np.arange(p.shape[0]), y] = 1.0
    return float(((p - one_hot) ** 2).sum(axis=1).mean())


def reliability_diagram(
    probs: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    num_bins: int = 15,
) -> ReliabilityDiagram:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).ravel()
    confidences = p.max(axis=1)
    predictions = p.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    confs = np.zeros(num_bins, dtype=np.float64)
    accs = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    for i, (lo, hi) in enumerate(pairwise(bin_edges)):
        in_bin = (
            (confidences > lo) & (confidences <= hi)
            if lo > 0
            else (confidences >= lo) & (confidences <= hi)
        )
        counts[i] = int(in_bin.sum())
        if counts[i] > 0:
            confs[i] = float(confidences[in_bin].mean())
            accs[i] = float(correct[in_bin].mean())
    return ReliabilityDiagram(
        bin_edges=bin_edges, bin_confidences=confs, bin_accuracies=accs, bin_counts=counts
    )
