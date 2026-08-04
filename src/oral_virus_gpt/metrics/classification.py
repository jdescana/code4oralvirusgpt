from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, roc_auc_score


def _to_int_array(x: NDArray[np.integer[Any]] | list[int]) -> NDArray[np.int64]:
    return np.asarray(x, dtype=np.int64).ravel()


def accuracy(predictions: NDArray[np.integer[Any]], labels: NDArray[np.integer[Any]]) -> float:
    p = _to_int_array(predictions)
    y = _to_int_array(labels)
    if p.shape != y.shape:
        raise ValueError("predictions and labels disagree on shape")
    if p.size == 0:
        return float("nan")
    return float((p == y).mean())


def macro_f1(predictions: NDArray[np.integer[Any]], labels: NDArray[np.integer[Any]]) -> float:
    p = _to_int_array(predictions)
    y = _to_int_array(labels)
    if p.size == 0:
        return float("nan")
    return float(f1_score(y, p, average="macro", zero_division=0))


def sensitivity(
    predictions: NDArray[np.integer[Any]],
    labels: NDArray[np.integer[Any]],
    num_classes: int,
) -> NDArray[np.floating[Any]]:
    p = _to_int_array(predictions)
    y = _to_int_array(labels)
    out = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = y == c
        if mask.sum() == 0:
            out[c] = float("nan")
        else:
            out[c] = float((p[mask] == c).mean())
    return out


def specificity(
    predictions: NDArray[np.integer[Any]],
    labels: NDArray[np.integer[Any]],
    num_classes: int,
) -> NDArray[np.floating[Any]]:
    p = _to_int_array(predictions)
    y = _to_int_array(labels)
    out = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        neg_mask = y != c
        if neg_mask.sum() == 0:
            out[c] = float("nan")
        else:
            out[c] = float((p[neg_mask] != c).mean())
    return out


def auc(probs: NDArray[np.floating[Any]], labels: NDArray[np.integer[Any]]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = _to_int_array(labels)
    if p.ndim != 2:
        raise ValueError("probs must be 2D [n, num_classes]")
    if p.shape[0] != y.shape[0]:
        raise ValueError("probs and labels disagree on batch")
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    try:
        return float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")
