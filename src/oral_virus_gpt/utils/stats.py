from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def safe_mean(values: Sequence[float] | NDArray[np.floating[Any]]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values: Sequence[float] | NDArray[np.floating[Any]], ddof: int = 1) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size <= ddof:
        return 0.0
    return float(np.std(arr, ddof=ddof))


def coefficient_of_variation(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    mean = float(np.mean(arr))
    if mean == 0.0:
        return float("nan")
    return float(np.std(arr, ddof=1) / abs(mean))
