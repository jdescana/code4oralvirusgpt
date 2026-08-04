from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score


def cohens_kappa(
    rater_a: NDArray[np.integer[Any]],
    rater_b: NDArray[np.integer[Any]],
) -> float:
    a = np.asarray(rater_a, dtype=np.int64).ravel()
    b = np.asarray(rater_b, dtype=np.int64).ravel()
    if a.shape != b.shape:
        raise ValueError("raters disagree on shape")
    if a.size == 0:
        return float("nan")
    return float(cohen_kappa_score(a, b))
