from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class StratifiedSplit:
    train_indices: NDArray[np.int64]
    val_indices: NDArray[np.int64]
    test_indices: NDArray[np.int64]


def patient_stratified_split(
    patient_ids: Sequence[str | int],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> StratifiedSplit:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1.0e-6:
        raise ValueError("ratios must sum to 1.0")
    if not patient_ids:
        return StratifiedSplit(
            train_indices=np.zeros(0, dtype=np.int64),
            val_indices=np.zeros(0, dtype=np.int64),
            test_indices=np.zeros(0, dtype=np.int64),
        )
    pid_array = np.asarray(list(patient_ids))
    unique_ids = np.unique(pid_array)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_ids)
    n = len(unique_ids)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    train_ids = set(unique_ids[:n_train].tolist())
    val_ids = set(unique_ids[n_train : n_train + n_val].tolist())
    train_idx, val_idx, test_idx = [], [], []
    for idx, pid in enumerate(patient_ids):
        if pid in train_ids:
            train_idx.append(idx)
        elif pid in val_ids:
            val_idx.append(idx)
        else:
            test_idx.append(idx)
    return StratifiedSplit(
        train_indices=np.array(train_idx, dtype=np.int64),
        val_indices=np.array(val_idx, dtype=np.int64),
        test_indices=np.array(test_idx, dtype=np.int64),
    )
