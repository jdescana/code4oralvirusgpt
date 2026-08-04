from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RiskTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True)
class RiskTierPolicy:
    low_entropy_threshold: float
    high_entropy_threshold: float
    medium_set_size: int = 2

    def assign(self, set_size: int, entropy: float) -> RiskTier:
        if set_size == 1 and entropy < self.low_entropy_threshold:
            return RiskTier.LOW
        if set_size <= self.medium_set_size and entropy < self.high_entropy_threshold:
            return RiskTier.MEDIUM
        return RiskTier.HIGH

    def assign_batch(
        self,
        set_sizes: NDArray[np.integer[Any]],
        entropies: NDArray[np.floating[Any]],
    ) -> NDArray[Any]:
        tiers = np.empty(set_sizes.shape[0], dtype=object)
        for i, (s, e) in enumerate(zip(set_sizes, entropies, strict=True)):
            tiers[i] = self.assign(int(s), float(e))
        return tiers


def calibrate_thresholds(
    set_sizes: NDArray[np.integer[Any]],
    entropies: NDArray[np.floating[Any]],
    correct: NDArray[np.integer[Any]],
    target_low_acc: float = 0.95,
    target_high_recall: float = 0.90,
    medium_set_size: int = 2,
) -> RiskTierPolicy:
    if not (set_sizes.shape == entropies.shape == correct.shape):
        raise ValueError("set_sizes, entropies, correct must share shape")
    quantiles = np.linspace(0.05, 0.95, 19)
    low_threshold = float(np.quantile(entropies[correct.astype(bool)], 0.50))
    high_threshold = float(np.quantile(entropies, 0.85))
    for q in quantiles:
        cand_low = float(np.quantile(entropies, q))
        mask_low = (set_sizes == 1) & (entropies < cand_low)
        if mask_low.sum() == 0:
            continue
        acc = float(correct[mask_low].mean())
        if acc >= target_low_acc:
            low_threshold = cand_low
            break
    error_entropies = entropies[~correct.astype(bool)]
    if error_entropies.size > 0:
        high_threshold = float(np.quantile(error_entropies, 1.0 - target_high_recall))
    if high_threshold < low_threshold:
        high_threshold = low_threshold
    return RiskTierPolicy(
        low_entropy_threshold=low_threshold,
        high_entropy_threshold=high_threshold,
        medium_set_size=medium_set_size,
    )
