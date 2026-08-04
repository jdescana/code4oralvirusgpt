from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oral_virus_gpt.uq.risk_tier import RiskTier


@dataclass(slots=True)
class TierMetrics:
    coverage: float
    accuracy: float
    sensitivity: float
    npv: float


def tier_metrics(
    tiers: NDArray[Any],
    predictions: NDArray[np.integer[Any]],
    labels: NDArray[np.integer[Any]],
    tier: RiskTier,
) -> TierMetrics:
    mask = np.asarray([t == tier for t in tiers], dtype=bool)
    coverage = float(mask.mean()) if mask.size else 0.0
    if mask.sum() == 0:
        return TierMetrics(
            coverage=coverage, accuracy=float("nan"), sensitivity=float("nan"), npv=float("nan")
        )
    p = predictions[mask]
    y = labels[mask]
    correct = (p == y).astype(np.float64)
    acc = float(correct.mean())
    sens = acc
    pos_pred = (p != 0).astype(bool)
    npv = float("nan") if pos_pred.sum() == 0 else acc
    return TierMetrics(coverage=coverage, accuracy=acc, sensitivity=sens, npv=npv)


def nnr(referrals: int, errors_avoided: int) -> float:
    if referrals < 0:
        raise ValueError("referrals must be non-negative")
    if errors_avoided <= 0:
        return float("inf")
    return referrals / errors_avoided
