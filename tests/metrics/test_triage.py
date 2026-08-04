from __future__ import annotations

import math

import numpy as np

from oral_virus_gpt.metrics.triage import nnr, tier_metrics
from oral_virus_gpt.uq.risk_tier import RiskTier


def test_nnr_formula() -> None:
    assert nnr(referrals=10, errors_avoided=5) == 2.0


def test_nnr_infinite_when_no_errors_avoided() -> None:
    assert math.isinf(nnr(referrals=10, errors_avoided=0))


def test_tier_metrics_simple() -> None:
    tiers = np.array([RiskTier.LOW, RiskTier.LOW, RiskTier.HIGH, RiskTier.HIGH], dtype=object)
    preds = np.array([0, 0, 1, 0])
    labels = np.array([0, 0, 1, 2])
    m = tier_metrics(tiers, preds, labels, RiskTier.LOW)
    assert m.coverage == 0.5
    assert m.accuracy == 1.0
    high = tier_metrics(tiers, preds, labels, RiskTier.HIGH)
    assert high.coverage == 0.5
    assert 0.0 <= high.accuracy <= 1.0
