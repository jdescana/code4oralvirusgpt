from __future__ import annotations

import numpy as np

from oral_virus_gpt.uq.risk_tier import RiskTier, RiskTierPolicy, calibrate_thresholds


def test_low_risk_when_singleton_and_low_entropy() -> None:
    policy = RiskTierPolicy(low_entropy_threshold=0.1, high_entropy_threshold=0.5)
    assert policy.assign(set_size=1, entropy=0.05) == RiskTier.LOW


def test_medium_when_set_size_two_and_moderate_entropy() -> None:
    policy = RiskTierPolicy(low_entropy_threshold=0.1, high_entropy_threshold=0.5)
    assert policy.assign(set_size=2, entropy=0.2) == RiskTier.MEDIUM


def test_high_when_large_set_or_high_entropy() -> None:
    policy = RiskTierPolicy(low_entropy_threshold=0.1, high_entropy_threshold=0.5)
    assert policy.assign(set_size=5, entropy=0.0) == RiskTier.HIGH
    assert policy.assign(set_size=1, entropy=2.0) != RiskTier.LOW


def test_calibrate_thresholds_returns_valid_policy() -> None:
    rng = np.random.default_rng(0)
    n = 200
    set_sizes = rng.integers(1, 4, size=n)
    entropies = rng.random(n) * 2.0
    correct = (entropies < 0.5).astype(np.int64)
    policy = calibrate_thresholds(set_sizes, entropies, correct)
    assert policy.high_entropy_threshold >= policy.low_entropy_threshold
