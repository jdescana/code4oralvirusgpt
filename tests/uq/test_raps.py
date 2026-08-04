from __future__ import annotations

import torch

from oral_virus_gpt.uq.raps import RAPSPredictor


def test_raps_calibrated_coverage_meets_target() -> None:
    torch.manual_seed(0)
    n, k = 200, 5
    logits = torch.randn(n, k)
    probs = torch.softmax(logits, dim=-1)
    labels = probs.argmax(dim=-1)
    predictor = RAPSPredictor(alpha=0.1, penalty=0.0, randomized=False)
    predictor.calibrate(probs, labels)
    coverage = predictor.coverage(probs, labels)
    assert coverage >= 0.85


def test_raps_sets_are_non_empty_when_disallow_zero() -> None:
    torch.manual_seed(0)
    n, k = 50, 3
    probs = torch.softmax(torch.randn(n, k), dim=-1)
    labels = probs.argmax(dim=-1)
    predictor = RAPSPredictor(alpha=0.5, penalty=0.5, disallow_zero=True, randomized=False)
    predictor.calibrate(probs, labels)
    sets = predictor.predict_sets(probs)
    assert all(len(s) >= 1 for s in sets)
