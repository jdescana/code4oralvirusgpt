from __future__ import annotations

import torch

from oral_virus_gpt.uq.pipeline import UQPipeline
from oral_virus_gpt.uq.raps import RAPSPredictor
from oral_virus_gpt.uq.risk_tier import RiskTierPolicy
from oral_virus_gpt.uq.temperature import TemperatureScaler


def test_pipeline_returns_full_prediction_dict() -> None:
    torch.manual_seed(0)
    raps = RAPSPredictor(alpha=0.1, penalty=0.0, randomized=False)
    cal_logits = torch.randn(64, 5) * 2.0
    cal_probs = torch.softmax(cal_logits, dim=-1)
    cal_labels = cal_probs.argmax(dim=-1)
    raps.calibrate(cal_probs, cal_labels)
    pipeline = UQPipeline(
        temperature=TemperatureScaler(init=1.0),
        raps=raps,
        tier_policy=RiskTierPolicy(low_entropy_threshold=0.1, high_entropy_threshold=0.5),
    )
    test_logits = torch.randn(4, 5)
    pred = pipeline(test_logits)
    assert pred.probs.shape == (4, 5)
    assert pred.entropies.shape == (4,)
    assert len(pred.sets) == 4
    assert pred.set_sizes.shape == (4,)
    assert pred.tiers.shape == (4,)
