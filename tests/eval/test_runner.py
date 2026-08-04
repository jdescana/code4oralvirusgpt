from __future__ import annotations

import torch

from oral_virus_gpt.eval.runner import EvalBatch, run_eval_loop
from oral_virus_gpt.fusion.hgcf import HGCF, HGCFConfig
from oral_virus_gpt.uq.pipeline import UQPipeline
from oral_virus_gpt.uq.raps import RAPSPredictor
from oral_virus_gpt.uq.risk_tier import RiskTierPolicy
from oral_virus_gpt.uq.temperature import TemperatureScaler


def _build_pipeline(num_classes: int = 4) -> UQPipeline:
    raps = RAPSPredictor(alpha=0.1, penalty=0.0, randomized=False)
    cal_logits = torch.randn(64, num_classes) * 2.0
    cal_probs = torch.softmax(cal_logits, dim=-1)
    cal_labels = cal_probs.argmax(dim=-1)
    raps.calibrate(cal_probs, cal_labels)
    return UQPipeline(
        temperature=TemperatureScaler(init=1.0),
        raps=raps,
        tier_policy=RiskTierPolicy(low_entropy_threshold=0.1, high_entropy_threshold=0.5),
    )


def test_eval_loop_emits_required_columns() -> None:
    torch.manual_seed(0)
    cfg = HGCFConfig(hidden_dim=16, num_heads=4, num_concept_slots=4, num_classes=4)
    hgcf = HGCF(cfg)
    pipeline = _build_pipeline(num_classes=4)
    batches = [
        EvalBatch(
            photo=torch.rand(2, 6, 16),
            radiograph=torch.rand(2, 6, 16),
            text=torch.rand(2, 8, 16),
            labels=torch.tensor([0, 1]),
            photo_present=torch.ones(2),
            radiograph_present=torch.ones(2),
        )
    ]
    res = run_eval_loop(hgcf, batches, pipeline=pipeline, mc_samples=2, num_classes=4)
    assert res.n == 2
    assert 0.0 <= res.coverage <= 1.0
    assert res.set_size >= 1.0
