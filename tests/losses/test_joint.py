from __future__ import annotations

import torch
from torch import nn

from oral_virus_gpt.losses.joint import JointObjective, JointWeights


class _StubModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.lora_A = nn.Parameter(torch.zeros(4, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 3))


def test_joint_decomposes_into_terms() -> None:
    model = _StubModel()
    obj = JointObjective(JointWeights(label_smoothing=0.0, lambda_cal=0.0, lambda_reg=0.0))
    logits = model.linear(torch.rand(4, 4))
    targets = torch.tensor([0, 1, 2, 0])
    terms = obj(logits, targets, model)
    assert torch.allclose(terms.total, terms.ce)
    assert terms.ece.item() >= 0.0
    assert terms.reg.item() == 0.0
