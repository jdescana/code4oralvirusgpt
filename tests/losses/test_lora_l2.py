from __future__ import annotations

import torch
from torch import nn

from oral_virus_gpt.losses.lora_l2 import LoraL2


class _StubLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen_w = nn.Parameter(torch.ones(8, 8), requires_grad=False)
        self.lora_A = nn.Parameter(torch.ones(8, 4), requires_grad=True)
        self.lora_B = nn.Parameter(torch.ones(4, 8), requires_grad=True)


def test_loraL2_only_targets_lora_parameters() -> None:
    model = _StubLM()
    loss = LoraL2(weight=1.0)
    value = loss(model)
    expected = float((model.lora_A.pow(2).sum() + model.lora_B.pow(2).sum()).item())
    assert abs(value.item() - expected) < 1.0e-6


def test_loraL2_zero_when_no_lora_params() -> None:
    model = nn.Linear(4, 4)
    loss = LoraL2(weight=1.0)
    assert loss(model).item() == 0.0
