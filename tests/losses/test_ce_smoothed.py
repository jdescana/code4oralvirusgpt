from __future__ import annotations

import torch
from torch.nn import functional as F

from oral_virus_gpt.losses.ce_smoothed import SmoothedCE


def test_label_smoothing_zero_matches_standard_ce() -> None:
    logits = torch.randn(4, 6)
    targets = torch.tensor([0, 1, 2, 3])
    a = SmoothedCE(label_smoothing=0.0)(logits, targets)
    b = F.cross_entropy(logits, targets)
    assert torch.allclose(a, b)


def test_label_smoothing_increases_loss_on_certain_logits() -> None:
    logits = torch.zeros(2, 5)
    logits[0, 0] = 100.0
    logits[1, 1] = 100.0
    targets = torch.tensor([0, 1])
    a = SmoothedCE(label_smoothing=0.0)(logits, targets)
    b = SmoothedCE(label_smoothing=0.1)(logits, targets)
    assert b.item() > a.item()
