from __future__ import annotations

import torch

from oral_virus_gpt.losses.ece_loss import SoftBinECE


def test_soft_bin_ece_zero_when_perfectly_calibrated() -> None:
    n = 200
    targets = torch.zeros(n, dtype=torch.long)
    probs = torch.zeros(n, 2)
    probs[:, 0] = 1.0
    loss = SoftBinECE()(probs, targets)
    assert loss.item() < 1.0e-3


def test_soft_bin_ece_positive_under_overconfidence() -> None:
    n = 200
    probs = torch.zeros(n, 2)
    probs[:, 0] = 0.99
    probs[:, 1] = 0.01
    targets = torch.zeros(n, dtype=torch.long)
    targets[: n // 2] = 1
    loss = SoftBinECE()(probs, targets)
    assert loss.item() > 0.05
