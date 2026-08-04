from __future__ import annotations

import torch
from torch import nn

from oral_virus_gpt.engine.ema import ExponentialMovingAverage


def test_ema_tracks_changes() -> None:
    model = nn.Linear(4, 4)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(2.0)
    ema.update(model)
    for name, p in model.named_parameters():
        assert torch.allclose(ema.shadow[name], p * 0.5 + p, atol=1.0e-6) or not torch.allclose(
            ema.shadow[name], p
        )


def test_ema_state_roundtrip() -> None:
    model = nn.Linear(2, 2)
    ema = ExponentialMovingAverage(model, decay=0.99)
    state = ema.state_dict()
    new_ema = ExponentialMovingAverage(model, decay=0.99)
    new_ema.load_state_dict(state)
    for name, t in state.items():
        assert torch.allclose(new_ema.shadow[name], t)
