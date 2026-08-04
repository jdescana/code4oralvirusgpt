from __future__ import annotations

import torch

from oral_virus_gpt.uq.temperature import TemperatureScaler


def test_temperature_init_passes_through() -> None:
    scaler = TemperatureScaler(init=1.0)
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(scaler(logits), logits)


def test_temperature_fit_reduces_nll_or_holds() -> None:
    torch.manual_seed(0)
    n = 64
    logits = torch.randn(n, 5) * 5.0
    labels = logits.argmax(dim=-1)
    scaler = TemperatureScaler(init=2.0)
    report = scaler.fit(logits, labels)
    assert report["nll_after"] <= report["nll_before"] + 1.0e-6
    assert report["tau"] > 0.0
