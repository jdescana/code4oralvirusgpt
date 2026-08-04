from __future__ import annotations

import torch

from oral_virus_gpt.fusion.gated_fusion import GatedResidual, SigmoidGate


def test_gates_in_unit_interval() -> None:
    gate = SigmoidGate(num_concept_slots=4, num_modalities=3, init="small")
    a = torch.rand(2, 4, 8)
    b = torch.rand(2, 4, 8)
    c = torch.rand(2, 4, 8)
    fused, gates = gate([a, b, c])
    assert fused.shape == a.shape
    assert torch.all(gates >= 0.0) and torch.all(gates <= 1.0)


def test_zero_init_makes_gates_half() -> None:
    gate = SigmoidGate(num_concept_slots=4, num_modalities=3, init="zero")
    a = torch.rand(1, 4, 8)
    b = torch.rand(1, 4, 8)
    c = torch.rand(1, 4, 8)
    _, gates = gate([a, b, c])
    assert torch.allclose(gates, torch.full_like(gates, 0.5), atol=1.0e-6)


def test_gated_residual_zero_alpha_returns_input() -> None:
    block = GatedResidual(fused_dim=8, hidden_dim=8, alpha_init=0.0)
    h = torch.rand(2, 5, 8)
    fused = torch.rand(2, 4, 8)
    out = block(h, fused)
    assert torch.allclose(out, h)
