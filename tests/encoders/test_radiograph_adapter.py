from __future__ import annotations

import torch

from oral_virus_gpt.encoders.radiograph import RadiographAdapter


def test_adapter_shape_preserved() -> None:
    adapter = RadiographAdapter(in_channels=3, mid_channels=8)
    x = torch.rand(2, 3, 32, 32)
    y = adapter(x)
    assert y.shape == x.shape


def test_adapter_residual_changes_output_only_modestly() -> None:
    torch.manual_seed(0)
    adapter = RadiographAdapter(in_channels=3, mid_channels=4)
    x = torch.rand(1, 3, 16, 16)
    y = adapter(x)
    diff = (y - x).abs().mean().item()
    assert diff < 1.0


def test_adapter_only_adapter_params_have_gradients() -> None:
    adapter = RadiographAdapter(in_channels=3, mid_channels=4)
    x = torch.rand(1, 3, 16, 16, requires_grad=False)
    y = adapter(x)
    y.mean().backward()
    grad_count = sum(1 for p in adapter.parameters() if p.grad is not None)
    assert grad_count == sum(1 for _ in adapter.parameters())
