from __future__ import annotations

import torch

from oral_virus_gpt.fusion.hgcf import HGCF, HGCFConfig


def _build(d: int = 16, k: int = 4, c: int = 5, style: str = "hgcf") -> HGCF:
    cfg = HGCFConfig(hidden_dim=d, num_heads=4, num_concept_slots=k, num_classes=c, style=style)
    return HGCF(cfg)


def test_hgcf_forward_shapes() -> None:
    hgcf = _build()
    out = hgcf(
        photo_tokens=torch.rand(2, 6, 16),
        radio_tokens=torch.rand(2, 6, 16),
        text_tokens=torch.rand(2, 8, 16),
    )
    assert out.logits.shape == (2, 5)
    assert out.fused.shape == (2, 4, 16)
    assert out.gates is not None and out.gates.shape == (2, 4, 3)


def test_concat_alternative_runs() -> None:
    hgcf = _build(style="concat")
    out = hgcf(torch.rand(2, 6, 16), torch.rand(2, 6, 16), torch.rand(2, 8, 16))
    assert out.logits.shape == (2, 5)
    assert out.gates is None


def test_weighted_alternative_runs() -> None:
    hgcf = _build(style="weighted")
    out = hgcf(torch.rand(2, 6, 16), torch.rand(2, 6, 16), torch.rand(2, 8, 16))
    assert out.logits.shape == (2, 5)


def test_missing_modality_zeros_radiograph_gate_implicitly() -> None:
    hgcf = _build()
    presence = torch.tensor([1.0, 0.0])
    out = hgcf(
        torch.rand(2, 6, 16),
        torch.rand(2, 6, 16),
        torch.rand(2, 8, 16),
        photo_present=torch.ones(2),
        radio_present=presence,
    )
    assert out.semantic_radio is not None
    diff = (out.semantic_radio[0] - out.semantic_radio[1]).abs().sum().item()
    assert diff > 0.0
