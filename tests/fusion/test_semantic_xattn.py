from __future__ import annotations

import torch

from oral_virus_gpt.fusion.semantic_xattn import SemanticCrossAttention


def test_semantic_output_per_modality_shape() -> None:
    block = SemanticCrossAttention(d_model=16, num_concept_slots=4, num_heads=4)
    tokens = torch.rand(3, 9, 16)
    out = block(tokens)
    assert out.shape == (3, 4, 16)
