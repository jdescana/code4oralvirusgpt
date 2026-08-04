from __future__ import annotations

import torch
from torch import Tensor, nn

from oral_virus_gpt.fusion.token_xattn import scaled_dot_product


class SemanticCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_concept_slots: int = 16, num_heads: int = 16) -> None:
        super().__init__()
        if num_concept_slots <= 0:
            raise ValueError("num_concept_slots must be positive")
        self.d_model = d_model
        self.num_concept_slots = num_concept_slots
        self.num_heads = num_heads
        self.concept_query = nn.Parameter(torch.randn(num_concept_slots, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, modality_tokens: Tensor) -> Tensor:
        b = modality_tokens.shape[0]
        q = self.concept_query.unsqueeze(0).expand(b, -1, -1)
        k = self.k_proj(modality_tokens)
        v = self.v_proj(modality_tokens)
        return scaled_dot_product(q, k, v, self.num_heads)
