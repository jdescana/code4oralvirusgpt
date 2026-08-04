from __future__ import annotations

import torch
from torch import Tensor, nn


class NullEmbedding(nn.Module):
    def __init__(self, num_concept_slots: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(num_concept_slots, d_model))
        nn.init.normal_(self.embedding, std=0.02)

    def forward(self, batch_size: int) -> Tensor:
        return self.embedding.unsqueeze(0).expand(batch_size, -1, -1)


def mask_modality(
    semantic: Tensor,
    presence: Tensor,
    null_module: NullEmbedding,
) -> tuple[Tensor, Tensor]:
    if presence.dim() != 1 or presence.shape[0] != semantic.shape[0]:
        raise ValueError("presence must be a 1D mask matching the batch size")
    presence = presence.bool()
    null = null_module(semantic.shape[0])
    expanded = presence.unsqueeze(-1).unsqueeze(-1)
    masked = torch.where(expanded, semantic, null)
    return masked, presence.float()
