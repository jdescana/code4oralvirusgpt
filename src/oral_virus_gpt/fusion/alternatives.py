from __future__ import annotations

import torch
from torch import Tensor, nn


class ConcatenationFusion(nn.Module):
    def __init__(self, d_model: int, num_modalities: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model * num_modalities, d_model, bias=False)

    def forward(self, modality_tensors: list[Tensor]) -> Tensor:
        pooled = [t.mean(dim=1) for t in modality_tensors]
        joined = torch.cat(pooled, dim=-1)
        return self.proj(joined).unsqueeze(1)


class WeightedAverageFusion(nn.Module):
    def __init__(self, d_model: int, num_modalities: int = 3) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.full((num_modalities,), 1.0 / num_modalities))
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, modality_tensors: list[Tensor]) -> Tensor:
        if len(modality_tensors) != self.weights.shape[0]:
            raise ValueError("weights and modality count mismatch")
        normed = torch.softmax(self.weights, dim=0)
        pooled = [t.mean(dim=1) for t in modality_tensors]
        weighted = sum(w * p for w, p in zip(normed, pooled, strict=True))
        assert isinstance(weighted, Tensor)
        return self.proj(weighted).unsqueeze(1)


class StackedTransformerFusion(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 2, num_heads: int = 16) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, modality_tensors: list[Tensor]) -> Tensor:
        joined = torch.cat(modality_tensors, dim=1)
        return self.encoder(joined)
