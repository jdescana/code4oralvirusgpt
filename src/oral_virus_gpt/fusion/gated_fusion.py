from __future__ import annotations

import torch
from torch import Tensor, nn


class SigmoidGate(nn.Module):
    def __init__(self, num_concept_slots: int, num_modalities: int = 3, init: str = "zero") -> None:
        super().__init__()
        if num_modalities not in (2, 3):
            raise ValueError("num_modalities must be 2 or 3")
        flat = num_concept_slots * num_modalities
        self.num_concept_slots = num_concept_slots
        self.num_modalities = num_modalities
        self.gate_linear = nn.Linear(flat, flat, bias=True)
        if init == "zero":
            nn.init.zeros_(self.gate_linear.weight)
            nn.init.zeros_(self.gate_linear.bias)
        elif init == "small":
            nn.init.normal_(self.gate_linear.weight, std=0.01)
            nn.init.zeros_(self.gate_linear.bias)
        else:
            raise ValueError(f"unknown init {init!r}")

    def forward(self, semantic_modalities: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(semantic_modalities) != self.num_modalities:
            raise ValueError(
                f"expected {self.num_modalities} modality tensors, got {len(semantic_modalities)}"
            )
        b, k, _d = semantic_modalities[0].shape
        if k != self.num_concept_slots:
            raise ValueError(f"semantic tensor has {k} slots, expected {self.num_concept_slots}")
        pooled = torch.stack([t.mean(dim=-1) for t in semantic_modalities], dim=-1)
        flat = pooled.reshape(b, k * self.num_modalities)
        gates = torch.sigmoid(self.gate_linear(flat)).reshape(b, k, self.num_modalities)
        weighted = torch.zeros_like(semantic_modalities[0])
        for m in range(self.num_modalities):
            weighted = weighted + gates[:, :, m].unsqueeze(-1) * semantic_modalities[m]
        return weighted, gates


class GatedResidual(nn.Module):
    def __init__(self, fused_dim: int, hidden_dim: int, alpha_init: float = 0.0) -> None:
        super().__init__()
        self.proj = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.alpha_res = nn.Parameter(torch.full((1,), float(alpha_init)))

    def forward(self, hidden_states: Tensor, fused: Tensor) -> Tensor:
        scale = torch.tanh(self.alpha_res)
        if fused.dim() == 3 and hidden_states.dim() == 3:
            pooled = fused.mean(dim=1)
            projected = self.proj(pooled).unsqueeze(1)
        else:
            projected = self.proj(fused)
        return hidden_states + scale * projected
