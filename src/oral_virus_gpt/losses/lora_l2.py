from __future__ import annotations

import torch
from torch import Tensor, nn


def _is_lora_param(name: str) -> bool:
    lower = name.lower()
    return "lora_a" in lower or "lora_b" in lower or "lora_embedding" in lower


class LoraL2(nn.Module):
    def __init__(self, weight: float = 1.0e-4) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self.weight = weight

    def forward(self, model: nn.Module) -> Tensor:
        device = (
            next(iter(model.parameters())).device
            if any(True for _ in model.parameters())
            else torch.device("cpu")
        )
        accum = torch.zeros((), device=device)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not _is_lora_param(name):
                continue
            accum = accum + param.pow(2).sum()
        return self.weight * accum
