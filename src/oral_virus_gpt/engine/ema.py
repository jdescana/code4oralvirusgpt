from __future__ import annotations

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.decay = decay
        self.shadow = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            shadow = self.shadow.get(n)
            if shadow is None:
                self.shadow[n] = p.detach().clone()
                continue
            shadow.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {n: t.clone() for n, t in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {n: t.clone() for n, t in state.items()}
