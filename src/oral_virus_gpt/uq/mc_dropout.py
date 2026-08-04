from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


def _enable_dropout_inference(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Dropout | nn.Dropout1d | nn.Dropout2d | nn.Dropout3d):
            m.train()


class MCDropoutEnsemble:
    def __init__(self, model: nn.Module, num_samples: int = 20) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.model = model
        self.num_samples = num_samples

    @torch.no_grad()
    def collect(self, forward: Callable[[], Tensor]) -> Tensor:
        was_training = self.model.training
        self.model.eval()
        _enable_dropout_inference(self.model)
        try:
            samples = torch.stack([forward() for _ in range(self.num_samples)], dim=0)
        finally:
            if was_training:
                self.model.train()
        return samples

    @torch.no_grad()
    def predict_logits(self, forward: Callable[[], Tensor]) -> Tensor:
        return self.collect(forward)

    @torch.no_grad()
    def predict_probs(self, forward: Callable[[], Tensor]) -> tuple[Tensor, Tensor]:
        logits = self.collect(forward)
        probs = torch.softmax(logits, dim=-1)
        mean = probs.mean(dim=0)
        return logits.mean(dim=0), mean


def predictive_entropy(probs: Tensor, eps: float = 1.0e-12) -> Tensor:
    safe = probs.clamp_min(eps)
    return -(safe * safe.log()).sum(dim=-1)
