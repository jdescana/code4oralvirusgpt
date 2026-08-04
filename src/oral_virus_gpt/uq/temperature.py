from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, init: float = 1.0) -> None:
        super().__init__()
        if init <= 0:
            raise ValueError("temperature init must be positive")
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init))))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        max_iter: int = 50,
        tolerance: float = 1.0e-4,
    ) -> dict[str, float]:
        logits_d = logits.detach()
        labels_d = labels.detach()
        optim = torch.optim.LBFGS(
            [self.log_temperature],
            lr=0.1,
            max_iter=max_iter,
            tolerance_grad=tolerance,
            tolerance_change=tolerance,
        )

        def closure() -> Tensor:
            optim.zero_grad()
            loss = F.cross_entropy(self(logits_d), labels_d)
            loss.backward()
            return loss

        loss_before = float(F.cross_entropy(self(logits_d), labels_d).item())
        optim.step(closure)
        loss_after = float(F.cross_entropy(self(logits_d), labels_d).item())
        return {
            "tau": float(self.temperature.item()),
            "nll_before": loss_before,
            "nll_after": loss_after,
        }
