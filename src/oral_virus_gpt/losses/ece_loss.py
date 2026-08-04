from __future__ import annotations

import torch
from torch import Tensor, nn


class SoftBinECE(nn.Module):
    def __init__(self, num_bins: int = 15, sigma: float = 0.05) -> None:
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.num_bins = num_bins
        self.sigma = sigma
        bin_centres = torch.linspace(1.0 / (2 * num_bins), 1.0 - 1.0 / (2 * num_bins), num_bins)
        self.register_buffer("bin_centres", bin_centres)

    def forward(self, probs: Tensor, targets: Tensor) -> Tensor:
        confidences, predictions = probs.max(dim=-1)
        correct = (predictions == targets).float()
        weights = torch.exp(
            -((confidences.unsqueeze(-1) - self.bin_centres.unsqueeze(0)) ** 2)
            / (2 * self.sigma**2)
        )
        weight_sum = weights.sum(dim=0).clamp_min(1.0e-8)
        bin_acc = (weights * correct.unsqueeze(-1)).sum(dim=0) / weight_sum
        bin_conf = (weights * confidences.unsqueeze(-1)).sum(dim=0) / weight_sum
        bin_count = weights.sum(dim=0)
        total = bin_count.sum().clamp_min(1.0e-8)
        gap = (bin_acc - bin_conf).abs()
        return (bin_count / total * gap).sum()
