from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional as F


class SmoothedCE(nn.Module):
    def __init__(self, label_smoothing: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
