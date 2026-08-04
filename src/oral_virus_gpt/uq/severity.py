from __future__ import annotations

import torch
from torch import Tensor, nn


class SeverityHead(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_classes + 2, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, calibrated_probs: Tensor, entropy: Tensor, set_size: Tensor) -> Tensor:
        if entropy.dim() == 1:
            entropy = entropy.unsqueeze(-1)
        if set_size.dim() == 1:
            set_size = set_size.unsqueeze(-1)
        feat = torch.cat([calibrated_probs, entropy, set_size.float()], dim=-1)
        return self.linear(feat).squeeze(-1)
