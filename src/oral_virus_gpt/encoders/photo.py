from __future__ import annotations

import torch
from torch import Tensor, nn


def freeze_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()
    return module


class PhotoEncoder(nn.Module):
    def __init__(
        self,
        vision_model: nn.Module,
        hidden_dim: int,
        output_dim: int,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        if freeze:
            freeze_module(self.vision_model)
        if hidden_dim != output_dim:
            self.proj: nn.Module = nn.Linear(hidden_dim, output_dim, bias=False)
        else:
            self.proj = nn.Identity()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    @torch.no_grad()
    def _encode(self, pixel_values: Tensor) -> Tensor:
        out = self.vision_model(pixel_values)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if isinstance(out, tuple):
            return out[0]
        return out

    def forward(self, pixel_values: Tensor) -> Tensor:
        with torch.no_grad():
            tokens = self._encode(pixel_values)
        return self.proj(tokens)
