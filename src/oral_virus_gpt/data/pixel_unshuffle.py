from __future__ import annotations

import torch
from torch import Tensor


def unshuffle(x: Tensor, factor: int = 2) -> Tensor:
    if factor <= 0:
        raise ValueError("factor must be positive")
    if x.dim() != 4:
        raise ValueError("expected 4D tensor [B, C, H, W]")
    b, c, h, w = x.shape
    if h % factor != 0 or w % factor != 0:
        raise ValueError(f"H={h} W={w} not divisible by factor={factor}")
    out = x.reshape(b, c, h // factor, factor, w // factor, factor)
    out = out.permute(0, 1, 3, 5, 2, 4).reshape(b, c * factor * factor, h // factor, w // factor)
    return out


def shuffle(x: Tensor, factor: int = 2) -> Tensor:
    return torch.nn.functional.pixel_shuffle(x, factor)
