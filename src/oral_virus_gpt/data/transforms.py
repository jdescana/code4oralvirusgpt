from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

PHOTO_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
PHOTO_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
RADIO_MEAN: float = 0.45
RADIO_STD: float = 0.225


@dataclass(slots=True)
class PhotoStats:
    mean: tuple[float, float, float] = PHOTO_MEAN
    std: tuple[float, float, float] = PHOTO_STD


@dataclass(slots=True)
class RadiographStats:
    mean: float = RADIO_MEAN
    std: float = RADIO_STD
    clip_low_pct: float = 0.5
    clip_high_pct: float = 99.5


def photo_normalize(image: Tensor, stats: PhotoStats | None = None) -> Tensor:
    s = stats if stats is not None else PhotoStats()
    if image.dim() != 3 or image.shape[0] != 3:
        raise ValueError("photo must be 3xHxW")
    mean = torch.tensor(s.mean, device=image.device, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(s.std, device=image.device, dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


def radiograph_normalize(image: Tensor, stats: RadiographStats | None = None) -> Tensor:
    s = stats if stats is not None else RadiographStats()
    flat = image.flatten()
    if flat.numel() == 0:
        return image
    lo = torch.quantile(flat, s.clip_low_pct / 100.0)
    hi = torch.quantile(flat, s.clip_high_pct / 100.0)
    clipped = image if (hi - lo).abs() < 1.0e-8 else ((image - lo) / (hi - lo)).clamp(0.0, 1.0)
    return (clipped - s.mean) / s.std
