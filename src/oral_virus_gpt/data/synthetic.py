from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(slots=True)
class SyntheticSample:
    photo_tokens: Tensor
    radiograph_tokens: Tensor
    text_tokens: Tensor
    photo_present: Tensor
    radiograph_present: Tensor
    label: Tensor
    patient_id: str


class SyntheticTriModalDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        num_samples: int = 16,
        num_classes: int = 5,
        hidden_dim: int = 32,
        photo_tokens: int = 8,
        radio_tokens: int = 8,
        text_tokens: int = 6,
        seed: int = 0,
    ) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be at least 2")
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.photo_tokens = photo_tokens
        self.radio_tokens = radio_tokens
        self.text_tokens = text_tokens
        gen = torch.Generator().manual_seed(seed)
        self.photo = torch.randn(num_samples, photo_tokens, hidden_dim, generator=gen)
        self.radio = torch.randn(num_samples, radio_tokens, hidden_dim, generator=gen)
        self.text = torch.randn(num_samples, text_tokens, hidden_dim, generator=gen)
        scoring = torch.randn(hidden_dim, num_classes, generator=gen)
        signal = (self.photo.mean(dim=1) + self.radio.mean(dim=1) + self.text.mean(dim=1)) / 3.0
        logits = signal @ scoring
        self.labels = logits.argmax(dim=-1).long()
        self.photo_present = torch.ones(num_samples)
        self.radio_present = torch.ones(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        return {
            "photo_tokens": self.photo[index],
            "radiograph_tokens": self.radio[index],
            "text_tokens": self.text[index],
            "photo_present": self.photo_present[index],
            "radiograph_present": self.radio_present[index],
            "label": self.labels[index],
            "patient_id": f"p{index:06d}",
        }
