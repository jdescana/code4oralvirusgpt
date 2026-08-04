from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from oral_virus_gpt.utils.logging_setup import get_logger

logger = get_logger("engine.stage_a")


class AdapterContrastiveTrainer:
    def __init__(
        self,
        adapter: nn.Module,
        photo_encoder: nn.Module,
        radiograph_encoder: nn.Module,
        text_encoder: nn.Module | None,
        lr: float = 1.0e-4,
        weight_decay: float = 0.0,
        temperature: float = 0.07,
    ) -> None:
        self.adapter = adapter
        self.photo_encoder = photo_encoder
        self.radiograph_encoder = radiograph_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature
        params = [p for p in self.adapter.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )

    def info_nce(self, anchors: Tensor, positives: Tensor) -> Tensor:
        if anchors.shape != positives.shape:
            raise ValueError("anchors and positives must share shape")
        a = F.normalize(anchors.flatten(1), dim=-1)
        b = F.normalize(positives.flatten(1), dim=-1)
        logits = a @ b.t() / self.temperature
        targets = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))

    def step(self, photo: Tensor, radiograph: Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        photo_tokens = self.photo_encoder(photo)
        radio_tokens = self.radiograph_encoder(radiograph)
        loss = self.info_nce(photo_tokens.mean(dim=1), radio_tokens.mean(dim=1))
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def train(self, batches: Iterable[tuple[Tensor, Tensor]]) -> list[float]:
        history = []
        for batch_idx, (photo, radio) in enumerate(batches):
            loss = self.step(photo, radio)
            history.append(loss)
            if batch_idx % 50 == 0:
                logger.info("phase1 step=%d loss=%.4f", batch_idx, loss)
        return history
