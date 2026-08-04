from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from oral_virus_gpt.fusion.hgcf import HGCF
from oral_virus_gpt.losses.joint import JointObjective, JointWeights
from oral_virus_gpt.utils.logging_setup import get_logger

logger = get_logger("engine.stage_b")


@dataclass(slots=True)
class TrainStepResult:
    loss: float
    ce: float
    ece: float
    reg: float


class HGCFJointTrainer:
    def __init__(
        self,
        hgcf: HGCF,
        weights: JointWeights | None = None,
        lr: float = 5.0e-5,
        warmup_steps: int = 500,
        total_steps: int | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        self.hgcf = hgcf
        self.objective = JointObjective(weights)
        params = [p for p in hgcf.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if total_steps is not None:
            self.scheduler = self._build_scheduler(lr=lr)
        self.step_idx = 0

    def _build_scheduler(self, lr: float) -> torch.optim.lr_scheduler.LRScheduler:
        warmup = max(self.warmup_steps, 1)
        total = max(self.total_steps or 1, warmup + 1)

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def step(
        self,
        photo_tokens: Tensor,
        radio_tokens: Tensor,
        text_tokens: Tensor,
        labels: Tensor,
        photo_present: Tensor | None = None,
        radio_present: Tensor | None = None,
    ) -> TrainStepResult:
        self.optimizer.zero_grad(set_to_none=True)
        out = self.hgcf(
            photo_tokens=photo_tokens,
            radio_tokens=radio_tokens,
            text_tokens=text_tokens,
            photo_present=photo_present,
            radio_present=radio_present,
        )
        terms = self.objective(out.logits, labels, self.hgcf)
        terms.total.backward()
        nn.utils.clip_grad_norm_(self.hgcf.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.step_idx += 1
        return TrainStepResult(
            loss=float(terms.total.item()),
            ce=float(terms.ce.item()),
            ece=float(terms.ece.item()),
            reg=float(terms.reg.item()),
        )

    def fit_steps(
        self,
        batches: Iterable[dict[str, Tensor]],
        max_steps: int,
    ) -> list[TrainStepResult]:
        history = []
        for batch in batches:
            if self.step_idx >= max_steps:
                break
            res = self.step(
                photo_tokens=batch["photo_tokens"],
                radio_tokens=batch["radiograph_tokens"],
                text_tokens=batch["text_tokens"],
                labels=batch["label"],
                photo_present=batch.get("photo_present"),
                radio_present=batch.get("radiograph_present"),
            )
            history.append(res)
            if self.step_idx % 50 == 0:
                logger.info(
                    "phase2 step=%d total=%.4f ce=%.4f ece=%.4f reg=%.4f",
                    self.step_idx,
                    res.loss,
                    res.ce,
                    res.ece,
                    res.reg,
                )
        return history
