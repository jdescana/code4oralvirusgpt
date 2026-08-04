from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class RAPSPredictor:
    def __init__(
        self,
        alpha: float = 0.05,
        penalty: float = 0.01,
        disallow_zero: bool = True,
        randomized: bool = True,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if penalty < 0:
            raise ValueError("penalty must be non-negative")
        self.alpha = alpha
        self.penalty = penalty
        self.disallow_zero = disallow_zero
        self.randomized = randomized
        self.qhat: float | None = None

    def _scores(
        self,
        probs: Tensor,
        labels: Tensor | None,
        random_offsets: Tensor | None,
    ) -> Tensor:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        n, k = probs.shape
        ranks_of_sorted = torch.arange(1, k + 1, device=probs.device).unsqueeze(0).expand(n, k)
        cumulative = sorted_probs.cumsum(dim=-1)
        regularised = cumulative + self.penalty * (ranks_of_sorted - 1).clamp_min(0).float()
        if labels is None:
            return regularised
        rank = torch.empty(n, dtype=torch.long, device=probs.device)
        for i in range(n):
            rank[i] = (sorted_idx[i] == labels[i]).nonzero(as_tuple=True)[0][0]
        score = regularised[torch.arange(n, device=probs.device), rank]
        if self.randomized and random_offsets is not None:
            picked = sorted_probs[torch.arange(n, device=probs.device), rank]
            score = score - random_offsets * picked
        return score

    def calibrate(self, probs: Tensor, labels: Tensor) -> float:
        if probs.shape[0] != labels.shape[0]:
            raise ValueError("probs and labels disagree on batch size")
        u = torch.rand(probs.shape[0]) if self.randomized else None
        scores = self._scores(probs.detach().cpu(), labels.detach().cpu(), u)
        n = scores.shape[0]
        q_level = float(np.ceil((n + 1) * (1 - self.alpha)) / n)
        q_level = min(max(q_level, 0.0), 1.0)
        self.qhat = float(torch.quantile(scores, q_level).item())
        return self.qhat

    def predict_sets(self, probs: Tensor) -> list[list[int]]:
        if self.qhat is None:
            raise RuntimeError("RAPS not calibrated; call calibrate() first")
        probs_cpu = probs.detach().cpu()
        regularised = self._scores(probs_cpu, labels=None, random_offsets=None)
        sorted_idx = torch.argsort(probs_cpu, dim=-1, descending=True)
        sets: list[list[int]] = []
        for i in range(probs_cpu.shape[0]):
            keep = (regularised[i] <= self.qhat).nonzero(as_tuple=True)[0]
            if keep.numel() == 0 and self.disallow_zero:
                keep = torch.tensor([0], dtype=torch.long)
            included_classes = sorted_idx[i, keep].tolist()
            sets.append(sorted(included_classes))
        return sets

    def coverage(self, probs: Tensor, labels: Tensor) -> float:
        sets = self.predict_sets(probs)
        labels_list = labels.detach().cpu().tolist()
        hits = sum(1 for s, y in zip(sets, labels_list, strict=True) if y in s)
        return hits / len(sets) if sets else 0.0
