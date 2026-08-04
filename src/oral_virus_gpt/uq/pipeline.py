from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from oral_virus_gpt.uq.mc_dropout import predictive_entropy
from oral_virus_gpt.uq.raps import RAPSPredictor
from oral_virus_gpt.uq.risk_tier import RiskTierPolicy
from oral_virus_gpt.uq.severity import SeverityHead
from oral_virus_gpt.uq.temperature import TemperatureScaler


@dataclass(slots=True)
class UQPrediction:
    probs: NDArray[np.floating[Any]]
    entropies: NDArray[np.floating[Any]]
    sets: list[list[int]]
    set_sizes: NDArray[np.integer[Any]]
    tiers: NDArray[Any]
    severity: NDArray[np.floating[Any]] | None


class UQPipeline:
    def __init__(
        self,
        temperature: TemperatureScaler,
        raps: RAPSPredictor,
        tier_policy: RiskTierPolicy,
        severity: SeverityHead | None = None,
    ) -> None:
        self.temperature = temperature
        self.raps = raps
        self.tier_policy = tier_policy
        self.severity = severity

    @torch.no_grad()
    def __call__(self, mean_logits: Tensor) -> UQPrediction:
        calibrated_logits = self.temperature(mean_logits)
        probs = torch.softmax(calibrated_logits, dim=-1)
        entropy = predictive_entropy(probs)
        sets = self.raps.predict_sets(probs)
        set_sizes = np.array([len(s) for s in sets], dtype=np.int64)
        entropy_np = entropy.detach().cpu().numpy().astype(np.float64)
        tiers = self.tier_policy.assign_batch(set_sizes, entropy_np)
        severity = None
        if self.severity is not None:
            sev = self.severity(probs, entropy, torch.from_numpy(set_sizes).to(probs.device))
            severity = sev.detach().cpu().numpy()
        return UQPrediction(
            probs=probs.detach().cpu().numpy(),
            entropies=entropy_np,
            sets=sets,
            set_sizes=set_sizes,
            tiers=tiers,
            severity=severity,
        )
