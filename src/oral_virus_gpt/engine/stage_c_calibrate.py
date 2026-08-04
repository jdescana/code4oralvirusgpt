from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from oral_virus_gpt.uq.mc_dropout import predictive_entropy
from oral_virus_gpt.uq.raps import RAPSPredictor
from oral_virus_gpt.uq.risk_tier import RiskTierPolicy, calibrate_thresholds
from oral_virus_gpt.uq.temperature import TemperatureScaler


@dataclass(slots=True)
class CalibrationArtefact:
    tau: float
    qhat: float
    tier_policy: RiskTierPolicy


class CalibrationFitter:
    def __init__(
        self,
        temperature: TemperatureScaler,
        raps: RAPSPredictor,
        target_low_acc: float = 0.95,
        target_high_recall: float = 0.90,
        medium_set_size: int = 2,
    ) -> None:
        self.temperature = temperature
        self.raps = raps
        self.target_low_acc = target_low_acc
        self.target_high_recall = target_high_recall
        self.medium_set_size = medium_set_size

    def fit(self, mean_logits: Tensor, labels: Tensor) -> CalibrationArtefact:
        if mean_logits.shape[0] != labels.shape[0]:
            raise ValueError("logits and labels disagree on batch")
        report = self.temperature.fit(mean_logits, labels)
        tau = float(report["tau"])
        with torch.no_grad():
            calibrated = self.temperature(mean_logits)
            probs = torch.softmax(calibrated, dim=-1)
        qhat = self.raps.calibrate(probs, labels)
        with torch.no_grad():
            sets = self.raps.predict_sets(probs)
            entropies = predictive_entropy(probs).cpu().numpy().astype(np.float64)
            preds = probs.argmax(dim=-1).cpu().numpy().astype(np.int64)
            set_sizes = np.array([len(s) for s in sets], dtype=np.int64)
        labels_np = labels.cpu().numpy().astype(np.int64)
        correct = (preds == labels_np).astype(np.int64)
        tier_policy = calibrate_thresholds(
            set_sizes=set_sizes,
            entropies=entropies,
            correct=correct,
            target_low_acc=self.target_low_acc,
            target_high_recall=self.target_high_recall,
            medium_set_size=self.medium_set_size,
        )
        return CalibrationArtefact(tau=tau, qhat=qhat, tier_policy=tier_policy)
