from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from oral_virus_gpt.losses.ce_smoothed import SmoothedCE
from oral_virus_gpt.losses.ece_loss import SoftBinECE
from oral_virus_gpt.losses.lora_l2 import LoraL2


@dataclass(slots=True)
class JointWeights:
    label_smoothing: float = 0.1
    lambda_cal: float = 0.1
    lambda_reg: float = 1.0e-4


@dataclass(slots=True)
class JointTerms:
    total: Tensor
    ce: Tensor
    ece: Tensor
    reg: Tensor


class JointObjective(nn.Module):
    def __init__(self, weights: JointWeights | None = None) -> None:
        super().__init__()
        self.weights = weights if weights is not None else JointWeights()
        self.ce = SmoothedCE(label_smoothing=self.weights.label_smoothing)
        self.ece = SoftBinECE()
        self.reg = LoraL2(weight=self.weights.lambda_reg)

    def forward(self, logits: Tensor, targets: Tensor, model: nn.Module) -> JointTerms:
        ce_term = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=-1)
        ece_term = self.ece(probs, targets)
        reg_term = self.reg(model)
        total = ce_term + self.weights.lambda_cal * ece_term + reg_term
        return JointTerms(total=total, ce=ce_term, ece=ece_term, reg=reg_term)
