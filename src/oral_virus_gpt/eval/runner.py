from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from oral_virus_gpt.fusion.hgcf import HGCF
from oral_virus_gpt.metrics.calibration import brier, ece
from oral_virus_gpt.metrics.classification import accuracy, auc, macro_f1
from oral_virus_gpt.metrics.conformal import marginal_coverage, mean_set_size
from oral_virus_gpt.uq.mc_dropout import MCDropoutEnsemble
from oral_virus_gpt.uq.pipeline import UQPipeline


@dataclass(slots=True)
class EvalBatch:
    photo: Tensor
    radiograph: Tensor
    text: Tensor
    labels: Tensor
    photo_present: Tensor
    radiograph_present: Tensor


@dataclass(slots=True)
class EvalResult:
    accuracy: float
    macro_f1: float
    auc: float
    ece: float
    brier: float
    coverage: float
    set_size: float
    n: int


@torch.no_grad()
def run_eval_loop(
    hgcf: HGCF,
    batches: Iterable[EvalBatch],
    pipeline: UQPipeline,
    mc_samples: int = 20,
    num_classes: int | None = None,
) -> EvalResult:
    all_probs = []
    all_labels = []
    all_sets: list[list[int]] = []
    for batch in batches:
        ensemble = MCDropoutEnsemble(hgcf, num_samples=mc_samples)

        def _forward(b: EvalBatch = batch) -> Tensor:
            return hgcf(b.photo, b.radiograph, b.text, b.photo_present, b.radiograph_present).logits

        mean_logits, _ = ensemble.predict_probs(_forward)
        prediction = pipeline(mean_logits)
        all_probs.append(prediction.probs)
        all_labels.append(batch.labels.detach().cpu().numpy().astype(np.int64))
        all_sets.extend(prediction.sets)
    if not all_probs:
        return EvalResult(
            accuracy=float("nan"),
            macro_f1=float("nan"),
            auc=float("nan"),
            ece=float("nan"),
            brier=float("nan"),
            coverage=float("nan"),
            set_size=float("nan"),
            n=0,
        )
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = probs.argmax(axis=1)
    del num_classes
    return EvalResult(
        accuracy=accuracy(preds, labels),
        macro_f1=macro_f1(preds, labels),
        auc=auc(probs, labels),
        ece=ece(probs, labels),
        brier=brier(probs, labels),
        coverage=marginal_coverage(all_sets, labels),
        set_size=mean_set_size(all_sets),
        n=int(probs.shape[0]),
    )
