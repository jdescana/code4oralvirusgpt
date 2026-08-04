from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(slots=True)
class CheckpointPayload:
    seed: int
    phase: str
    step: int
    epoch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    tau: float | None = None
    cp_qhat: float | None = None
    tier_thresholds: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt-tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_checkpoint(payload: CheckpointPayload, path: Path) -> None:
    serialisable = {
        "seed": payload.seed,
        "phase": payload.phase,
        "step": payload.step,
        "epoch": payload.epoch,
        "model_state": payload.model_state,
        "optimizer_state": payload.optimizer_state,
        "scheduler_state": payload.scheduler_state,
        "tau": payload.tau,
        "cp_qhat": payload.cp_qhat,
        "tier_thresholds": payload.tier_thresholds,
        "extras": payload.extras,
    }
    atomic_save(serialisable, path)


def load_checkpoint(path: Path, model: nn.Module | None = None) -> CheckpointPayload:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a dict")
    if model is not None and "model_state" in payload:
        model.load_state_dict(payload["model_state"], strict=False)
    return CheckpointPayload(
        seed=int(payload.get("seed", 0)),
        phase=str(payload.get("phase", "")),
        step=int(payload.get("step", 0)),
        epoch=int(payload.get("epoch", 0)),
        model_state=payload.get("model_state", {}),
        optimizer_state=payload.get("optimizer_state"),
        scheduler_state=payload.get("scheduler_state"),
        tau=payload.get("tau"),
        cp_qhat=payload.get("cp_qhat"),
        tier_thresholds=payload.get("tier_thresholds", {}),
        extras=payload.get("extras", {}),
    )
