from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from oral_virus_gpt.engine.checkpoint import CheckpointPayload, load_checkpoint, save_checkpoint


def test_checkpoint_atomic_roundtrip(tmp_path: Path) -> None:
    model = nn.Linear(4, 3)
    target = tmp_path / "ckpt.pt"
    payload = CheckpointPayload(
        seed=42,
        phase="phase2",
        step=100,
        epoch=5,
        model_state=model.state_dict(),
        tau=1.32,
        cp_qhat=0.85,
        tier_thresholds={"low": 0.1, "high": 0.5},
    )
    save_checkpoint(payload, target)
    new_model = nn.Linear(4, 3)
    loaded = load_checkpoint(target, new_model)
    assert loaded.seed == 42
    assert loaded.phase == "phase2"
    assert loaded.tau == 1.32
    assert loaded.tier_thresholds == {"low": 0.1, "high": 0.5}
    for k in model.state_dict():
        assert torch.allclose(model.state_dict()[k], new_model.state_dict()[k])


def test_no_temp_files_left_behind(tmp_path: Path) -> None:
    target = tmp_path / "ckpt.pt"
    payload = CheckpointPayload(seed=0, phase="x", step=0, epoch=0, model_state={})
    save_checkpoint(payload, target)
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".ckpt-tmp-")]
    assert leftovers == []
