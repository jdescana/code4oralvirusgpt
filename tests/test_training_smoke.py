from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from oral_virus_gpt.data.synthetic import SyntheticTriModalDataset
from oral_virus_gpt.engine.stage_b_hgcf import HGCFJointTrainer
from oral_virus_gpt.fusion.hgcf import HGCF, HGCFConfig
from oral_virus_gpt.losses.joint import JointWeights


def test_smoke_two_step_loss_decreases() -> None:
    torch.manual_seed(0)
    cfg = HGCFConfig(hidden_dim=32, num_heads=4, num_concept_slots=4, num_classes=5)
    hgcf = HGCF(cfg)
    weights = JointWeights(label_smoothing=0.0, lambda_cal=0.0, lambda_reg=0.0)
    trainer = HGCFJointTrainer(hgcf, weights=weights, lr=1.0e-2, warmup_steps=0, total_steps=2)
    dataset = SyntheticTriModalDataset(
        num_samples=8, num_classes=5, hidden_dim=32, photo_tokens=6, radio_tokens=6, text_tokens=8
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    history = []
    iters = iter(loader)
    for _ in range(2):
        batch = next(iters)
        res = trainer.step(
            photo_tokens=batch["photo_tokens"],
            radio_tokens=batch["radiograph_tokens"],
            text_tokens=batch["text_tokens"],
            labels=batch["label"].long(),
            photo_present=torch.as_tensor(batch["photo_present"]).float(),
            radio_present=torch.as_tensor(batch["radiograph_present"]).float(),
        )
        history.append(res)
    assert len(history) == 2
    assert history[1].loss < history[0].loss + 1.0e-4
