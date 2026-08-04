from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

from oral_virus_gpt.data.synthetic import SyntheticTriModalDataset
from oral_virus_gpt.engine.checkpoint import CheckpointPayload, save_checkpoint
from oral_virus_gpt.engine.seed import seed_everything
from oral_virus_gpt.engine.stage_b_hgcf import HGCFJointTrainer
from oral_virus_gpt.fusion.hgcf import HGCF, HGCFConfig
from oral_virus_gpt.losses.joint import JointWeights
from oral_virus_gpt.utils.logging_setup import get_logger

logger = get_logger("runners.train")


def _select(cfg: Any, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cast(DictConfig, cfg), key, default=default)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("train", help="Run a training stage.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--phase", choices=["phase1", "phase2", "phase3"], default="phase2")
    p.add_argument("--output", type=Path, default=Path("checkpoints"))
    p.add_argument("--max-steps", type=int, default=None)
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    cfg = OmegaConf.load(args.config)
    seed = int(_select(cfg, "seed", default=0))
    seed_everything(seed)
    primary = _select(cfg, "data.primary")
    if primary != "synthetic":
        logger.warning(
            "phase=%s requested with non-synthetic data %s; this CLI runner currently supports synthetic smoke runs end-to-end",
            args.phase,
            primary,
        )
    return _run_synthetic(cfg, args)


def _run_synthetic(cfg: Any, args: argparse.Namespace) -> int:
    cfg_data = _select(cfg, "data")
    cfg_model = _select(cfg, "model")
    cfg_phase = _select(cfg, args.phase)
    if cfg_phase is None:
        raise ValueError(f"phase {args.phase!r} not present in config")
    num_classes = int(_select(cfg_data, "num_classes", default=5))
    hidden = int(_select(cfg_model, "hidden_dim", default=32))
    num_concept_slots = int(_select(cfg_model, "num_concept_slots", default=4))
    num_heads = int(_select(cfg_model, "num_heads", default=4))
    dataset = SyntheticTriModalDataset(
        num_samples=int(_select(cfg_data, "num_train", default=16)),
        num_classes=num_classes,
        hidden_dim=hidden,
    )
    loader: DataLoader[dict[str, Tensor | str]] = DataLoader(
        dataset,
        batch_size=int(_select(cfg_phase, "batch_size_per_gpu", default=2)),
        shuffle=True,
    )
    config = HGCFConfig(
        hidden_dim=hidden,
        num_heads=num_heads,
        num_concept_slots=num_concept_slots,
        num_classes=num_classes,
        use_token=bool(_select(cfg_model, "use_token", default=True)),
        use_semantic=bool(_select(cfg_model, "use_semantic", default=True)),
        use_gating=bool(_select(cfg_model, "use_gating", default=True)),
        use_uncertainty_gating=bool(_select(cfg_model, "use_uncertainty_gating", default=True)),
        style=str(_select(cfg_model, "fusion", default="hgcf")),
    )
    hgcf = HGCF(config)
    weights_cfg = _select(cfg_phase, "loss") or {}
    weights = JointWeights(
        label_smoothing=float(_select(weights_cfg, "label_smoothing", default=0.1)),
        lambda_cal=float(_select(weights_cfg, "lambda_cal", default=0.1)),
        lambda_reg=float(_select(weights_cfg, "lambda_reg", default=1.0e-4)),
    )
    trainer = HGCFJointTrainer(
        hgcf=hgcf,
        weights=weights,
        lr=float(_select(cfg_phase, "lr", default=5.0e-5)),
        warmup_steps=int(_select(cfg_phase, "warmup_steps", default=0)),
        total_steps=args.max_steps,
    )
    max_steps = args.max_steps or int(_select(cfg_phase, "steps", default=2))
    history = trainer.fit_steps(_iter_batches(loader), max_steps=max_steps)
    if not history:
        logger.error("trainer did not produce any steps")
        return 2
    payload = CheckpointPayload(
        seed=int(_select(cfg, "seed", default=0)),
        phase=args.phase,
        step=trainer.step_idx,
        epoch=0,
        model_state=hgcf.state_dict(),
        optimizer_state=trainer.optimizer.state_dict(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_checkpoint(payload, args.output / f"{args.phase}.pt")
    logger.info("wrote checkpoint to %s with %d steps", args.output, len(history))
    return 0


def _iter_batches(loader: DataLoader[dict[str, Tensor | str]]) -> Iterator[dict[str, Tensor]]:
    for raw in loader:
        yield {
            "photo_tokens": cast(Tensor, raw["photo_tokens"]),
            "radiograph_tokens": cast(Tensor, raw["radiograph_tokens"]),
            "text_tokens": cast(Tensor, raw["text_tokens"]),
            "label": cast(Tensor, raw["label"]).long(),
            "photo_present": torch.as_tensor(raw["photo_present"]).float(),
            "radiograph_present": torch.as_tensor(raw["radiograph_present"]).float(),
        }
