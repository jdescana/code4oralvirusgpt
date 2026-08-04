from __future__ import annotations

import argparse
from pathlib import Path

from oral_virus_gpt.utils.logging_setup import get_logger

logger = get_logger("runners.infer")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("infer", help="Single-case inference with UQ output.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--photo", type=Path, default=None)
    p.add_argument("--radiograph", type=Path, default=None)
    p.add_argument("--text", type=str, default="")
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    if args.photo is None and args.radiograph is None and not args.text:
        logger.error("infer requires at least one modality (photo, radiograph, or text)")
        return 2
    logger.info("infer stage scaffolded; emits {probs, set, entropy, tier, severity}")
    return 0
