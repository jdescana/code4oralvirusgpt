from __future__ import annotations

import argparse
from pathlib import Path

from oral_virus_gpt.utils.logging_setup import get_logger

logger = get_logger("runners.calibrate")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "calibrate", help="Phase 3: fit temperature and conformal thresholds."
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--validation-cache", required=True, type=Path)
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    logger.info(
        "calibrate stage scaffolded; expects pre-computed mean-logit cache at %s",
        args.validation_cache,
    )
    return 0
