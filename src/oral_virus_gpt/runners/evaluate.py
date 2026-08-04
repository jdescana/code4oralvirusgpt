from __future__ import annotations

import argparse
from pathlib import Path

from oral_virus_gpt.utils.logging_setup import get_logger

logger = get_logger("runners.evaluate")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("evaluate", help="Run the full evaluation pipeline for a config.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("results"))
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    logger.info("evaluate stage scaffolded; binds to eval.runner.run_eval_loop")
    args.output.mkdir(parents=True, exist_ok=True)
    return 0
