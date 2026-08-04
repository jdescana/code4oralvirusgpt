from __future__ import annotations

import argparse
from pathlib import Path

from oral_virus_gpt.utils.logging_setup import get_logger

logger = get_logger("runners.export_onnx")


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser("export-onnx", help="Export the no-MC inference graph as ONNX.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--opset", type=int, default=17)
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    logger.info("export-onnx scaffolded; opset=%d, target=%s", args.opset, args.output)
    return 0
