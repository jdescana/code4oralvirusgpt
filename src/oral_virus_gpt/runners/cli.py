from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from oral_virus_gpt import __version__
from oral_virus_gpt.runners import calibrate as calibrate_cmd
from oral_virus_gpt.runners import evaluate as evaluate_cmd
from oral_virus_gpt.runners import export_onnx as export_onnx_cmd
from oral_virus_gpt.runners import infer as infer_cmd
from oral_virus_gpt.runners import train as train_cmd
from oral_virus_gpt.runners.figures import cli as figures_cli
from oral_virus_gpt.utils.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oral-virus-gpt")
    parser.add_argument("--version", action="version", version=f"oral-virus-gpt {__version__}")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    train_cmd.register(sub)
    evaluate_cmd.register(sub)
    calibrate_cmd.register(sub)
    infer_cmd.register(sub)
    export_onnx_cmd.register(sub)
    figures_cli.register(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=args.log_level)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("no handler bound for command")
    return int(handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
