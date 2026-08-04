from __future__ import annotations

import argparse
from pathlib import Path

from oral_virus_gpt.utils.logging_setup import get_logger

logger = get_logger("runners.figures")

KNOWN_FIGURES = (
    "fig2_hgcf_uq",
    "fig3_case_comparison",
    "fig4_calibration",
    "fig5_risk_tier",
    "figS1_perclass",
)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "figure", help="Re-render one of the reported figures from saved tensors."
    )
    p.add_argument("name", choices=KNOWN_FIGURES)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    logger.info("figure %s -> %s", args.name, args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return 0
