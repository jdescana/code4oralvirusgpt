#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-conf/experiment/main.yaml}"
NPROC="${2:-4}"

torchrun \
    --nproc-per-node="$NPROC" \
    -m oral_virus_gpt.runners.cli train \
    --config "$CONFIG" \
    --phase phase2 \
    --output checkpoints/phase2
