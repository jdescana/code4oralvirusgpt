#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-conf/experiment/main.yaml}"
CKPT="${2:-checkpoints/phase2/phase2.pt}"
OUT="${3:-results}"

python -m oral_virus_gpt.runners.cli evaluate \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --output "$OUT"
