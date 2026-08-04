#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-conf/experiment/main.yaml}"
CKPT="${2:-checkpoints/phase2/phase2.pt}"
CACHE="${3:-results/validation_cache.pt}"

python -m oral_virus_gpt.runners.cli calibrate \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --validation-cache "$CACHE"
