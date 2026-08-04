#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(main)

for cfg in "${CONFIGS[@]}"; do
    bash scripts/launch_eval.sh "conf/experiment/${cfg}.yaml" \
        "checkpoints/phase2/phase2.pt" \
        "results/${cfg}"
done
