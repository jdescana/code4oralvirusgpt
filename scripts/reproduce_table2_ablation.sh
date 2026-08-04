#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
    ablation_token
    ablation_semantic
    ablation_gated
    ablation_uq_gating
    ablation_concat
    ablation_weighted
    ablation_no_adapter
    ablation_eqcompute
)

for cfg in "${CONFIGS[@]}"; do
    bash scripts/launch_eval.sh "conf/experiment/${cfg}.yaml" \
        "checkpoints/phase2/phase2.pt" \
        "results/${cfg}"
done
