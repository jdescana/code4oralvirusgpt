#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
    uq_ablation_base
    uq_ablation_mc
    uq_ablation_ts
    uq_ablation_cp
    uq_ablation_mc_ts
    uq_ablation_mc_cp
    uq_ablation_ts_cp
    uq_ablation_full
    crossdataset_dentex
    crossdataset_cairo
    sensitivity_T
    sensitivity_temperature
    sensitivity_raps
    baseline_oralgpt_omni
    baseline_internvl25
    baseline_swin_b
)

for cfg in "${CONFIGS[@]}"; do
    bash scripts/launch_eval.sh "conf/experiment/${cfg}.yaml" \
        "checkpoints/phase2/phase2.pt" \
        "results/${cfg}"
done
