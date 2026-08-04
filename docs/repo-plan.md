# Repository Plan — OralVirusGPT

This plan locks in the repository layout, names every concrete module
file, pins the runtime stack, and enumerates the test grid that Turn 2
will scaffold and Turn 3 / Turn 4 will fill in. All files paths are
relative to the repo root `oral_virus_gpt/`.

## 1. Directory tree (final)

```
oral_virus_gpt/
├── README.md                                # Turn 6
├── LICENSE                                  # MIT
├── pyproject.toml                           # Turn 2
├── requirements.txt                         # Turn 2
├── environment.yml                          # Turn 2
├── Dockerfile                               # Turn 6
├── .gitignore                               # Turn 2
├── .dockerignore                            # Turn 2
├── .pre-commit-config.yaml                  # Turn 2
├── .python-version                          # 3.11
├── .github/workflows/ci.yml                 # ruff + mypy + pytest on _smoke
├── conf/                                    # Hydra-compatible YAML
│   ├── model/
│   │   ├── hgcf.yaml                        # default fusion stack
│   │   ├── encoders.yaml                    # photo / radiograph / text
│   │   ├── lora.yaml                        # rank, alpha, target modules
│   │   └── backbone.yaml                    # InternVL2.5-8B revision pin
│   ├── data/
│   │   ├── code.yaml
│   │   ├── dentex.yaml
│   │   ├── cairo.yaml
│   │   ├── tufts.yaml
│   │   ├── caries.yaml
│   │   ├── mmoral.yaml
│   │   └── modid.yaml
│   ├── optim/
│   │   ├── adamw.yaml
│   │   ├── cosine_warmup_500.yaml
│   │   └── deepspeed_zero2.yaml
│   ├── uq/
│   │   ├── mc_dropout.yaml                  # T = 20, p = 0.1
│   │   ├── temperature.yaml
│   │   ├── raps.yaml                        # alpha = 0.05, lambda = 0.01
│   │   └── risk_tier.yaml                   # theta_L, theta_H, k_med
│   └── experiment/
│       ├── main.yaml                        # Table 1 primary
│       ├── ablation_token.yaml
│       ├── ablation_semantic.yaml
│       ├── ablation_gated.yaml
│       ├── ablation_uq_gating.yaml
│       ├── ablation_concat.yaml
│       ├── ablation_weighted.yaml
│       ├── ablation_no_adapter.yaml
│       ├── ablation_eqcompute.yaml
│       ├── uq_ablation_base.yaml
│       ├── uq_ablation_mc.yaml
│       ├── uq_ablation_ts.yaml
│       ├── uq_ablation_cp.yaml
│       ├── uq_ablation_mc_ts.yaml
│       ├── uq_ablation_mc_cp.yaml
│       ├── uq_ablation_ts_cp.yaml
│       ├── uq_ablation_full.yaml
│       ├── crossdataset_dentex.yaml
│       ├── crossdataset_cairo.yaml
│       ├── sensitivity_T.yaml               # Supp. Table S5
│       ├── sensitivity_temperature.yaml     # Supp. Table S5
│       ├── sensitivity_raps.yaml            # Supp. Table S5
│       ├── baseline_oralgpt_omni.yaml       # Supp. Table S8
│       ├── baseline_internvl25.yaml         # Supp. Table S8
│       ├── baseline_swin_b.yaml             # Supp. Table S8
│       └── _smoke.yaml                      # 2-step pytest only
├── src/oral_virus_gpt/
│   ├── __init__.py                          # exports __version__
│   ├── data/
│   │   ├── __init__.py
│   │   ├── code_loader.py
│   │   ├── dentex_loader.py
│   │   ├── cairo_loader.py
│   │   ├── tufts_loader.py
│   │   ├── caries_loader.py
│   │   ├── mmoral_loader.py
│   │   ├── modid_loader.py
│   │   ├── tile_window.py                   # 448x448 dynamic tiling
│   │   ├── pixel_unshuffle.py               # 4x token reduction
│   │   ├── transforms.py                    # photo / radiograph stat sheets
│   │   ├── manifest.py                      # SHA-256 manifests
│   │   └── splits.py                        # patient_id-stratified splitter
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── photo.py                         # frozen InternViT-300M
│   │   ├── radiograph.py                    # Conv adapter + frozen ViT (Eq. 2)
│   │   └── text.py                          # InternLM2 tokeniser + LoRA r=16, a=32
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── token_xattn.py                   # Eq. (3)-(4)
│   │   ├── semantic_xattn.py                # Eq. (5)
│   │   ├── gated_fusion.py                  # Eq. (6)-(7)
│   │   ├── hgcf.py                          # full HGCF assembly
│   │   ├── missing_modality.py              # null-embedding policy
│   │   └── alternatives.py                  # concat / weighted-avg / stacked-xformer for ablations
│   ├── uq/
│   │   ├── __init__.py
│   │   ├── mc_dropout.py                    # Eq. (8)-(9), T=20, p=0.1
│   │   ├── temperature.py                   # Eq. (10), tau via NLL on 15% holdout
│   │   ├── raps.py                          # Eq. (11), alpha=0.05, lambda=0.01
│   │   ├── risk_tier.py                     # Eq. (12)
│   │   ├── severity.py                      # Eq. (13)
│   │   └── pipeline.py                      # MC -> TS -> RAPS chain
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── ce_smoothed.py                   # epsilon = 0.1
│   │   ├── ece_loss.py                      # soft-bin ECE, lambda_cal = 0.1
│   │   ├── lora_l2.py                       # lambda_reg = 1e-4
│   │   └── joint.py                         # Eq. (1)
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── classification.py                # Acc, F1, Sens, Spec, AUC
│   │   ├── calibration.py                   # ECE 15-bin, Brier, reliability
│   │   ├── conformal.py                     # marginal + conditional + set size
│   │   ├── triage.py                        # tier acc / sens / NPV / NNR
│   │   ├── agreement.py                     # Cohen's kappa
│   │   ├── bootstrap.py                     # 1000-iter percentile CI
│   │   └── significance.py                  # paired bootstrap + Holm-Bonferroni + BH-FDR
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── stage_a_adapter.py               # Phase 1: 5 epochs, contrastive
│   │   ├── stage_b_hgcf.py                  # Phase 2: 10 epochs, joint
│   │   ├── stage_c_calibrate.py             # Phase 3: tau + CP thresholds
│   │   ├── ddp.py                           # DeepSpeed ZeRO-2 wrapper
│   │   ├── checkpoint.py                    # tmp + os.replace
│   │   ├── seed.py                          # set_seed -> ckpt -> restore
│   │   └── ema.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── runner.py                        # base eval loop
│   │   ├── ablation.py                      # T2 row driver
│   │   ├── crossdataset.py                  # T-S2 zero-shot transfer
│   │   ├── subgroup.py                      # T-S3
│   │   ├── error_pattern.py                 # T-S9
│   │   ├── compute_cost.py                  # T-S4 FLOPs / latency / memory
│   │   ├── sensitivity.py                   # T-S5 sweeps
│   │   ├── seed_sweep.py                    # T-S6
│   │   ├── reproduce_baselines.py           # T-S8
│   │   ├── tiered_metrics.py                # T3
│   │   ├── perclass.py                      # T-S1
│   │   ├── uq_ablation.py                   # T4
│   │   └── clinical_utility.py              # T-S10
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_setup.py                 # stdlib logging only
│   │   ├── stats.py
│   │   └── text_template.py                 # clinical-text -> prompt
│   └── runners/
│       ├── __init__.py
│       ├── train.py                         # phase-aware trainer
│       ├── evaluate.py
│       ├── infer.py
│       ├── calibrate.py
│       ├── export_onnx.py
│       └── figures/
│           ├── __init__.py
│           ├── cli.py                       # entrypoint dispatcher
│           ├── hgcf_uq_panels.py            # Fig. 2
│           ├── case_comparison.py           # Fig. 3
│           ├── calibration_panels.py        # Fig. 4
│           ├── risk_tier_grid.py            # Fig. 5
│           └── perclass_bar.py              # Fig. S1
├── scripts/
│   ├── prepare_code.sh
│   ├── prepare_dentex.sh
│   ├── prepare_cairo.sh
│   ├── prepare_tufts.sh
│   ├── prepare_caries.sh
│   ├── prepare_mmoral.sh
│   ├── prepare_modid.sh
│   ├── launch_phase1.sh                     # torchrun / DeepSpeed
│   ├── launch_phase2.sh
│   ├── launch_phase3_calibrate.sh
│   ├── launch_eval.sh
│   ├── reproduce_table1.sh                  # one-shot for Table 1
│   ├── reproduce_table2_ablation.sh
│   └── reproduce_supplementary.sh
├── tests/
│   ├── __init__.py
│   ├── conftest.py                          # tmp ckpt dir, dummy tokenizer
│   ├── data/
│   │   ├── test_loaders.py
│   │   ├── test_tile_window.py
│   │   ├── test_pixel_unshuffle.py
│   │   ├── test_splits.py
│   │   └── test_manifest.py
│   ├── encoders/
│   │   ├── test_photo.py
│   │   ├── test_radiograph_adapter.py
│   │   └── test_text.py
│   ├── fusion/
│   │   ├── test_token_xattn.py
│   │   ├── test_semantic_xattn.py
│   │   ├── test_gated_fusion.py
│   │   ├── test_missing_modality.py
│   │   └── test_hgcf.py
│   ├── uq/
│   │   ├── test_mc_dropout.py
│   │   ├── test_temperature.py
│   │   ├── test_raps.py
│   │   ├── test_risk_tier.py
│   │   ├── test_severity.py
│   │   └── test_pipeline.py
│   ├── losses/
│   │   ├── test_ce_smoothed.py
│   │   ├── test_ece_loss.py
│   │   ├── test_lora_l2.py
│   │   └── test_joint.py
│   ├── metrics/
│   │   ├── test_classification.py
│   │   ├── test_calibration.py
│   │   ├── test_conformal.py
│   │   ├── test_triage.py
│   │   ├── test_agreement.py
│   │   ├── test_bootstrap.py
│   │   └── test_significance.py
│   ├── engine/
│   │   ├── test_checkpoint.py
│   │   ├── test_seed.py
│   │   └── test_ddp.py
│   ├── eval/
│   │   └── test_runner.py
│   └── test_training_smoke.py               # 2 steps on _smoke.yaml
├── docs/
│   ├── project-context.md                   # Turn 0 (done)
│   ├── implementation-map.md                # Turn 1 (this turn)
│   ├── repo-plan.md                         # Turn 1 (this turn)
│   └── deviations.md                        # empty stub created in Turn 2
└── benchmarks/
    └── manifests/                           # SHA-256 dataset manifests
```

## 2. Module-level responsibilities

### `data/`

- Owns dataset ingestion, splitting, transforms, and manifest hashing.
- One `*_loader.py` per dataset; each exposes `Dataset` + a `from_config`
  factory that consumes `conf/data/<name>.yaml`.
- Splits respect patient-level grouping (CODe, Cairo) — no mixing of
  the same patient across train/val/test (Methods §Datasets ¶1, ¶3).
- `transforms.py` keeps photo and radiograph statistic sheets separate
  per Methods §Encoders ¶3 (radiographs are not fed through
  photo-trained mean / std).
- `manifest.py` writes / verifies a SHA-256 manifest to
  `benchmarks/manifests/<dataset>.json`; the README exposes the hash.

### `encoders/`

- `photo.py` wraps the frozen `InternViT-300M` (no gradient flow).
- `radiograph.py` wraps the same frozen ViT preceded by the
  Conv-BN-Conv residual adapter (Eq. 2). The adapter is the only
  trainable component in Phase 1.
- `text.py` wraps the InternLM2 tokeniser and applies PEFT LoRA
  (rank 16, α 32) onto the LM's attention projections.

### `fusion/`

- `token_xattn.py` houses both Photo↔Text (Eq. 3) and Radiograph↔Text
  (Eq. 4) modules; the latter shares K, V projections with the former
  but introduces its own Q.
- `semantic_xattn.py` introduces the K = 16 learnable concept slots and
  emits modality-specific semantic tensors S_V, S_R, S_T (Eq. 5).
- `gated_fusion.py` implements the sigmoid-gated mixture (Eq. 6) and
  the gated residual back into the LM hidden state (Eq. 7).
- `missing_modality.py` swaps a missing input for a learnable null
  embedding and zeros that modality's gate slot (Methods §HGCF
  ¶ Missing modality).
- `alternatives.py` houses the three flat-fusion ablations
  (concatenation, weighted average, stacked-transformer with matched
  parameter budget).
- `hgcf.py` is the top-level module that the LM is wrapped around.

### `uq/`

- `mc_dropout.py` flips dropout to inference-time on, runs T = 20
  passes, and emits MC-averaged probabilities + predictive entropy.
- `temperature.py` fits a single τ on the calibration split via NLL
  minimisation (LBFGS, 50 iters); persists τ in the checkpoint.
- `raps.py` implements RAPS prediction sets at α = 0.05 with
  λ_RAPS = 0.01; emits sets of variable size.
- `risk_tier.py` applies the piecewise rule of Eq. (12); thresholds
  θ_L, θ_H, k_med are loaded from the calibration artefact.
- `severity.py` applies the linear head of Eq. (13).
- `pipeline.py` chains the four pieces into a single `predict`-style
  call returning `{probs, set, entropy, tier, severity}`.

### `losses/`

- `ce_smoothed.py` — label-smoothed CE (ε = 0.1).
- `ece_loss.py` — soft-binned ECE used as a regulariser (λ_cal = 0.1).
- `lora_l2.py` — L2 over LoRA A, B matrices (λ_reg = 1e-4).
- `joint.py` — combines the above per Eq. (1).

### `metrics/`

- One file per family of metrics (classification / calibration /
  conformal / triage / agreement / bootstrap / significance).
- All return `numpy` scalars or arrays — no torch tensors leak out.
- `bootstrap.py` is the single source of truth for resampling (used by
  significance and CI computation alike).

### `engine/`

- Three trainers — `stage_a_adapter`, `stage_b_hgcf`, `stage_c_calibrate` —
  share the same DDP wrapper and checkpoint utilities but differ in
  what they touch:
  - Phase 1 trains only the radiograph adapter on photo↔text
    contrastive pairs from CODe.
  - Phase 2 trains HGCF + LoRA + adapter under Eq. (1) with cosine
    decay and a 500-step linear warmup.
  - Phase 3 fits τ and the RAPS / tier thresholds; no backprop.
- `checkpoint.py` writes via tmp file + `os.replace`. Each checkpoint
  carries `seed`, `phase`, `step`, `model_state`, `optim_state`,
  `lr_scheduler_state`, `tau`, `cp_thresholds`, `tier_thresholds`.
- `seed.py` exposes one `set_seed(seed: int)` helper. Resuming a run
  always re-seeds before any randomness.

### `eval/`

- One file per reported table / figure that needs an evaluation
  pipeline (see implementation-map §E–§F).
- `runner.py` is the shared loop; the per-table modules wrap it with
  the correct selection of metrics, dataset slice, and post-hoc
  significance call.

### `runners/`

- Thin CLI entrypoints — argparse + Hydra → call into engine / eval.
- `runners/figures/` is a sub-package that re-uses saved tensors from
  evaluation runs to rebuild any reported figure without touching the
  GPU.

## 3. Pinned dependencies (initial proposal — Turn 2 will lock)

`pyproject.toml` will list these under `[project.dependencies]`. Pins
are upper-bounded with `~=` where stability is well known and exact
where the manuscript depends on a specific minor.

| Package          | Pin              | Reason                                                               |
|------------------|------------------|----------------------------------------------------------------------|
| `python`         | `>=3.11,<3.13`   | typed `dataclass(slots=True)` and `match` statements used in losses. |
| `torch`          | `==2.1.2`        | Supp. Table S8 baseline reproduction column.                         |
| `torchvision`    | `==0.16.2`       | Pinned to torch 2.1.2.                                               |
| `transformers`   | `>=4.40,<4.42`   | Earliest version that ships InternVL2.5 config classes.              |
| `peft`           | `~=0.10`         | LoRA r/α API stable; no breaking change expected.                    |
| `accelerate`     | `~=0.30`         | DeepSpeed bridge.                                                    |
| `deepspeed`      | `~=0.14`         | ZeRO-2 + bf16 stable on A100.                                        |
| `bitsandbytes`   | `~=0.43`         | optional 8-bit optimiser path; off by default.                       |
| `numpy`          | `<2.0`           | torch 2.1 not built against numpy 2.x ABI.                           |
| `scipy`          | `~=1.13`         | bootstrap, quantile regression for RAPS.                             |
| `scikit-learn`   | `~=1.5`          | metrics utilities, ROC.                                              |
| `pandas`         | `~=2.2`          | result table emission.                                               |
| `pyyaml`         | `~=6.0`          | config loading.                                                      |
| `omegaconf`      | `~=2.3`          | Hydra-compatible config merging.                                     |
| `hydra-core`     | `~=1.3`          | CLI overrides.                                                       |
| `pillow`         | `~=10.3`         | image IO.                                                            |
| `pydicom`        | `~=2.4`          | dental X-ray DICOM ingestion (DENTEX, Tufts).                        |
| `opencv-python`  | `~=4.10`         | photo preprocessing.                                                 |
| `tifffile`       | `~=2024.5.10`    | MODID 16-band multispectral.                                         |
| `matplotlib`     | `~=3.9`          | figures.                                                             |
| `seaborn`        | `~=0.13`         | calibration / risk-tier panels.                                      |
| `tqdm`           | `~=4.66`         | progress.                                                            |
| `rich`           | `~=13.7`         | logging-formatter.                                                   |
| `httpx`          | `~=0.27`         | dataset fetchers (HuggingFace, Zenodo).                              |
| `huggingface-hub`| `~=0.23`         | CODe download via DOI 10.57967/hf/6421.                              |
| `safetensors`    | `~=0.4`          | checkpoint serialisation.                                            |
| `onnx`           | `~=1.16`         | export-onnx CLI.                                                     |

`requirements.txt` will mirror the runtime subset; `environment.yml`
will mirror it via conda channels (`pytorch::pytorch=2.1.2`,
`nvidia::cudatoolkit=12.1`, `conda-forge::deepspeed`).

Dev-only stack (`[project.optional-dependencies.dev]`):

| Package              | Pin             | Use |
|----------------------|-----------------|-----|
| `pytest`             | `~=8.2`         | test runner. |
| `pytest-xdist`       | `~=3.6`         | parallel tests. |
| `pytest-cov`         | `~=5.0`         | coverage report. |
| `pytest-mock`        | `~=3.14`        | mocks for the DeepSpeed wrapper. |
| `mypy`               | `~=1.10`        | `--strict` on `src/oral_virus_gpt/*`. |
| `ruff`               | `~=0.4`         | lint + import sort. |
| `black`              | `~=24.4`        | format (CI checks `--check`). |
| `isort`              | `~=5.13`        | redundant with ruff but kept for editor parity. |
| `pre-commit`         | `~=3.7`         | pre-commit hooks. |
| `types-pyyaml`       | `~=6.0`         | mypy stubs. |
| `types-pillow`       | `~=10.2`        | mypy stubs. |

## 4. Test grid (target coverage)

Smoke-tier (must pass on every CI run, no GPU):

| File                                              | Asserts                                                                       |
|---------------------------------------------------|-------------------------------------------------------------------------------|
| `tests/data/test_tile_window.py`                  | tile count = ⌈H/448⌉⌈W/448⌉ × 256/4 for randomised H, W.                     |
| `tests/data/test_pixel_unshuffle.py`              | output channel count quadruples; spatial halves both axes.                    |
| `tests/data/test_splits.py`                       | no patient_id leaks across train/val/test; ratio 70/10/20 ± 1 %.              |
| `tests/data/test_manifest.py`                     | manifest hash invariant under reordering, sensitive to byte change.            |
| `tests/encoders/test_radiograph_adapter.py`       | adapter forward shape = input shape; gradient flows only through the adapter. |
| `tests/encoders/test_photo.py`                    | encoder weights remain frozen across one fake training step.                  |
| `tests/encoders/test_text.py`                     | LoRA exposes `r=16, α=32`; only LoRA matrices are trainable.                  |
| `tests/fusion/test_token_xattn.py`                | softmax row-sums to 1; output shape matches V_T.                              |
| `tests/fusion/test_semantic_xattn.py`             | output `[K, d]` per modality; K = 16 enforced.                                |
| `tests/fusion/test_gated_fusion.py`               | gates ∈ [0, 1]; mass roughly equal at zero-init weights.                      |
| `tests/fusion/test_missing_modality.py`           | dropping a modality zeros that gate and replaces input with null embedding.   |
| `tests/fusion/test_hgcf.py`                       | full HGCF forward equivalence under deterministic seed.                       |
| `tests/uq/test_mc_dropout.py`                     | T = 20 distinct outputs; entropy ≥ 0; entropy(certain prediction) ≈ 0.         |
| `tests/uq/test_temperature.py`                    | NLL strictly decreases over τ optimisation iterations.                        |
| `tests/uq/test_raps.py`                           | empirical coverage on synthetic data ≥ 1 - α at α = 0.05.                     |
| `tests/uq/test_risk_tier.py`                      | piecewise rule matches Eq. (12) on hand-set entropy / set-size cases.          |
| `tests/uq/test_severity.py`                       | output ∈ ℝ; deterministic given inputs.                                       |
| `tests/uq/test_pipeline.py`                       | full chain returns dict keys {probs, set, entropy, tier, severity}.           |
| `tests/losses/test_ce_smoothed.py`                | ε = 0 collapses to standard CE; ε = 0.1 raises loss on certain prediction.    |
| `tests/losses/test_ece_loss.py`                   | perfectly calibrated synthetic distribution → loss = 0.                       |
| `tests/losses/test_lora_l2.py`                    | gradient touches only LoRA A, B params.                                       |
| `tests/losses/test_joint.py`                      | joint loss reduces correctly to component losses when other λ = 0.            |
| `tests/metrics/test_classification.py`            | accuracy, F1, AUC match `sklearn` on a hand-coded confusion matrix.           |
| `tests/metrics/test_calibration.py`               | ECE on uniform-in-bin perfect data ≈ 0.                                       |
| `tests/metrics/test_conformal.py`                 | coverage estimator equals 1.0 on dataset where every set covers the label.    |
| `tests/metrics/test_triage.py`                    | NNR formula = referrals / (referrals - errors_avoided) on synthetic counts.   |
| `tests/metrics/test_agreement.py`                 | Cohen κ matches `sklearn` reference.                                          |
| `tests/metrics/test_bootstrap.py`                 | 1 000-iter CI is monotonic in confidence level.                               |
| `tests/metrics/test_significance.py`              | Holm-Bonferroni and BH-FDR step functions match published worked example.    |
| `tests/engine/test_checkpoint.py`                 | atomic write — interrupted write does not corrupt prior file.                 |
| `tests/engine/test_seed.py`                       | re-seeding produces identical fake-data forward output.                       |
| `tests/engine/test_ddp.py`                        | DeepSpeed wrapper instantiates with ZeRO-2 config; mocked.                    |
| `tests/eval/test_runner.py`                       | runner emits a CSV with required columns for Table 1.                         |
| `tests/test_training_smoke.py`                    | 2 steps on `_smoke.yaml`; loss decreases monotonically over the two steps.    |

GPU-tier (skipped on CI; runs nightly on the lab cluster):

| Script                                            | Asserts |
|---------------------------------------------------|---------|
| `scripts/launch_phase1.sh --smoke`                | Phase 1 trainer runs one epoch, checkpoint roundtrips. |
| `scripts/launch_phase2.sh --smoke`                | Phase 2 with HGCF + LoRA + DeepSpeed ZeRO-2 runs one step at bf16. |
| `scripts/launch_phase3_calibrate.sh --smoke`      | τ + RAPS + tier thresholds are written to disk; Phase 3 wall-clock ≤ 5 min. |

CI rules:

- `ruff check . && black --check . && isort --check-only .` runs on
  every PR.
- `mypy --strict src/oral_virus_gpt` runs on every PR.
- `pytest -q -m "not gpu"` runs on every PR.
- `pytest -q -m gpu` runs nightly on the lab self-hosted runner.

## 5. Dependency on Turn-2 work

Turn 2 will create:

1. The empty repo layout (every directory + `__init__.py` files).
2. `pyproject.toml`, `requirements.txt`, `environment.yml`,
   `Dockerfile`, `.gitignore`, `.dockerignore`,
   `.pre-commit-config.yaml`, `.github/workflows/ci.yml`.
3. Empty Python files containing only typed stubs (no logic).
4. `docs/deviations.md` as an empty stub.
5. The `_smoke.yaml` config so that `pytest -q -m "not gpu"` returns
   "passed" on a fresh checkout.

Turn 3 fills `data/`, `encoders/`, `fusion/`, `uq/`, `losses/`,
`metrics/`. Turn 4 fills `engine/`, `eval/`, `runners/` and adds the
two-step smoke trainer test.

---

**End of Turn 1.** Awaiting user approval of the implementation map and
this repo plan before scaffolding starts in Turn 2.
