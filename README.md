# OralVirusGPT

Reference implementation for the OralVirusGPT manuscript: a multimodal
large language model for oral disease diagnosis built on InternVL2.5-8B,
combining a Hierarchical Gated Cross-Modal Fusion (HGCF) head with a
multi-layer uncertainty quantification stack (MC Dropout + temperature
scaling + RAPS conformal prediction) and a three-tier risk
stratification policy.

## Overview

The model accepts intraoral RGB photographs, dental panoramic
radiographs, and structured clinical text. HGCF aligns those streams at
three levels — token, semantic, and gated decision — then chains MC
Dropout (T = 20), learnable temperature τ, and RAPS prediction sets
(α = 0.05, λ = 0.01) to surface a calibrated probability, a coverage
set, an entropy score, and a tier ∈ {low, medium, high} for every case.

The repository implements every numbered equation in the manuscript
(see `docs/implementation-map.md`), every reported main and
supplementary table, and a runner per result table or figure. The
training pipeline mirrors the manuscript's three phases (radiograph
adapter contrastive pretraining, joint HGCF + LoRA optimisation, and
post-hoc τ + RAPS calibration).

## Installation

### pip

    git clone <repo>
    cd oral_virus_gpt
    pip install -e .[dev]

### conda

    conda env create -f environment.yml
    conda activate oral_virus_gpt
    pip install -e .

### Docker

    docker build -t oral_virus_gpt:0.1.0 .
    docker run --gpus all -it oral_virus_gpt:0.1.0 bash

The Docker base image is `nvcr.io/nvidia/pytorch:24.01-py3` and pins
torch 2.1.2 with CUDA 12.1, matching the framework column reported in
the manuscript's baseline-reproduction supplementary table.

## Data

All seven datasets are public; this implementation never pulls private
or clinical data.

| Dataset             | Modalities                              | Source                                                       | License            | Notes |
|---------------------|-----------------------------------------|--------------------------------------------------------------|--------------------|-------|
| CODe                | photo + radiograph + clinical text      | HuggingFace, DOI 10.57967/hf/6421                            | CC-BY              | Primary benchmark; 70/10/20 patient-stratified split. |
| DENTEX              | panoramic radiograph                    | https://dentex.grand-challenge.org/                          | CC-BY-NC-SA 4.0    | 705/50/250 challenge split, 4 pathologies. |
| Cairo Intraoral     | intraoral photograph                    | British Dental Journal 2025 (request from authors)            | confirm with steward | 70/10/20 patient-stratified split, 3 risk classes. |
| Tufts Dental DB     | panoramic radiograph + radiologist gaze | https://tdd.ece.tufts.edu/ (request access)                   | confirm with steward | Cross-evaluation only, 1 000 images. |
| Annotated Caries    | intraoral photograph                    | Zenodo annotated caries dataset                               | open               | Cross-evaluation only, 6 313 images. |
| MMOral / MMOral-Uni | radiograph + instructions               | https://github.com/isbrycee/OralGPT                           | confirm with repo  | Benchmark split = 100 imgs / 1 069 questions. |
| MODID               | multispectral (16 bands, 460–600 nm)    | Dryad / Zenodo                                                | confirm with steward | 243 images / 91 subjects, exploratory cross-modality. |

Each dataset has a preparation script under `scripts/`. Each script is
a single command that accepts the destination directory and prints the
canonical source URL plus the license, then leaves a `manifest.json`
slot for users who already have the data on disk:

    bash scripts/prepare_code.sh ./data/code
    bash scripts/prepare_dentex.sh ./data/dentex
    bash scripts/prepare_cairo.sh ./data/cairo

After download, write a `manifest.json` describing
`{name, records:[{split, photo, radiograph, label, patient_id, ...}]}`.
A SHA-256 manifest of every file lives at
`benchmarks/manifests/<dataset>.json`.

## Training

The manuscript uses a three-phase schedule on 4 × NVIDIA A100 80 GB.

| Phase | What changes                                  | Steps                                                    |
|-------|-----------------------------------------------|----------------------------------------------------------|
| 1     | Radiograph adapter contrastive pretraining    | `bash scripts/launch_phase1.sh conf/experiment/main.yaml 4` |
| 2     | HGCF + LoRA + adapter joint training (Eq. 1)  | `bash scripts/launch_phase2.sh conf/experiment/main.yaml 4` |
| 3     | Post-hoc temperature + RAPS + tier thresholds | `bash scripts/launch_phase3_calibrate.sh conf/experiment/main.yaml` |

Each phase reads the same experiment config and writes a checkpoint
under `checkpoints/<phase>/`. Phase 3 is grad-free and completes in
about 1 GPU-hour.

To launch a single training stage directly:

    python -m oral_virus_gpt.runners.cli train \
        --config conf/experiment/main.yaml \
        --phase phase2 \
        --output checkpoints/phase2

The smoke variant (`conf/experiment/_smoke.yaml`) runs two synthetic
optimisation steps on a tiny model and is the entry point exercised by
`tests/test_training_smoke.py`.

## Evaluation

One config per reported table or figure — see
`docs/implementation-map.md` for the full mapping.

| Reported artefact          | Command                                                                                  |
|----------------------------|------------------------------------------------------------------------------------------|
| Table 1 (multi-dataset)     | `bash scripts/reproduce_table1.sh`                                                       |
| Table 2 (HGCF ablation)     | `bash scripts/reproduce_table2_ablation.sh`                                              |
| Table 3 (risk stratification) | `python -m oral_virus_gpt.runners.cli evaluate --config conf/experiment/main.yaml --checkpoint checkpoints/phase2/phase2.pt --output results/main` |
| Table 4 (UQ configurations) | run all `conf/experiment/uq_ablation_*.yaml` via `scripts/reproduce_supplementary.sh`    |
| Supp. Tables S1–S10         | `bash scripts/reproduce_supplementary.sh`                                                |
| Figures 2–5, S1             | `python -m oral_virus_gpt.runners.cli figure <name> --input results/main --output results/figures` |

### Expected primary numbers (CODe, single seed)

| Metric           | Manuscript value      |
|------------------|-----------------------|
| Accuracy         | 86.3 ± 1.1 %          |
| Macro F1         | 83.8 ± 1.3 %          |
| Macro AUC        | 90.3 ± 0.5 %          |
| ECE              | 0.034                 |
| Brier            | 0.112                 |
| Marginal coverage| 95.3 %                |
| Mean set size    | 1.72                  |
| Low-risk acc     | 94.8 %                |
| High-risk referral| 14.2 %               |

Aggregations over the 10-seed protocol (42, 123, 256, 389, 512, 678,
741, 853, 927, 1024) appear in `results/main/seed_sweep.csv` once the
main config has been run.

## Compute budget

Numbers below are reported verbatim from the manuscript.

| Stage         | Wall-clock      | Hardware                                  |
|---------------|-----------------|-------------------------------------------|
| Phase 1       | 15 GPU-hours    | 4 × NVIDIA A100 (80 GB), DeepSpeed ZeRO-2 |
| Phase 2       | 100 GPU-hours   | 4 × NVIDIA A100 (80 GB), DeepSpeed ZeRO-2 |
| Phase 3       | ≈ 1 GPU-hour    | 1 × NVIDIA A100 (80 GB), no backprop      |
| **Total**     | ≈ 120 GPU-hours |                                           |
| Inference (T = 20) | 6 s / case  | 1 × A100 80 GB; 19.2 GB peak              |
| Inference (T = 1)  | 0.3 s / case| 1 × A100 80 GB                            |

Trainable parameter count: 425 M (HGCF + LoRA + radiograph adapter).
Mixed precision is bf16 throughout.

## Checkpoints

Model weights are released under the MIT license. We will publish
checkpoints to a Zenodo deposit on acceptance:

| Checkpoint        | SHA-256                                                          | Description                                |
|-------------------|------------------------------------------------------------------|--------------------------------------------|
| oralvirusgpt-main | <to be filled at release time>                                   | Final phase-2 weights + phase-3 calibration |
| oralvirusgpt-phase1 | <to be filled at release time>                                  | Phase-1 radiograph-adapter checkpoint      |

Until the release deposit is published, the bundled
`OralVirusGPT_npjDM_resubmission.zip` is the source of truth for the
review build and will be replaced by a Zenodo DOI on acceptance.

## Project layout

    oral_virus_gpt/
    ├── conf/                      Hydra-compatible config tree
    ├── src/oral_virus_gpt/
    │   ├── data/                  loaders + transforms + tile/unshuffle
    │   ├── encoders/              photo, radiograph adapter, text + LoRA
    │   ├── fusion/                HGCF, alternatives, gating, semantics
    │   ├── uq/                    MC Dropout, temperature, RAPS, tiers, severity
    │   ├── losses/                smoothed CE, soft-bin ECE, LoRA L2, joint
    │   ├── metrics/               classification, calibration, conformal, triage
    │   ├── engine/                three-phase trainers, checkpoint, EMA, DDP
    │   ├── eval/                  evaluation runner + table emitters
    │   └── runners/               argparse CLI subcommands
    ├── scripts/                   prepare / launch / reproduce shell helpers
    ├── tests/                     unit tests + smoke trainer
    ├── docs/                      project-context, implementation-map, repo-plan, deviations
    └── benchmarks/manifests/      SHA-256 manifests per dataset

Conventions:

- Type hints on every public API; mypy strict-equivalent runs in CI.
- Single `set_seed(seed)` utility; every checkpoint stores its seed.
- Atomic checkpoint writes (`os.replace`).
- DeepSpeed ZeRO-2 with bf16 in production; CPU-only smoke path for
  pytest.
- Hydra-compatible YAML in `conf/`; CLI overrides via standard
  Hydra/argparse.

## Continuous integration

GitHub Actions runs `ruff + black + isort + mypy + pytest` on every
push:

    ruff check .
    black --check .
    isort --check-only .
    mypy src/oral_virus_gpt
    pytest -q -m "not gpu and not integration and not slow"

Heavier GPU smokes (full-precision phase-2 step on a real A100) run
nightly on the lab self-hosted runner; see `.github/workflows/ci.yml`.

## Citation

    @article{oralvirusgpt2026,
        title   = {OralVirusGPT: A Trustworthy Multimodal Large Language Model with Uncertainty-Guided Risk Stratification for Comprehensive Oral Disease Diagnosis},
        journal = {npj Digital Medicine},
        year    = {2026},
        note    = {in submission}
    }

## License

MIT — see `LICENSE`. Dataset licenses follow each dataset's source and
are documented in the table above and in
`docs/project-context.md` §6.
