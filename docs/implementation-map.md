# Implementation Map — OralVirusGPT

This map binds every numbered equation, every reported table, every
figure, every ablation, and every metric in the manuscript to the file
and module that will implement it. Anything in the manuscript that is
not covered by a row below is a gap to be closed in Turn 1 before
scaffolding starts in Turn 2.

Conventions

- File paths are relative to repo root (`oral_virus_gpt/`).
- "Module" names are the dotted path under `src/oral_virus_gpt/`.
- "Test" names are the pytest file under `tests/`.
- A row marked `Σ` covers an aggregate (a whole table or figure panel)
  and lists the contributing modules in the notes column.

---

## A. Equations (1–13)

| Paper section                   | Eq.  | File path                                 | Module                                   | Test                                     | Notes |
|---------------------------------|------|-------------------------------------------|------------------------------------------|------------------------------------------|-------|
| Methods §Overview ¶3            | (1)  | `src/oral_virus_gpt/losses/joint.py`      | `losses.joint.JointObjective`            | `tests/losses/test_joint.py`             | Combines CE (label smoothing 0.1), ECE-as-loss (λ_cal = 0.1), and LoRA L2 (λ_reg = 1e-4). Conformal coverage constraint is enforced post-hoc, not in-loss. |
| Methods §Encoders ¶3            | (2)  | `src/oral_virus_gpt/encoders/radiograph.py` | `encoders.radiograph.RadiographAdapter` | `tests/encoders/test_radiograph_adapter.py` | Conv3×3 → BN → Conv3×3 + residual; ~5 M params; trained from scratch in Phase 1. |
| Methods §HGCF ¶6                | (3)  | `src/oral_virus_gpt/fusion/token_xattn.py` | `fusion.token_xattn.TokenCrossAttention` | `tests/fusion/test_token_xattn.py`       | Photo↔Text. h = 16 heads, d_k = d / h. |
| Methods §HGCF ¶6                | (4)  | `src/oral_virus_gpt/fusion/token_xattn.py` | `fusion.token_xattn.RadiographCrossAttention` | `tests/fusion/test_token_xattn.py`     | Radiograph↔Text. Shares K/V projections with (3) but has a separate Q projection — implement as a sibling class so the shared/split structure is explicit. |
| Methods §HGCF ¶7                | (5)  | `src/oral_virus_gpt/fusion/semantic_xattn.py` | `fusion.semantic_xattn.SemanticCrossAttention` | `tests/fusion/test_semantic_xattn.py` | K = 16 learnable concept slots. Produces S_V, S_R, S_T. |
| Methods §HGCF ¶8                | (6)  | `src/oral_virus_gpt/fusion/gated_fusion.py` | `fusion.gated_fusion.SigmoidGate`        | `tests/fusion/test_gated_fusion.py`      | W_g initialised to zero (Flamingo-style). |
| Methods §HGCF ¶8                | (7)  | `src/oral_virus_gpt/fusion/gated_fusion.py` | `fusion.gated_fusion.GatedResidual`      | `tests/fusion/test_gated_fusion.py`      | α_res learnable scalar, init zero, tanh-squashed. |
| Methods §UQ ¶1                  | (8)  | `src/oral_virus_gpt/uq/mc_dropout.py`     | `uq.mc_dropout.MCDropoutEnsemble`        | `tests/uq/test_mc_dropout.py`            | T = 20 stochastic forward passes; dropout p = 0.1 stays active at inference. |
| Methods §UQ ¶1                  | (9)  | `src/oral_virus_gpt/uq/mc_dropout.py`     | `uq.mc_dropout.predictive_entropy`       | `tests/uq/test_mc_dropout.py`            | H[y∣x] over MC-averaged probabilities. |
| Methods §UQ ¶2                  | (10) | `src/oral_virus_gpt/uq/temperature.py`    | `uq.temperature.TemperatureScaler`       | `tests/uq/test_temperature.py`           | Scalar τ, NLL minimisation on 15 % validation holdout; learned τ ≈ 1.32 (Supp. Table S5). |
| Methods §UQ ¶3                  | (11) | `src/oral_virus_gpt/uq/raps.py`           | `uq.raps.RAPSPredictor`                  | `tests/uq/test_raps.py`                  | α = 0.05; RAPS λ = 0.01 (Supp. Table S5); calibration set = 15 % of validation. |
| Methods §Risk ¶1                | (12) | `src/oral_virus_gpt/uq/risk_tier.py`      | `uq.risk_tier.RiskTierPolicy`            | `tests/uq/test_risk_tier.py`             | θ_L, θ_H, k_med tuned on validation; emits {low, medium, high}. |
| Methods §Risk ¶3                | (13) | `src/oral_virus_gpt/uq/severity.py`       | `uq.severity.SeverityHead`               | `tests/uq/test_severity.py`              | Linear head on [p̄; H; ∣C∣]. |

## B. Architecture beyond equations

| Paper section                   | Component                              | File path                                 | Module                                | Test                                          | Notes |
|---------------------------------|----------------------------------------|-------------------------------------------|---------------------------------------|-----------------------------------------------|-------|
| Methods §Overview ¶3            | InternViT-300M frozen photo encoder    | `src/oral_virus_gpt/encoders/photo.py`    | `encoders.photo.PhotoEncoder`         | `tests/encoders/test_photo.py`                | 448 × 448 dynamic tiling + pixel-unshuffle (4×). |
| Methods §Encoders ¶1            | Dynamic-resolution tiling              | `src/oral_virus_gpt/data/tile_window.py`  | `data.tile_window.dynamic_tile`       | `tests/data/test_tile_window.py`              | Computes N from ⌈H/448⌉ × ⌈W/448⌉ × 256/4. |
| Methods §Encoders ¶1            | Pixel-unshuffle (visual-token reduction)| `src/oral_virus_gpt/data/pixel_unshuffle.py` | `data.pixel_unshuffle.unshuffle`     | `tests/data/test_pixel_unshuffle.py`          | Quarter-tokenisation; preserves channels. |
| Methods §Encoders ¶5            | Text encoder + LoRA                    | `src/oral_virus_gpt/encoders/text.py`     | `encoders.text.TextEncoder`           | `tests/encoders/test_text.py`                 | InternLM2 tokeniser + LoRA r = 16, α = 32. |
| Methods §HGCF (full)            | Top-level HGCF assembly                | `src/oral_virus_gpt/fusion/hgcf.py`       | `fusion.hgcf.HGCF`                    | `tests/fusion/test_hgcf.py`                   | Stacks Eq. (3)-(7); emits H_fused. |
| Methods §HGCF ¶ Missing modality | Null-embedding handling                | `src/oral_virus_gpt/fusion/missing_modality.py` | `fusion.missing_modality.NullEmbedding` | `tests/fusion/test_missing_modality.py`      | Per-modality learnable null token; sets gate value to zero on absence. |
| Methods §UQ                     | UQ pipeline orchestrator               | `src/oral_virus_gpt/uq/pipeline.py`       | `uq.pipeline.UQPipeline`              | `tests/uq/test_pipeline.py`                   | Wires MCDropout → Temperature → RAPS in order. |
| Methods §Risk ¶3                | Severity scalar                        | `src/oral_virus_gpt/uq/severity.py`       | `uq.severity.SeverityHead`            | `tests/uq/test_severity.py`                   | Continuous score companion to discrete tier. |

## C. Loss components (Eq. 1 expansion)

| Component         | File path                                 | Module                              | Notes |
|-------------------|-------------------------------------------|-------------------------------------|-------|
| Smoothed CE       | `src/oral_virus_gpt/losses/ce_smoothed.py`| `losses.ce_smoothed.SmoothedCE`     | ε = 0.1. |
| ECE-as-loss       | `src/oral_virus_gpt/losses/ece_loss.py`   | `losses.ece_loss.SoftBinECE`        | Soft-binned, 15 bins, λ_cal = 0.1. |
| LoRA L2           | `src/oral_virus_gpt/losses/lora_l2.py`    | `losses.lora_l2.LoraL2`             | λ_reg = 1e-4 over LoRA A/B matrices. |

## D. Metrics

| Paper section            | Metric                                    | File path                                 | Module                                | Notes |
|--------------------------|-------------------------------------------|-------------------------------------------|---------------------------------------|-------|
| Methods §Metrics ¶1      | Accuracy                                  | `src/oral_virus_gpt/metrics/classification.py` | `metrics.classification.accuracy`   | Top-1 hard label. |
| Methods §Metrics ¶1      | Macro F1                                  | `src/oral_virus_gpt/metrics/classification.py` | `metrics.classification.macro_f1`   | Class-imbalance robust averaging. |
| Methods §Metrics ¶1      | Sensitivity                               | `src/oral_virus_gpt/metrics/classification.py` | `metrics.classification.sensitivity`| Per-class TPR. |
| Methods §Metrics ¶1      | Specificity                               | `src/oral_virus_gpt/metrics/classification.py` | `metrics.classification.specificity`| Per-class TNR. |
| Methods §Metrics ¶1      | AUC                                       | `src/oral_virus_gpt/metrics/classification.py` | `metrics.classification.auc`        | One-vs-rest macro AUC. |
| Methods §Metrics ¶1      | Bootstrap 95 % CI                         | `src/oral_virus_gpt/metrics/bootstrap.py` | `metrics.bootstrap.bootstrap_ci`      | n = 1 000 iterations. |
| Methods §Metrics ¶2      | Cohen's κ                                 | `src/oral_virus_gpt/metrics/agreement.py` | `metrics.agreement.cohens_kappa`      | Inter-rater agreement vs. expert labels. |
| Methods §Metrics ¶2      | Expected Calibration Error (ECE)          | `src/oral_virus_gpt/metrics/calibration.py` | `metrics.calibration.ece`           | 15 bin widths. |
| Methods §Metrics ¶2      | Brier score                               | `src/oral_virus_gpt/metrics/calibration.py` | `metrics.calibration.brier`         | — |
| Methods §Metrics ¶2      | Reliability diagram                       | `src/oral_virus_gpt/metrics/calibration.py` | `metrics.calibration.reliability`   | Used in Fig. 4(a). |
| Methods §Metrics ¶2      | Marginal coverage                         | `src/oral_virus_gpt/metrics/conformal.py` | `metrics.conformal.marginal_coverage` | RAPS sets at α = 0.05. |
| Methods §Metrics ¶2      | Conditional coverage                      | `src/oral_virus_gpt/metrics/conformal.py` | `metrics.conformal.conditional_coverage` | Per disease class. |
| Methods §Metrics ¶2      | Mean prediction-set size                  | `src/oral_virus_gpt/metrics/conformal.py` | `metrics.conformal.mean_set_size`     | — |
| Methods §Metrics ¶3      | Tier accuracy / sensitivity / NPV         | `src/oral_virus_gpt/metrics/triage.py`    | `metrics.triage.tier_metrics`         | Reported in Table 3. |
| Methods §Metrics ¶3      | Number Needed to Refer (NNR)              | `src/oral_virus_gpt/metrics/triage.py`    | `metrics.triage.nnr`                  | Reported in Supp. Table S10. |
| Methods §Metrics ¶3      | Paired bootstrap + Holm–Bonferroni        | `src/oral_virus_gpt/metrics/significance.py` | `metrics.significance.holm_bonferroni` | 30 primary comparisons. |
| Methods §Metrics ¶3      | Benjamini–Hochberg FDR                    | `src/oral_virus_gpt/metrics/significance.py` | `metrics.significance.bh_fdr`         | Exploratory comparisons. |

## E. Main-text tables (Tables 1–4)

| Table                                | Reproduced by                                  | Config                                     | Output artefact                          | Notes |
|--------------------------------------|------------------------------------------------|--------------------------------------------|-------------------------------------------|-------|
| **Table 1** — multi-dataset comparison (CODe / DENTEX / Cairo × {Acc, F1}, 19 baselines + ours) | `runners.evaluate.main_table` | `conf/experiment/main.yaml`            | `results/table1_main.csv`                 | OralVirusGPT row from full pipeline; baselines marked `†` from published numbers, `‡` from independent reproduction (Supp. Table S8). |
| **Table 2** — HGCF ablation (8 rows × {Acc, F1, AUC, Params}) | `runners.evaluate.ablation`   | `conf/experiment/ablation_*.yaml` ×8   | `results/table2_ablation.csv`             | One config per row; Σ — see §G below. |
| **Table 3** — risk stratification (3 tiers + full coverage × {Cov, Acc, F1, Sens, NPV, Set size}) | `runners.evaluate.tiered_metrics` | `conf/experiment/main.yaml` (+ post-hoc tier scan) | `results/table3_risk.csv`     | Tier thresholds re-loaded from Phase 3 calibration artefact. |
| **Table 4** — UQ configuration ablation (8 configs × {ECE, Brier, Coverage, Set size}) | `runners.evaluate.uq_ablation` | `conf/experiment/uq_ablation_*.yaml` ×8 | `results/table4_calibration.csv`         | Combinations of {Base, MC, TS, CP, MC+TS, MC+CP, TS+CP, Full}. |

## F. Supplementary tables (S1–S10)

| Table     | Reproduced by                                | Config                                         | Notes |
|-----------|----------------------------------------------|------------------------------------------------|-------|
| **S1** — per-class top-10 (Sens, Spec)        | `runners.evaluate.perclass`                  | `conf/experiment/main.yaml --select=top10_codecategories` | Compares OralVirusGPT vs. OralGPT-Omni vs. ConvNeXt. |
| **S2** — cross-site generalisation             | `eval.crossdataset.run`                      | `conf/experiment/crossdataset_dentex.yaml`, `crossdataset_cairo.yaml` | Zero-shot (no fine-tuning) on DENTEX and Cairo from CODe-trained checkpoint. |
| **S3** — subgroup analysis (anatomical site, risk level) | `eval.subgroup.run`                | `conf/experiment/main.yaml --subgroup=site,risk_level` | Reports max gap ≤ 3.5 pp by site, expected gradient by risk. |
| **S4** — computational cost                    | `eval.compute_cost.run`                      | `conf/experiment/main.yaml --measure_cost`     | Total params, trainable params, FLOPs/img, latency, memory. |
| **S5** — sensitivity to T, τ, λ_RAPS           | `runners.evaluate.sensitivity`              | `conf/experiment/sensitivity_T.yaml`, `sensitivity_temperature.yaml`, `sensitivity_raps.yaml` | T ∈ {5, 10, 20, 50}; τ ∈ {0.8, 1.0, 1.32, 1.8}; λ ∈ {0, 0.01, 0.1}. |
| **S6** — seed variability                      | `runners.evaluate.seed_sweep`                | `conf/experiment/main.yaml --seeds=42,123,…,1024` | 10 seeds; reports mean ± std + CV. |
| **S7** — statistical significance              | `metrics.significance.run_table`             | `conf/experiment/main.yaml --post_hoc=significance` | Paired bootstrap + Holm–Bonferroni; 30 primary comparisons. |
| **S8** — baseline reproduction                 | `runners.evaluate.reproduce_baselines`       | `conf/experiment/baseline_{oralgpt_omni,internvl25,swin_b}.yaml` | Top-3 baselines reproduced within ± 0.2 pp of published. |
| **S9** — error pattern analysis                | `eval.error_pattern.run`                     | `conf/experiment/main.yaml --error_pattern=top5` | Top-5 misclassification pairs; UQ Capture %. |
| **S10** — clinical-utility (NNR comparison)    | `eval.clinical_utility.run`                  | `conf/experiment/main.yaml --triage_strategies=all` | Compares full-UQ vs. random / entropy-only / set-size-only. |

## G. Ablation rows in Table 2 (eight configs)

| Row | Description                          | Config flag                                                          | Acc Δ (pp) | Trainable params (M) |
|-----|--------------------------------------|----------------------------------------------------------------------|------------|----------------------|
| A0  | OralVirusGPT (Full)                  | `conf/experiment/main.yaml`                                          | 0.0        | 425                  |
| A1  | − Token-level cross-attn             | `conf/experiment/ablation_token.yaml` → `model.fusion.use_token=False` | −2.2     | 395                  |
| A2  | − Semantic-level cross-attn          | `conf/experiment/ablation_semantic.yaml` → `model.fusion.use_semantic=False` | −2.8 | 398             |
| A3  | − Gated decision-level fusion        | `conf/experiment/ablation_gated.yaml` → `model.fusion.use_gating=False` | −4.5    | 388                  |
| A4  | − Uncertainty-aware gating           | `conf/experiment/ablation_uq_gating.yaml` → `model.fusion.use_uncertainty_gating=False` | −0.6 | 420   |
| A5  | HGCF → Concatenation                 | `conf/experiment/ablation_concat.yaml` → `model.fusion.style=concat` | −3.8       | 310                  |
| A6  | HGCF → Weighted average              | `conf/experiment/ablation_weighted.yaml` → `model.fusion.style=weighted` | −2.9    | 315                  |
| A7  | − Radiograph adapter                 | `conf/experiment/ablation_no_adapter.yaml` → `model.encoders.radiograph_adapter=False` | −1.2 | 420 |
| A8  | Equal-compute extra transformer layers | `conf/experiment/ablation_eqcompute.yaml` → `model.fusion.style=stacked_xformer; model.match_param_budget=True` | −3.3 | 425 |

## H. Datasets

| #  | Dataset             | Loader                                            | Splits                                  | Preprocessing notes |
|----|---------------------|---------------------------------------------------|-----------------------------------------|---------------------|
| 1  | CODe                | `data.code_loader.CODeDataset`                    | 70/10/20 stratified by patient_id        | Photo + radiograph + clinical text JSON; 120+ classes; bf16 tensors. |
| 2  | DENTEX              | `data.dentex_loader.DentexDataset`                | 705/50/250 challenge-fixed                | Panoramic X-ray only; 4 pathologies; histogram normalisation per Methods §Encoders ¶3. |
| 3  | Cairo Intraoral     | `data.cairo_loader.CairoIntraoralDataset`         | 70/10/20 stratified by patient_id        | Intraoral photo only; 3 risk classes. |
| 4  | Tufts Dental DB     | `data.tufts_loader.TuftsDataset`                  | challenge-provided                       | Panoramic X-ray + radiologist gaze; cross-eval only. |
| 5  | Annotated Caries    | `data.caries_loader.CariesDataset`                | dataset-provided                         | YOLO/Pascal/COCO annotations; cross-eval only. |
| 6  | MMOral / MMOral-Uni | `data.mmoral_loader.MMOralDataset`                | benchmark eval = 100 imgs / 1 069 Q      | MLLM benchmark format. |
| 7  | MODID               | `data.modid_loader.MODIDDataset`                  | dataset-provided                         | 16-band multispectral, 460–600 nm; cross-eval only. |

Common transforms live in `data/transforms.py` (resize-to-tile,
photometric normalisation, photo-vs-radiograph stat sheets).

## I. Figures

| Figure                                       | Reproduced by                                | Output                                       | Notes |
|----------------------------------------------|----------------------------------------------|----------------------------------------------|-------|
| **Fig. 1** — overall architecture             | static SVG, hand-drawn, in `assets/figures/`  | `assets/figures/architecture.svg`            | No code path; the SVG is shipped, not regenerated. |
| **Fig. 2** — HGCF + UQ principles             | `runners.figures.hgcf_uq_panels`             | `results/figures/fig2_hgcf_uq.pdf`           | Panels (a) attention map, (b) semantic pooling, (c) gating, (d) MC + TS + CP, (e) risk tiers. |
| **Fig. 3** — case A vs. case B                | `runners.figures.case_comparison`            | `results/figures/fig3_case_comparison.pdf`   | Pulls 2 illustrative cases by `case_id`; renders heatmap + tier label. |
| **Fig. 4** — calibration analysis (4 panels)  | `runners.figures.calibration_panels`         | `results/figures/fig4_calibration.pdf`       | (a) reliability, (b) set-size dist, (c) ECE bar, (d) accuracy–coverage curve. |
| **Fig. 5** — risk-tier examples (3 rows)      | `runners.figures.risk_tier_grid`             | `results/figures/fig5_risk_tier.pdf`         | Top low / mid medium / bottom high. Grad-CAM overlay. |
| **Fig. 6** — three-stage training + workflow  | static SVG, hand-drawn, in `assets/figures/`  | `assets/figures/study_workflow.svg`          | Mirrors Methods §Implementation details. |
| **Fig. S1** — per-class bar plot              | `runners.figures.perclass_bar`               | `results/figures/figS1_perclass.pdf`         | Top-10 most frequent CODe classes. |

## J. Training-stage map

| Stage      | Module                                            | Trainables                              | Loss                          | Schedule                              | Wall-clock |
|------------|---------------------------------------------------|------------------------------------------|-------------------------------|---------------------------------------|------------|
| Phase 1    | `engine.stage_a_adapter.AdapterContrastiveTrainer`| Radiograph adapter only                 | Image-text contrastive (NT-Xent style on CODe pairs) | 5 epochs; lr = 1e-4; bs = 32; cosine | 15 GPU-h   |
| Phase 2    | `engine.stage_b_hgcf.HGCFJointTrainer`           | HGCF + LoRA + radiograph adapter (joint)| Eq. (1)                       | 10 epochs; lr = 5e-5; bs = 16; cosine + 500-step linear warmup | 100 GPU-h  |
| Phase 3    | `engine.stage_c_calibrate.CalibrationFitter`     | τ scalar + RAPS thresholds (no grad)     | NLL (τ) + quantile regression (RAPS) | one pass over 15 % calibration holdout | ≈ 1 GPU-h  |

## K. CLI surface

| Sub-command                  | Module                              | One-liner |
|------------------------------|-------------------------------------|-----------|
| `oral-virus-gpt train`       | `runners.train`                     | Phase-aware trainer; respects `phase=1|2`. |
| `oral-virus-gpt calibrate`   | `runners.calibrate`                 | Phase 3: fits τ and CP thresholds. |
| `oral-virus-gpt evaluate`    | `runners.evaluate`                  | Single-config eval producing one of the result tables. |
| `oral-virus-gpt infer`       | `runners.infer`                     | Single-case inference with UQ + tier output. |
| `oral-virus-gpt export-onnx` | `runners.export_onnx`               | Exports the no-MC inference graph for downstream tooling. |
| `oral-virus-gpt figure`      | `runners.figures.cli`               | Regenerates one named figure from saved tensors. |

---

## Coverage check

- Equations (1)–(13): 13 / 13 mapped.
- Main-text tables (1, 2, 3, 4): 4 / 4 mapped.
- Supplementary tables (S1–S10): 10 / 10 mapped.
- Main-text figures (1–6): 6 / 6 mapped.
- Supplementary figures (S1): 1 / 1 mapped.
- Ablation rows in Table 2 (A0–A8): 9 / 9 mapped.
- Datasets (1–7): 7 / 7 mapped.
- Algorithm boxes: 0 declared in the manuscript — none required.
- Metrics: 17 metric implementations covered in §D.

No outstanding paper item is unbound. Implementation of every row above
begins in Turn 3 (science modules) and Turn 4 (training / evaluation
pipelines).
