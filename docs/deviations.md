# Deviations from manuscript

This file tracks any place where the implementation diverges from the
manuscript. An empty entry list means perfect fidelity.

## Format

Each entry uses:

| ID | Manuscript anchor | Code anchor | Nature | Justification |

`Nature` ∈ {clarification, parameter-default, mechanism, scope}.

## Entries

| D-01 | Methods §Implementation details ¶2 | `conf/optim/adamw.yaml`, `conf/experiment/main.yaml` | parameter-default | The manuscript fixes lr, schedule, warmup, weight decay (implicit via λ_reg), and label smoothing for Phase 2 but does not name the optimiser. We default to `AdamW(betas=(0.9, 0.999), eps=1e-8)`, which is the InternVL2.5 release default; the value is recorded in `conf/optim/adamw.yaml` for explicit override. |

| D-02 | Methods §Implementation details ¶2 | `conf/experiment/main.yaml` (`phase2.batch_size_per_gpu = 16`, `compute.world_size = 4`) | parameter-default | The manuscript states "batch size 16" on 4 × A100. We interpret this as per-GPU 16 → world batch 64 with `grad_accum_steps = 1`, which matches the reported 100 GPU-hour wall-clock at the CODe split sizes. |

| D-03 | Methods §Datasets ¶3, ¶4, ¶6, ¶7 | `conf/data/cairo.yaml`, `conf/data/tufts.yaml`, `conf/data/mmoral.yaml`, `conf/data/modid.yaml` | clarification | The manuscript does not state explicit licences for Cairo Intraoral, Tufts Dental DB, MMOral, or MODID. Each dataset config carries `license: TBD`; the canonical licence text will be filled in from each dataset's source-of-record landing page when the corresponding `prepare_<dataset>.sh` script first lands the data. |

| D-04 | Code-availability statement | `runners/{evaluate,calibrate,infer,export_onnx}.py`, `runners/figures/cli.py` | scope | These five CLI subcommands expose the argparse contract and create the output directory but rely on a trained checkpoint (`checkpoints/phase2/phase2.pt`) and pre-computed validation cache (`results/validation_cache.pt`) that are produced only when `phase1`/`phase2` finish on actual GPU hardware. The end-to-end smoke path covered by `tests/test_training_smoke.py` exercises HGCF + the joint objective + the optimiser step at synthetic scale, asserting that the loss decreases over the first two steps. |

| D-05 | Methods §Modality-specific encoders ¶1, Methods §Encoders ¶5 | `encoders/photo.py`, `encoders/text.py` | mechanism | The InternViT-300M and InternLM2-7B backbones are loaded via dependency injection (`PhotoEncoder(vision_model=...)`, `TextEncoder(language_model=...)`). On a real A100 deployment, the production builder calls `transformers.AutoModel.from_pretrained("OpenGVLab/InternVL2_5-8B", revision="main", trust_remote_code=True)` once and shares its sub-modules; on CI the builder is replaced by a small synthetic stub so unit tests pass on CPU. This is a clean-room engineering choice rather than a deviation from the algorithmic specification — the per-module behaviour (frozen visual encoder, LoRA r/α on the LM, residual radiograph adapter trained from scratch) is implemented exactly as stated. |
