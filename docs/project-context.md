# Project Context — OralVirusGPT

```
project_name       : oral_virus_gpt                         [HIGH]
domain             : multimodal medical AI — oral disease   [HIGH]
                     diagnosis with calibrated uncertainty
framework          : PyTorch 2.1 + HuggingFace Transformers [HIGH]
                     + PEFT/LoRA + DeepSpeed ZeRO-2 (bf16)
venue              : npj Digital Medicine                   [HIGH]
primary_datasets   : 7 datasets (see §6)                    [HIGH]
compute_target     : 4 × NVIDIA A100 80 GB, ~120 GPU-h      [HIGH]
hparams_reference  : Methods §Implementation details        [HIGH]
                     + Supp. Tables S5, S6
supp_path          : inline in manuscript (Supplementary
                     Information section, Tables S1–S10)
extra_signals      : InternVL2.5-8B backbone; LoRA r=16, α=32;
                     MIT licence on release; 10-seed protocol;
                     Holm–Bonferroni + BH-FDR significance

NEEDS_USER_DECISION: 4 (see §10)
Ready to proceed to Turn 1? awaiting approval.
```

---

## 1. project_name — `oral_virus_gpt`            [HIGH]

The manuscript is titled *OralVirusGPT: A Trustworthy Multimodal Large
Language Model …* (Title page; first sentence of the Abstract). The
model already carries a project name; the snake_case form
`oral_virus_gpt` is the obvious match. The Python package, the repo
root, and every CLI entrypoint module use this spelling.

A footnote at the end of the Introduction (¶7) clarifies that the
"Virus" segment reflects an early oral-viral-screening origin and is not
a current scope claim — the implementation must therefore not introduce
virus-specific code paths or comments.

## 2. supp_path — inline (Supplementary Information section)            [HIGH]

No standalone supplementary file is shipped. Tables S1–S10 and Fig. S1
are bundled into the same source as the main text under the
Supplementary Information section that follows the bibliography. There
is a project resubmission archive co-located with the manuscript that
contains a code zip and the cover letter; that archive does not contain
a separate supplementary document.

Implication for Turn 1: cross-references in `implementation-map.md`
must point at `Supp. Table S<n>` rather than at any standalone file.

## 3. domain — multimodal medical AI for oral disease diagnosis with calibrated uncertainty            [HIGH]

Derived from the Abstract (sentence 1: "Oral diseases affect 3.5 billion
people worldwide"), Introduction §1 ¶1, and Methods §Overview ¶1.
Specifically:

- **Modality space**: intraoral RGB photographs, panoramic dental
  X-rays, multispectral images (MODID), and structured clinical text.
- **Tasks**: 120+-way classification on CODe; 4-way pathology
  classification on DENTEX; 3-way risk classification on Cairo
  Intraoral; cross-dataset transfer (CODe → DENTEX, CODe → Cairo)
  reported in Supp. Table S2.
- **Trust framing**: marginal-coverage prediction sets via RAPS plus a
  three-tier triage — the implementation must surface uncertainty
  alongside every prediction.
- **Regulatory framing**: retrospective evaluation on de-identified
  public datasets (Ethics statement, ¶1) — no prospective patient
  enrolment, no private data ingestion.

## 4. framework — PyTorch 2.1 + HuggingFace Transformers + PEFT (LoRA) + DeepSpeed ZeRO-2 (bf16)            [HIGH]

Direct evidence:

- Methods §Implementation details ¶3: *"all experiments on 4 × NVIDIA
  A100 (80 GB) GPUs using DeepSpeed ZeRO-2 for memory optimization …
  Mixed precision training (bf16)."*
- Supp. Table S8 (Baseline Reproduction) lists `PyTorch 2.1` as the
  Framework column for OralGPT-Omni, InternVL2.5-8B, and Swin-B.
- Methods §Modality-specific encoders cites InternViT-300M and InternLM2
  as backbone components — the canonical reference implementations live
  in `OpenGVLab/InternVL` (PyTorch + Transformers).
- Methods §Encoders ¶5 fixes LoRA `r = 16`; Methods §Implementation
  details ¶2 fixes LoRA `α = 32`. The PEFT library is the standard
  mechanism for r/α-parameterised LoRA on Transformers backbones.

Initial pin proposal (final pins recorded in Turn 1):

| Package          | Pin              | Reason                                            |
|------------------|------------------|---------------------------------------------------|
| `torch`          | `==2.1.*`        | Supp. Table S8                                    |
| `transformers`   | `==4.40.*`       | minimum version that ships `OpenGVLab/InternVL2_5-8B` config classes |
| `peft`           | `==0.10.*`       | LoRA r/α API stable                               |
| `deepspeed`      | `==0.14.*`       | ZeRO-2 + bf16 stability                            |
| `accelerate`     | `==0.30.*`       | DeepSpeed bridge                                  |

## 5. venue — npj Digital Medicine            [HIGH]

Triangulated from four signals in the manuscript bundle:

- the cover letter is addressed to *npj Digital Medicine*;
- the resubmission archive carries the venue tag `npjDM`;
- the LaTeX class is `sn-jnl.cls` and the bibliography style is
  `sn-nature.bst` — the Springer Nature journal-family templates;
- formatting cues consistent with npj DM: line numbering enabled,
  unnumbered Methods/Results/Discussion headings
  (`\setcounter{secnumdepth}{0}`), structured abstract.

The publisher's main-text word budget for npj DM is ~5 000 words; the
manuscript already routes Tables S1–S10 and the per-class breakdown to
the SI section, so no implementation-time word-budget gymnastics are
required.

## 6. primary_datasets — 7 public datasets, 96 000+ images            [HIGH on names; LICENCES partially open]

| #  | Name                | Modality                                | Size                                                                        | Licence                                | Access                                              | Paper location                                |
|----|---------------------|-----------------------------------------|-----------------------------------------------------------------------------|----------------------------------------|-----------------------------------------------------|------------------------------------------------|
| 1  | CODe                | photo + X-ray + clinical text           | 8 775 exams / 4 800 patients; 50 000+ photos; 8 056 X-rays; 120+ classes    | CC-BY                                  | HuggingFace, DOI 10.57967/hf/6421                   | Methods §Datasets ¶1; Data availability ¶1     |
| 2  | DENTEX              | panoramic X-ray                         | 1 005 images, 4 pathologies; train/val/test 705/50/250                      | CC-BY-NC-SA 4.0                        | https://dentex.grand-challenge.org/                 | Methods §Datasets ¶2                           |
| 3  | Cairo Intraoral     | intraoral photo                         | 9 201 images, 3 risk classes (4 405 / 2 314 / 2 482)                        | NEEDS_USER_DECISION (see §10)          | British Dental Journal 2025; on request from authors| Methods §Datasets ¶3; Data availability ¶2     |
| 4  | Tufts Dental DB     | panoramic X-ray + radiologist gaze      | 1 000 images                                                                | NEEDS_USER_DECISION (see §10)          | https://tdd.ece.tufts.edu/ (request)                | Methods §Datasets ¶4                           |
| 5  | Annotated Caries    | intraoral photo                         | 6 313 labelled images; YOLO / Pascal VOC / COCO formats                     | open licence (Zenodo) — exact label TBD| Zenodo                                              | Methods §Datasets ¶5                           |
| 6  | MMOral / MMOral-Uni | panoramic X-ray + 1.3 M instructions    | 20 563 images; eval split 100 imgs / 1 069 questions                         | NEEDS_USER_DECISION (see §10)          | https://github.com/isbrycee/OralGPT                 | Methods §Datasets ¶6                           |
| 7  | MODID               | multispectral (16 bands, 460–600 nm)    | 243 images, 91 subjects                                                     | NEEDS_USER_DECISION (see §10)          | Dryad / Zenodo                                      | Methods §Datasets ¶7                           |

Splits:

- **CODe** — 70/10/20 (train/val/test), stratified by patient ID
  (Methods §Datasets ¶1).
- **DENTEX** — 705 / 50 / 250 challenge-provided (Methods §Datasets ¶2).
- **Cairo Intraoral** — 70/10/20 patient-stratified
  (Methods §Datasets ¶3).
- **Tufts / Caries / MMOral / MODID** — used for cross-dataset
  evaluation only; no fine-tuning split (Supp. Table S2; Methods
  §Datasets ¶4–7).

The licences flagged `NEEDS_USER_DECISION` will be confirmed in Turn 1
by fetching each dataset's canonical landing page; the manuscript itself
does not record them verbatim.

## 7. compute_target — 4 × NVIDIA A100 80 GB, DeepSpeed ZeRO-2, bf16; ≈ 120 GPU-hours            [HIGH]

From Methods §Implementation details ¶3 and Supp. Table S4:

| Stage | Description                                  | Wall-clock         |
|-------|----------------------------------------------|--------------------|
| 1     | Radiograph-adapter contrastive training      | 15 GPU-hours       |
| 2     | HGCF + LoRA joint optimisation               | 100 GPU-hours      |
| 3     | Temperature + conformal calibration (no grad)| ≈ 1 GPU-hour       |
|       | **total**                                    | **≈ 120 GPU-hours**|

Inference cost (Supp. Table S4): 6 s/case at T = 20 MC Dropout passes;
0.3 s/case at single pass. Memory: 19.2 GB on a single A100. Trainable
parameter count (Supp. Table S4): 425 M (HGCF + LoRA + radiograph
adapter).

The Turn 6 README will publish these numbers verbatim, without
softening.

## 8. hparams_reference — Methods §Implementation details + Supp. Table S5 + Supp. Table S6            [HIGH]

| Block                         | Value                                                                | Source                                          |
|-------------------------------|----------------------------------------------------------------------|-------------------------------------------------|
| Phase 1 epochs / lr / bs      | 5 / 1 × 10⁻⁴ / 32                                                    | Methods §Impl ¶1                                |
| Phase 1 objective             | image–text contrastive on CODe split                                 | Methods §Impl ¶1                                |
| Phase 2 epochs / lr / bs      | 10 / 5 × 10⁻⁵ / 16                                                   | Methods §Impl ¶2                                |
| Phase 2 schedule              | cosine decay, 500-step linear warmup                                 | Methods §Impl ¶2                                |
| Label smoothing ε             | 0.1                                                                  | Methods §Impl ¶2                                |
| Loss weights                  | λ_cal = 0.1, λ_reg = 1 × 10⁻⁴                                        | Eq. (1); Methods §Impl ¶2                       |
| LoRA                          | rank 16, α 32                                                        | Methods §Encoders ¶5; Methods §Impl ¶2          |
| MC Dropout                    | T = 20 forward passes, dropout p = 0.1                               | Methods §UQ ¶1; Eq. (8)                         |
| Temperature                   | τ learned by NLL minimisation; learned τ ≈ 1.32                      | Eq. (10); Supp. Table S5                        |
| Conformal                     | α = 0.05; RAPS λ = 0.01; calibration split = 15 % of validation set  | Eq. (11); Methods §UQ ¶3; Methods §Risk ¶1; Supp. Table S5 |
| Risk-tier thresholds          | θ_L, θ_H, k_med tuned on validation (≥95 % low-risk acc target)       | Eq. (12); Methods §Risk                         |
| Precision                     | bf16                                                                 | Methods §Impl ¶3                                |
| Optimiser                     | NEEDS_USER_DECISION (proposed: AdamW, β = (0.9, 0.999))              | inferred — see §10                              |
| Effective batch composition   | NEEDS_USER_DECISION (proposed: per-GPU 16 × 4 GPUs = world batch 64) | inferred — see §10                              |
| Seeds                         | 42, 123, 256, 389, 512, 678, 741, 853, 927, 1024 (10 seeds)          | Main Table 1 footnote; Supp. Table S6           |
| Bootstrap                     | 1 000 iterations, 95 % CI                                            | Methods §Metrics ¶1                             |
| Significance                  | paired bootstrap + Holm–Bonferroni (30 primary), BH-FDR (exploratory)| Methods §Metrics ¶3                             |

## 9. extra_signals

- **Tokeniser** — InternLM2 sentence-piece tokeniser, used unchanged. No
  proprietary tokeniser is trained.
- **Released checkpoints** — the Reproducibility statement (¶1) commits
  to MIT-licensed code release on acceptance; no public checkpoint URL
  is yet declared.
- **Algorithm boxes** — zero formal `\begin{algorithm}` environments in
  the manuscript; the logic is fully captured by Equations (1)–(13)
  plus the three-stage training description in Methods §Implementation
  details. The Turn 1 implementation map will track Eq. (1)–(13)
  explicitly.
- **Supplementary tables and figures** — Tables S1 (per-class CODe), S2
  (cross-site generalisation), S3 (subgroup), S4 (computational cost),
  S5 (T / τ / λ_RAPS sensitivity), S6 (seed variability), S7
  (significance), S8 (baseline reproduction), S9 (error patterns), S10
  (clinical-utility / NNR), and Fig. S1 (per-class bar plot). Each
  numbered SI table needs a corresponding `experiment` config in
  Turn 5.
- **Code-availability statement, verbatim**: *"The complete OralVirusGPT
  source code … is provided as a supplementary code archive accompanying
  this submission. The code is released under an open-source MIT
  licence."* The Turn 6 README mirrors this language.
- **Cross-dataset evaluation** — Supp. Table S2 reports zero-shot
  transfer CODe → DENTEX and CODe → Cairo; the Turn 4 evaluation
  pipeline must support a no-fine-tune cross-evaluation mode.
- **Ethics framing** — all data are public and de-identified; the
  `data/` loaders must not exfiltrate, re-identify, or recombine patient
  identifiers.
- **Funding / competing interests** — declared none.

## 10. NEEDS_USER_DECISION

1. **Reconcile against the resubmission code zip vs. clean-room build?**
   The Code-availability statement points at a code archive bundled with
   the resubmission. The kickoff explicitly asks for a first-person
   clean-room build ("not reproducing someone else's work — this is the
   lab's own implementation").
   - (a) Treat the manuscript as the sole specification and build from
     scratch — *recommended*.
   - (b) Ingest the existing zip as a starting reference and reconcile
     line-by-line.
   - Reasoning: option (b) risks importing AI-writing tells, hard-coded
     local paths, or unverified code; option (a) keeps the voice
     authentically lab-side.
   - **Default if no objection: (a).**

2. **Optimiser identity for Phase 2.** The manuscript fixes lr,
   schedule, warmup, weight-decay (implicit via λ_reg), and label
   smoothing, but does not name the optimiser.
   - (a) AdamW (the InternVL2.5 release default — *most likely*);
   - (b) Adam with explicit weight-decay decoupling (matches the
     λ_reg = 1 × 10⁻⁴ reading);
   - (c) Lion (lighter footprint at bf16, sometimes used with DeepSpeed
     ZeRO-2).
   - **Recommended: (a) AdamW with `betas = (0.9, 0.999)`.**

3. **Effective batch composition.** Methods §Impl ¶2 states "batch
   size 16" in Phase 2 on 4 × A100. Three readings:
   - (a) per-GPU 16 → world batch 64, grad_accum 1;
   - (b) global batch 16 → per-GPU 4, grad_accum 1;
   - (c) per-GPU 16, grad_accum 2 → world batch 128.
   - **Recommended: (a)** — matches the reported 100 GPU-hour wall-clock
     at ~3 200 train steps on the CODe split.

4. **Dataset licences for Cairo, Tufts, MMOral, MODID.** The manuscript
   does not state these explicitly. Turn 1 will fetch the canonical
   landing pages and record the verbatim licence text; flagging now so
   that the user can decide whether to (a) accept the fetched text, or
   (b) require direct upstream contact with the dataset stewards before
   inclusion in the public code release.

---

**End of Turn 0.** Awaiting user approval (or override) of the nine
fields and the four `NEEDS_USER_DECISION` items above before scaffolding
begins in Turn 1 / Turn 2.
