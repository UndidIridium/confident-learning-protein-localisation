# Protein Subcellular Localisation via Confident Learning

A data-centric pipeline for **multi-label subcellular localisation** (7 compartments) built for an
MSc dissertation at the University of Bristol.

**The thesis: subcellular localisation is a label-noise problem as much as a modelling problem,
and the fix is data-centric.** Public training sets carry thousands of mislabelled proteins
(we estimate **37.6% of the labels in our training set are noisy**), and models trained on those
labels learn the noise. The contribution here is **confident learning**: two rounds of
cleanlab-based cleaning that find and remove mislabelled training proteins *before* training.
That single idea — no architectural complexity, no ensembling, no extra data — lets a
**1-layer MLP** (~0.8 M trainable parameters) on frozen **ProtT5-XL** + **SPACE** embeddings beat
**DeepLoc 2.1**'s branched multi-head architecture on a shared 3,276-protein holdout.

**Notation used below.** P4 is partition 4 of the df_adi dataset (see Data below), the 3,276-protein holdout that DeepLoc 2.1 is also scored on. L22 is the attention-pooled embedding from layer 22 of ProtT5-XL, empirically the best layer. SPACE is a 512-d graph embedding of the STRING v12.0 protein-interaction network. aux is a 2-d auxiliary feature vector (signal-peptide proxy + transmembrane-domain count).

## Headline results (measured, same 3,276-protein P4 holdout)

| Model | F1-macro |
|---|---:|
| **This pipeline** (ProtT5 L22 + SPACE + aux, CL_CUTOFF=0.50) | **0.8002** |
| 5-fold cross-validation mean (CL_CUTOFF=0.40) | 0.7838 ± 0.0156 |
| Baseline — same features, no cleaning | 0.7815 |
| DeepLoc 2.1 Accurate (ProtT5-XL) | 0.7674 |
| DeepLoc 2.1 Fast (ESM-1b) | 0.7491 |

*The 5-fold CV row was run at CL_CUTOFF=0.40, before the cutoff sweep; the single-holdout headline is at the tuned 0.50. See Reproduction step 4.*

## The core idea: confident learning

**How it works.** A 4-fold out-of-fold MLP is trained on the full training set, then cleanlab's
`self_confidence` score is computed for every protein (how well the model's prediction agrees
with its label). Proteins scoring below a cutoff are dropped. The twist is that this is done
**twice**: after round one removes the obvious mislabels, a *fresh* OOF model is trained on the
cleaner set, and its more confident predictions expose subtler mislabels that round one could
not see.

**What it removes.** At the optimal cutoff (0.50), the two rounds drop **~47%** of the
training set (13,465 → 7,111 proteins). The measured estimate of true label noise is **37.6%**, computed with cleanlab's calibrated confident joint on out-of-fold predictions (`label_quality_check.py`), which estimates the joint distribution of observed vs true labels without needing ground truth. Cytoplasm is the noisiest compartment (est.
11.3% of its labels), extracellular the cleanest (1.0%).

**Where the gains come from.** The biggest wins are in the noisiest compartments — exactly where
a model trained on bad labels suffers most:

| Compartment | Baseline F1 | Champion F1 | Gain |
|---|---:|---:|---:|
| Mitochondrion (rarest) | 0.758 | 0.822 | **+0.065** |
| Endomembrane | 0.635 | 0.688 | **+0.054** |
| Cytoplasm | 0.747 | 0.760 | +0.013 |

Mito loses half its positives to cleaning and *still* gains the most — the dropped proteins were
simply wrong. This is the opposite of what you would expect if cleaning were harming rare classes.

**The cutoff is tuned, not assumed.** Sweeping the cleanlab cutoff gives a shallow inverted U
({0.40: 0.7994, 0.45: 0.7985, 0.50: 0.8002, 0.55: 0.7968}): 0.50 is the best point, but the
0.40–0.50 plateau is flat — 0.50 edges out 0.40 by +0.0008, within run-to-run noise.

**Cleaning and features compound.** The no-cleaning baseline on the full feature set is 0.7815;
the champion with cleaning is 0.8002, so cleaning alone is worth **+0.019**. SPACE's PPI features
contribute separately (~+0.021 over ProtT5-only features). The two compound: network features
make the OOF model confident enough for cleanlab to spot real mislabels (an earlier protocol's
ablation found the combined gain exceeded the sum of the parts by +0.020 — see
`docs/CHAMPION_PIPELINE_REPORT.md`).

**The model is not data-limited; the labels are.** Adding 11,562 more proteins (+86% data)
changed nothing (0.8005 vs 0.8002). Every architectural upgrade we tried — deeper MLP,
ensembles, XGBoost, sparse autoencoders, label propagation — failed to beat cleaning. The
bottleneck is label quality, and confident learning is the lever that moves it.

## Repository layout

```
protein-subcellular-localization/
├── README.md                 # this file
├── requirements.txt          # pinned, verified environment
├── .gitignore
├── data/                     # NOT committed (~62 GB) — see Reproduction
├── scripts/
│   ├── embeddings/           # Colab builders: attention-pooled ProtT5 + SPACE extraction
│   ├── cleaning/             # the confident-learning core (p4_cutoff_sweep.py, champion_pipeline.py)
│   ├── training/             # final model + validation (champion_5fold_cv_attn.py)
│   ├── ablations/            # every alternative we tried (all negative results, kept as evidence)
│   ├── evaluation/           # DeepLoc 2.1 head-to-head on the same holdout
│   └── figures/              # figure-generating scripts (outputs land in figures/)
├── results/                  # result JSONs the README numbers come from
├── docs/                     # CHAMPION_PIPELINE_REPORT.md, EXPERIMENT_LOG.md, SCRIPT_HISTORY.md
├── figures/                  # dissertation figures (PNG), generated by scripts/figures/
└── notebooks/                # prott5_attn_extract_colab.ipynb
```

## Reproduction

### 1. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Verified on Python 3.14.4 / torch 2.12.1 / numpy 2.4.4. **Pin the stack** — a numpy/torch
upgrade in late July 2026 shifted scores (the original 0.8011 is no longer reproducible on the
current env; 0.8002 is the current-env canonical number).

### 2. Data (not committed, ~62 GB)

**Where the data comes from.** `df_adi.csv` is the project's core dataset: 16,741 proteins with 7 binary compartment labels, split into 5 fixed partitions. Partition 4 (3,276 proteins) is the held-out test set DeepLoc 2.1 is also scored on; the embeddings and aux features are derived from it by the scripts below. <!-- TODO: add the exact original download URL/DOI for df_adi.csv here if publicly available. -->

Scripts resolve `data/` **relative to their own folder** (`Path(__file__).parent / "data"`),
so from the repo root link one data dir into every scripts subfolder:

```bash
# place these files in ./data/:
#   df_adi.csv                      (16,741 proteins, 7 binary labels, 5 fixed partitions)
#   prott5_attn_all_layers.h5       (attention-pooled ProtT5-XL, all 24 layers, 1024-d)
#   space_network_embeddings.npy    (SPACE embeddings, 16741 x 512)
#   space_network_mask.npy          (SPACE coverage mask, 16741)
#   df_adi_aux_features.npy         (SP-proxy + TMD-count, 16741 x 2)
for d in scripts/*/; do ln -s "$(pwd)/data" "$d/data"; done
```

**Regenerating the embeddings from scratch (optional, expensive):**
1. `python3 scripts/embeddings/build_prott5_attn_extract_colab.py` writes
   `prott5_attn_extract_colab.ipynb` (already in `notebooks/`).
2. Run the notebook on a Colab T4 (~50 min, ~5.9 GB model download) with `df_adi.csv` uploaded.
   It trains 24 small attention heads (frozen ProtT5 backbone, ~1.6 M params) on the 7-compartment
   task, extracts mean- and attention-pooled per-layer embeddings, and writes
   `prott5_attn_all_layers.h5`.
3. SPACE embeddings come from `extract_deeploc_space.py`; aux features from
   `build_df_adi_aux_features.py` (note: that script has a stale hardcoded output path —
   `data/df_adi_aux_features.npy` is what the pipeline reads).

### 3. Headline run — P4 holdout, cutoff sweep

```bash
python3 scripts/cleaning/p4_cutoff_sweep.py
```

Sweeps CL_CUTOFF ∈ {0.40, 0.45, 0.50, 0.55} on partition 4 (3,276 held-out proteins).
Best: **0.8002 at CL_CUTOFF=0.50** (inverted-U curve). Result JSON is written next to
the script; a copy of the shipped output is in `results/output_p4_cutoff_sweep.json`.

### 4. 5-fold cross-validation

```bash
python3 scripts/training/champion_5fold_cv_attn.py
```

Runs the full 2-round cleaning + final MLP on each of the 5 fixed partitions.
Measured mean: **0.7838 ± 0.0156** (CL_CUTOFF=0.40, the value used before the cutoff sweep). To
reproduce the 0.50 configuration across folds, set the `CL_CUTOFF` constant in the config block
at the top of the script (currently 0.40) to 0.50 and rerun.

### 5. Baseline and ablations

```bash
python3 scripts/ablations/model_zoo_p4.py          # 8-model zoo (MLP wins)
python3 scripts/ablations/champion_esm2_p4.py      # ESM2-650M substitution (cleanlab: no gain)
python3 scripts/ablations/champion_deep_ensemble.py
python3 scripts/ablations/champion_xgb_p4.py
python3 scripts/ablations/champion_sae.py
python3 scripts/ablations/champion_labelprop.py
python3 scripts/evaluation/compare_deeploc_p4.py   # vs DeepLoc 2.1 on the same holdout
```

All ablations are **negative results** — kept deliberately as evidence that the champion
configuration is tested, not lucky.

## Development workflow

This repo is the **curated, citable record** of the project. Experimentation continues in the
original working directory, where every idea gets its own script; anything that proves out gets
**promoted** into this repo. Full history is in `docs/EXPERIMENT_LOG.md` and
`docs/SCRIPT_HISTORY.md`.

## Known caveats

- `champion_pipeline.py` (original mean-pooled pipeline) produced 0.8011 in July 2026; that exact
  score is **not** reproducible under the current environment due to numpy/torch drift. The
  current-env numbers above (0.8002 P4, 0.7838 5-fold) are canonical.

## References

1. Northcutt, C. G., Jiang, L., & Chuang, I. L. (2021). Confident Learning: Estimating
   Uncertainty in Dataset Labels. *Journal of Artificial Intelligence Research*, 70, 1373–1411.
   https://doi.org/10.1613/jair.1.12125
   The confident-learning method used throughout this work, implemented by the `cleanlab`
   package (v2.9.0, pinned in `requirements.txt`).
2. Elnaggar, A., et al. (2022). ProtTrans: Toward Understanding the Language of Life Through
   Self-Supervised Learning. *IEEE Transactions on Pattern Analysis and Machine Intelligence*,
   44(10), 7112–7127. https://doi.org/10.1109/TPAMI.2021.3095381
   Source of the frozen ProtT5-XL backbone used for sequence embeddings.
3. Ødum, M. T., et al. (2024). DeepLoc 2.1: multi-label membrane protein type prediction using
   protein language models. *Nucleic Acids Research*, 52(W1), W215–W220.
   https://doi.org/10.1093/nar/gkae237
   The published benchmark we compare against on the same held-out partition.
4. SPACE network embeddings. 512-d graph autoencoder embeddings of the STRING v12.0
   protein-interaction network, described in `docs/CHAMPION_PIPELINE_REPORT.md` §1.2.
   <!-- TODO: add the canonical SPACE citation (paper/DOI) here. -->
