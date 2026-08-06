# Script History (Pre-v22 Archive)

The `archive/` directory held pre-Stage-E experimentation scripts and a few
pre-cleaning utilities. They have been retired. This file is the record of what
each script did, the dependencies it needed, and how to recreate it if ever
needed again.

The architecture decision isn't recorded here — that's in
`PRELIMINARY_RESULTS.md` and `FINDINGS_REPORT.md`. This file is a script-level
inventory only.

## Conventions used in each entry

* **Inputs** — CSVs / `.npy` / `.h5` files the script expects.
* **Output** — files the script writes (predictions, logs, summaries).
* **Recreate** — the minimum command(s) and dependencies.

Most early scripts assume the working directory is the project root (`./`)
and that embeddings are loaded from `train_*.npy` / `test_*.npy` next to the
script.

When the script's only docstring was empty or it had a pre-existing syntax
error, the entry flags that.

---

## 1. Stage A baseline prototypes (pre-v22)

### `archive/baselines/baseline_v5_tuned.py`  *(2026-07-04, 9.0 KB)*

Tuned XGBoost on all CSV-derived features (length, molecular weight, pI, etc.),
followed by per-label Cleanlab multilabel API cleaning (v3 logic).

* Inputs: `train.csv`, `test.csv`, `train_prot5_embs.npy`, `test_prot5_embs.npy`
* Output: `output_v5/submission_v1.csv`, `output_v5/submission_final.csv`
* Recreate: `pip install xgboost cleanlab scikit-learn pandas numpy`

### `archive/baselines/baseline_v6_ensemble.py`  *(2026-07-04, 8.1 KB)*

Diversity ensemble — multiple XGBoost models with different hyperparameters
and seeds, averaged. Per-label Cleanlab cleaning at the end.

* Inputs: `train.csv`, `test.csv`, `train_prot5_embs.npy`, `test_prot5_embs.npy`
* Output: `output_v6/submission_v1.csv`, `output_v6/submission_final.csv`
* Recreate: same as v5 plus a seed sweep.

### `archive/baselines/baseline_v7_nn.py`  *(2026-07-04, 4.5 KB)*

Sklearn `MLPClassifier` baseline on frozen embeddings + CSV tabular features.
Simple neural-net sanity check before deeper architectures.

* Inputs: same as v5/v6.
* Output: `output_v7/submission_v1.csv`
* Recreate: `pip install scikit-learn` (MLP is in core sklearn).

### `archive/baselines/baseline_v9_hybrid.py`  *(2026-07-04, 6.6 KB)*

Hybrid Stage A. PCA-denoises the 1024-dim ProtT5 embedding to 256-dim (95%
variance), trains a single multi-task MLP on all six labels, then blends
predictions with an XGBoost trained on raw embeddings.

* Inputs: ProtT5 embeddings.
* Output: `output_v9/submission_v1.csv`
* Recreate: `pip install xgboost scikit-learn`.

### `archive/baselines/baseline_v10_lightgbm.py`  *(2026-07-04, 5.2 KB)*

Stage A LightGBM champion. L2-normalises ProtT5 embeddings (unit length) for
GBDT stability, applies `scale_pos_weight` per label, threshold-tunes on OOF
probs. No Cleanlab.

* Inputs: `train.csv`, ProtT5 / ESM2 embeddings.
* Output: `output_v10/submission_v1.csv`
* Recreate: `pip install lightgbm scikit-learn`.

### `archive/baselines/baseline_v11_autogluon.py`  *(2026-07-04, 4.7 KB)*

Stage A AutoGluon `TabularPredictor` baseline. Hands the task to AutoGluon's
auto-stacker (XGB + LGBM + CatBoost + NN), bagging for stable OOF.

* Inputs: tabular features only (CSV columns).
* Output: `output_v11/submission_v1.csv`
* Recreate: `pip install autogluon.tabular`.

### `archive/baselines/baseline_v12_super_stacker.py`  *(2026-07-04, 6.4 KB)*

Two-level stacker. Level 0: XGBoost + LightGBM + MLP. Level 1:
Logistic Regression meta-classifier on the OOF probability matrix.

* Inputs: ProtT5 + CSV features.
* Output: `output_v12/submission_v1.csv`
* Recreate: `pip install xgboost lightgbm scikit-learn`.

### `archive/baselines/baseline_v13_feature_rich.py`  *(2026-07-04, 6.0 KB)*

Feature-rich XGBoost with hand-engineered features computed in-script:

* `get_aac()` — Amino Acid Composition, 20 features (normalised counts).
* `get_dpc()` — Dipeptide Composition, 400 features (normalised pair counts).
* Plus 1024-dim mean-pooled ProtT5 embedding.

* Inputs: protein sequences, ProtT5 embeddings.
* Output: `output_v13/submission_v1.csv`
* Recreate: `pip install xgboost scikit-learn`.

### `archive/baselines/baseline_v14.py`  *(2026-07-04, 9.8 KB, SYNTAX ERROR)*

Pre-existing unmatched `}` at line 274. Untouched in cleanup. Recreate by
copying a working Stage A baseline (e.g. v10) and adjusting the XGBoost
hyperparameters it sets in `main()`. The file is functionally superseded by
v16 — recreate from there if needed.

### `archive/baselines/baseline_v16_esm2_protocol.py`  *(2026-07-04, 9.7 KB)*

Honest partition-aware ESM2 baseline. Switches from ProtT5 to ESM2 (650M)
Triple-Pooled embeddings (3840-dim). Same robust partition-aware CV and OOF
thresholding as the v14 line. LightGBM with `colsample_bytree=0.5` for
high-dim stability.

* Inputs: `data/train.csv`, `data/test.csv`, `data/train_esm2_embs.npy`,
  `data/test_esm2_embs.npy`
* Output: `submission_v16_<model>.csv`, `results_v16.txt`
* Recreate: `pip install lightgbm scikit-learn`. Run from inside `data/`
  (relative paths are hard-coded).

### `archive/baselines/baseline_v17_esm2_ensemble.py`  *(2026-07-04, 6.1 KB)*

"Heavyweight Champion" — XGB + LightGBM blended soft-vote on 3840-dim ESM2,
per-label threshold tuned on the blended OOF. Partition-aware validation.

* Inputs: ESM2 embeddings.
* Output: `results_v17.txt`
* Recreate: same deps as v16. Outputs flagged results to `results_v17.txt`
  rather than `output_vNN_*/`.

### `archive/baselines/baseline_v19_npy_rf.py`  *(2026-07-04, 2.8 KB)*

Random Forest baseline on raw `.npy` embeddings. Compact script, no
docstring. Likely an intermediate test before formalising the v17 ESM2
ensemble.

* Recreate: `pip install scikit-learn tqdm`. Load CSVs and embeddings next to
  this script; outputs a CSV via `MultiOutputClassifier`.

### `archive/baselines/baseline_v20_esm2_rf.py`  *(2026-07-04, 2.7 KB)*

Random Forest baseline on ESM2 embeddings. Same as v19 but with the ESM2
embedding matrix loaded.

* Recreate: same as v19, point at ESM2 npy files instead of ProtT5.

### `archive/baselines/baseline_v21_final_test.py`  *(2026-07-04, 2.0 KB)*

Random Forest final test version before the formal Stage A shift. Tiny script;
covered by v17.

* Recreate: same as v19 / v20.

---

## 2. Confident-learning prototypes (pre-v22)

These were the first Cleanlab iterations that produced `output_v1..v4`. They
are the prototype series that fed into the later Stage B cleaning sweep.

### `archive/baselines/confident_learning_v1.py`  *(2026-07-04, 10.8 KB)*

First-pass pipeline: XGBoost (per-label binary), Cleanlab ≥ 98% confidence,
class-imbalance guard. Output: `output_v1/confidence_v1.csv`,
`output_v1/submission_v1.csv`.

* Recreate: `pip install xgboost cleanlab.filter scikit-learn`. Files expected
  in CWD: `train.csv`, `test.csv`, `train_prot5_embs.npy`,
  `test_prot5_embs.npy`.

### `archive/baselines/confident_learning_v2.py`  *(2026-07-04, 14.3 KB)*

Adds (a) per-label `scale_pos_weight` based on imbalance ratio, (b) per-label
threshold grid search on OOF probs, (c) Cleanlab threshold lowered to 96%.

* Output: `output_v2/confidence_v1.csv`, `confidence_v2.csv`,
  `optimal_thresholds_v1.csv`, `optimal_thresholds_v2.csv`,
  `cleanlab_summary_v1.csv`, `submission_v1.csv`.

### `archive/baselines/confident_learning_v3.py`  *(2026-07-04, 13.8 KB)*

Adds MULTILABEL Cleanlab (uses 2D label matrix `pred_probs: [n_samples,
n_labels]`) so per-row mis-labelling is caught across multiple labels
simultaneously.

* Output: `output_v3/confidence_v1.csv`, `confidence_v2.csv`,
  `flagged_proteins_v1.csv`, `submission_v1.csv`, `submission_v2.csv`.

### `archive/baselines/confident_learning_v4.py`  *(2026-07-04, 17.0 KB)*

Adds ITERATIVE Cleanlab. First pass cleans on baseline OOF, retrains to get
new OOF, second pass cleans again. Output:
`output_v4/submission_v1.csv`, `submission_final.csv`,
`flagged_proteins_iterative.csv`.

---

## 3. Threshold sweeps (pre-v22)

### `archive/baselines/threshold_sweep.py`  *(2026-07-04, 8.3 KB)*

Sweep over Cleanlab confidence thresholds (95%, 97%) applied to v2 baseline
OOF probs. Outputs submissions per-threshold into `threshold_sweep/`.

* Recreate: requires pre-computed v2 OOF probs.

### `archive/baselines/threshold_sweep_98.py`  *(2026-07-04, 7.0 KB)*

Single-threshold variant — 98% confidence only, head-to-head against the 96%
winner from the wider sweep.

### `archive/baselines/threshold_sweep_colab.py`  *(2026-07-04, 7.9 KB, SYNTAX ERROR)*

Pre-existing `!pip install` shell-magic on line 10 (Jupyter/IPython cell-magic that is invalid in a `.py` file)
that is invalid in a `.py` file. Conceptually identical to `threshold_sweep.py`
but meant for Colab. Recreate by fixing the `!pip install` lines (comment them
with `#` or split them into a separate `!`-prefixed install cell).

---

## 4. v22 fusion variants (local reruns)

### `archive/baselines/baseline_v22_fusion_compressed.py`  *(2026-07-04, 4.8 KB)*

Local rerun of the Colab v22 fusion. PCA compresses the concatenated
ProtT5-Aligned + ESM2-Triple (4864-dim) down to 1024-dim for local-CPU speed.
No cleaning.

### `archive/baselines/baseline_v22_fusion_ensemble.py`  *(2026-07-04, 5.1 KB)*

"Grand Unified" full-dim (4864-dim) tree ensemble. The full-data no-compression
variant. Outputs `results_v22.txt`.

### `archive/baselines/baseline_v22_fusion_mps.py`  *(2026-07-04, 4.8 KB)*

Apple-M3 MPS GPU acceleration for XGBoost on the same v22 fusion setup. Uses
a separate output dir to avoid collision with the CPU run.

### `archive/baselines/cleaned_baseline_v22_fusion_compressed.py`  *(2026-07-04, 4.8 KB)*

Replica of `baseline_v22_fusion_compressed.py` retrained on the Stage B
cleaned training CSV. Comparison purpose.

### `archive/baselines/final_cleaned_baseline_v22_fusion_compressed.py`  *(2026-07-04, 4.8 KB)*

Same setup as the previous cleaned variant — slightly different re-run; both
feed into the early Stage B lift tables.

* Recreate, all five: `pip install xgboost lightgbm scikit-learn tqdm joblib`.
  Most pipeline logic is duplicated near-identically across these — the
  differences are limited to (a) output dir name, (b) PCA on/off, (c) whether
  Cleanlab cleaning was applied upstream.

---

## 5. v23–v36 archive baselines (local reruns)

These are local reruns of the Stage B cleaning iteration series. Each runs
the same v22 compressed setup but with a different cleaning rule applied.
The **canonical** v23–v36 series lives at the repo root (see
`baseline_v46_uniprot_validated_drop.py`, `baseline_v47_drops_only.py`, etc.).
Recreate any archived version by copying the canonical equivalent near the
top-level and changing the cleaning rule.

* **v23_cleaned_fusion_full.py** — full 4864-dim, cleaned via Stage B
  (top 500 noise removed).
* **v23_cleaned_fusion_local.py** — same data, adds an optional PCA
  "TURBO" mode for 8x local speedup.
* **v24_corrected_fusion.py** — full 16077 proteins, surgically corrected
  labels via Cleanlab. PCA on.
* **v25_dropped_labels_fusion.py** — per-organelle label-drop.
* **v26_evidence_fusion.py** — `train_v26_evidence_cleaned.csv`.
* **v27_enriched_fusion.py** — `train_v27_enriched.csv` (mito-enriched).
* **v28_final_champion_local.py** — full-dim fusion with v25 surgical drops.
* **v28_parallel_champion.py** — same as v28 local, parallel via `joblib`
  (6 workers × 2 threads).
* **v30_weighted_compressed.py** — sample-weighting strategy using a
  "discordance score" from v29-caliber OOF probs (w = 1 − discordance).
* **v31_surgical_correction_compressed.py** — conflict-resolution strategy;
  combines v29 mito enrichment with v26 evidence-based label dropping.
* **v32_full_enriched_global.py** — global "missing-1s" rescue: flips
  label=0 to label=1 if UniProt evidence matches. Full-dim parallel via
  `joblib`.
* **v33_surgical_rescue.py** — high-precision surgical rescue (ECO:0000269
  + v22 OOF > 0.5). Full-dim.
* **v33_surgical_rescue_compressed.py** — same logic, PCA-1024, mirror of
  v22 compressed pipeline.
* **v34_surgical_rescue_compressed_strict.py** — ECO:0000269 + v22 OOF > 0.7.
* **v35_surgical_rescue_targeted.py** — organelle-aware threshold
  (cytoplasm gets > 0.9, others use > 0.5).
* **v36_precision_rescue_compressed.py** — global rescue, ECO:0000269,
  per-organelle OOF threshold > 0.8 (> 0.9 for cytoplasm).
* **v36_precision_rescue_fusion.py** — same logic as v36 compressed but in
  full-dim parallel (joblib).

* Recreate, all v23–v36: `pip install xgboost lightgbm scikit-learn joblib`.
  These are the local-cleanup iterations leading up to the top-level
  `baseline_v37_pca_500.py` and onward.

---

## 6. Tools / utilities

### `archive/baselines/ESM_model_colab.py`  *(2026-07-04, 4.6 KB)*

ESM2 embedding generator + immediate XGBoost sanity check. Defines
`get_protein_embeddings()` (transformers ESM2 model loaded with `torch`)
and `process_df()` (mean-pool the per-residue embeddings into one
vector per protein). Built for the Colab environment.

* Recreate: `pip install transformers torch xgboost`.

### `archive/baselines/check_leak.py`  *(2026-07-04, 1.0 KB)*

Tiny `pandas` + `numpy` overlap-check between `train.csv` and `test.csv` on
protein identifier columns. Used as a one-shot firewall before fusing
embeddings.

* Recreate: `pip install pandas numpy`. Run from project root.

### `archive/generate_figures.py`  *(2026-06-09, 3.2 KB)*

Early figure generator. Creates `figures/` folder and ships F1 progression
charts across v1–v4, plus a few single-cell mitochondrion plots
(`mito_location_diversity.png`, `mito_stress_triggers.png`,
`threshold_sweep_trend.png`).

* Outputs: `figures/f1_progression.png`,
  `figures/flagged_distribution_v3.png`,
  `figures/mito_location_diversity.png`,
  `figures/mito_stress_triggers.png`,
  `figures/threshold_sweep_trend.png`.
* Recreate: `pip install pandas matplotlib numpy`. Reads
  `output_v3/flagged_proteins_v1.csv`.

### `archive/baselines/advanced_figures.py`  *(2026-06-09, 5.3 KB)*

PCA plotting for flagged proteins. Plots:

* `figures/embedding_pca_flagged.png`
* `figures/flagged_confidence_dist.png`
* `figures/flagged_label_overlap.png`
* `figures/property_shifts.png`

* Recreate: `pip install pandas matplotlib scikit-learn numpy`. Reads
  `output_v3/flagged_proteins_v1.csv` and the relevant `.npy` embeddings.

---

## 7. Notebooks

### `archive/baselines/BBINF_2026_code.ipynb`  *(2026-06-10, 20.7 KB)*

Kaggle-side scratch notebook from June 10, 2026 (BBINF 2026 review). Defines
`getEmbeddings()` against `h5py`, slices target labels, stacks the embedding
matrix. Useful as a reference for the upstream pipeline shape, but the
canonical re-runnable notebook is `notebooks/baseline_v22_fusion_colab.ipynb`
at the repo root.

* Recreate: pull from a Kaggle / Drive snapshot; the cell-level code is
  plain Python + `h5py` + sklearn `multioutput` and is re-runnable in any
  Jupyter environment with the same data files.

### `archive/baselines/confident_learning_v8_eval.ipynb`  *(2026-06-08, 21.5 KB)*

Per-class F1 evaluation notebook for confident learning. Holds out a
stratified 20% validation set, trains baseline XGB on the remaining 80%,
runs Cleanlab on the 80% only, retrains on cleaned 80%, evaluates both
models on the held-out 20%. Outputs per-class F1/precision/recall and per-
label accession files for UniProt batch lookup.

* Outputs: `per_class_results_v8.csv`, `per_class_f1_v8.png`,
  `dropped_proteins_v8.csv`, `dropped_ALL_accessions.txt`, plus
  per-label `dropped_<label>_accessions.txt` files.

* Recreate: `pip install xgboost cleanlab.filter scikit-learn`. Need
  `train.csv`, `test.csv`, `train_prot5_embs.npy`, `test_prot5_embs.npy`.

---

---

## 8. Post-v22 active submission: v65

This section is **not part of the pre-v22 archive**. It is appended so that the active shipped submission `baseline_v65_multi_target_17mito.py` has a record of its Kaggle benchmark delta and its structural mechanism alongside the archived predecessors in sections 1–7.

### `baseline_v65_multi_target_17mito.py`  *(2026-07-06, ACTIVE — at repo root)*

Clone of `baseline_v61_multi_target.py` (the prior PVT-PRIORITY ship slot) with two surgical modifications layered on top of the existing drops-only cleaning rule.

1. **Enrichment.** 17 UniProt-verified mitochondrial accessions are forced to `mitochondrion=1` regardless of the raw label in `data/train.csv`. The 17 accessions are read from `/Users/aditya/Downloads/mito_FN_named_15.csv` (the same source used by `scripts/stage_b_mito_enrichment.py`); an inline fallback list of 17 accession-isoform IDs is used if the CSV is inaccessible.
2. **Drop-rule exemption.** The drop mask `(raw=1 & v22_OOF < 0.005)` is suppressed on the mitochondrion axis for the 17 accessions only. Without this clause, any newly-enriched `mito=1` cell whose v22 OOF happens to be below 0.005 would be flipped straight back to `0` by the cleaning step.

Architecture is unchanged from v61: ProtT5-Aligned + ESM2 fused → PCA-500, the 50-d multi-target sorting block appended post-PCA, per-organelle XGB + LGBM ensemble averaged 50/50, 4-fold partition CV across the `partition` column, per-organelle threshold optimised on `np.linspace(0.1, 0.75, 50)`.

#### Cleaning rule at a glance

* Correction rule (raw=0 & v22_OOF > 0.98 → 1): **suppressed** (matches v61).
* Drop rule (raw=1 & v22_OOF < 0.005 → 0): **applied** but exempted on the mito axis for the 17 accessions.
* Effective drop count: 988 − (drops on 17-mito mito axis) ≈ 988 − small handful. In this run, **1 hit** on a 17-mito row had `y_aug[mito]=1 & v22_oof[mito]<0.005` post-enrichment and was preserved via the exemption.

#### Inputs and outputs

* **Inputs**: `data/train.csv`, `data/test.csv`, `data/v22_oof_probs.npy`, `data/{train,test}_prot5_aligned.npy`, `data/{train,test}_esm2_embs.npy`, `data/{train,test}_sorting_features.npy`, `/Users/aditya/Downloads/mito_FN_named_15.csv` (with inline fallback list of 17 accession-isoform IDs).
* **Outputs** (written to `output_v65_multi_target_17mito/`): `submission_v65_multi_target_17mito.csv`, `oof_probs.npy`, `oof_preds.npy`, `y_train_cleaned.npy`, `corrections_and_drops_log.csv`, `deltas_vs_v61.txt`.

#### Recreate

From project root: `python3 -u baseline_v65_multi_target_17mito.py 2>&1 | tee v65_run.log`. Wall time ~8 min on a 12-core MacBook.

### Kaggle benchmarks (2026-07-06 LB readout)

| Submission | OOF Macro F1 | Private | Public | ΔPVT vs v61 | ΔPUB vs v61 |
|---|---:|---:|---:|---:|---:|
| **v65** (this script) | 0.75958 | **0.73284** | 0.72517 | **+0.00094** | **−0.00172** |
| v61 (prior PVT-PRIORITY) | 0.75940 | 0.73190 | 0.72689 | — | — |
| v63 `d005_corr_0.95` (PRODUCTION balanced) | 0.76791 | 0.72847 | 0.72801 | (v63 PVT −0.00437 vs v65; v63 PUB +0.00284 vs v65) | — |

### Structural takeaway

v65 establishes a new **project-wide PVT-best** (PVT 0.73284, +0.00094 over the prior PVT-PRIORITY v61 and +0.00437 over the balanced v63). It passes the existing `PVT > 0.73190` PVT-PRIORITY decision rule from `PRELIMINARY_RESULTS.md §6` and replaces v61 in that role. v63 retains the **PUB-best** slot (PUB 0.72801, ahead of v65 by +0.00284). The three mechanisms that drove the result:

1. **Enrichment** of 22 raw mito=0 cells on the 17 accessions to `mitochondrion=1`, sourced from the UniProt-verified `stage_b_mito_enrichment.py` Phase-0 manual labels.
2. **Drop-rule exemption** ensuring that the v22-OOF-driven drop rule cannot silently reverse the new `mito=1` cells when their v22 OOF happens to be below 0.005 (1 hit preserved via exemption in this run).
3. **Trade-off:** the same enrichment unilaterally re-weighting toward `mitochondrion=1` produces a small private-set lift but a slightly miscalibrated public-set response (−0.00172 vs v61, −0.00284 vs v63). This is consistent with prior cleaning-sweep patterns: any enrichment that the cleaning oracle weights unilaterally regresses on Kaggle public when the cleaning oracle's label distribution diverges from the Kaggle held-out ground truth.

* **Local-OOF prediction** modeled the private lift accurately (+0.00018 OOF macro delta → +0.00094 Kaggle PVT; +0.00107 mito F1 OOF delta → +0.00107 mt PVT).
* **PUB regression predictability**: not detectable from OOF alone.
* **Biological correctness**: independent of leaderboard — the 17 accessions are UniProt-verified mitochondrial hits that the cleaning oracle missed in raw. The PVT lift is honest.

### Ship-slot decision (per `PRELIMINARY_RESULTS.md §6`)

| Submission | Role after v65 readout |
|---|---|
| `submission_v65_multi_target_17mito.csv` | **PVT-PRIORITY** (new; replaces v61). |
| `submission_v63_d005_corr_0.95.csv` | **PRODUCTION balanced** (unchanged; still PUB-best). |
| `submission_v61_multi_target.csv` | Drops-only reference (demoted from shipped-PVT to documented reference; still in `output_v61_multi_target/`). |

---

## Status

Sections 1–7 cover pre-v22 scripts that **were in `archive/`** before it was deleted from the filesystem; the entries above are everything needed to recreate them. **Section 8 is the active shipped submission** `baseline_v65_multi_target_17mito.py` (still on disk at the project root), which produces the new Kaggle PVT-PRIORITY submission as of 2026-07-06.

If you want to actually
re-create one, copy the closest top-level equivalent (`baseline_vNN_*.py`)
and adjust:

* the embedding file paths (ProtT5 vs ESM2 / Aligned / Triple),
* the cleaning rule (none / Stage B Cleanlab / UniProt-validation),
* the output directory name,
* the model hyperparameters (XGB + LightGBM ensemble are the default).

For the confident-learning prototypes specifically, the modern equivalent is
`scripts/stage_b_apply_cleaning.py` plus a top-level baseline; the v1–v4
series is logically superseded.

For the Stage A prototype series in particular, the closest top-level
equivalents are around `baseline_v22_fusion_colab.ipynb` and
`baseline_v44_manual_and_dropped.py`. Verify against `EXPERIMENT_LOG.md`
before recreating — exact equivalence was not asserted when this entry was
written.
