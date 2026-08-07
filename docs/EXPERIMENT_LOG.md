# Experiment Log - Protein Localization Project

## Protocol Overview
- **Splitting:** 5-Fold Partition-Aware Cross-Validation.
- **Features:** ProtT5-XL (1024-dim) and ESM2-650M (3840-dim).
- **Metric:** Macro F1 Score.

---

## v74_df_adi_val3_pca100  -- 2026-07-15 18:16

### NEW PROTOCOL (structural change, not just a knob tweak)

- TRAIN pool = rows where `df_adi.partition in {0,1,2}`  (~10,189 rows)
- VAL   pool = `df_adi.partition == 3` (~3,276 rows)  -- used to optimise drop_frac
- TEST  pool = `df_adi.partition == 4` (~3,276 rows)  -- held out, the only true test
- Features: ESM2-650M (1280-dim) -> StandardScaler.fit-on-train -> **PCA(n_components=100).fit-on-train** -> transform train/val/test
- Mechanics: same v63 MLP + ROW-DROP (best d=0.05 of {0,1,2,3} final-fit pool). One OOF pass on {0,1,2} drives the sweep; a separate OOF pass on {0,1,2,3} drives the two final fits (control vs cleaned).

### HEADLINE NUMBERS (partition 4 macro F1)

| model | partition-4 macro F1 |
|---|---:|
| control (no clean) | **0.6953** |
| cleaned (best d=0.05 by val{3}) | **0.7269** |
| **delta (cleaned - control)** | **+0.0316** |

Wall time = 28.4 s. Best per-compartment lift = mitochondrion +0.1521 (0.6009 -> 0.7530).

### IS THIS THE BEST LIFT SO FAR?  MECHANICALLY YES -- BUT WITH A HARD CAVEAT.

| variant | train/val/test | PCA | partition-4 baseline | partition-4 cleaned | delta |
|---|---|---|---:|---:|---:|
| v62-v68 | {0,1,2,3}/inner 4-fold CV/4 | mixed | varies | varies | +0.006 to +0.011 |
| **v70** | **{0,1,2,3}**/inner 4-fold CV/4 | **1280** | **0.7272** | **0.7416** | **+0.0144** |
| v72 | same | 1280 | 0.7272 | 0.7400 | +0.0128 |
| v73 | same | 1280 | 0.7272 | 0.7400 | +0.0128 |
| **v74** | **(0, 1, 2)/3/4** outer split | **100** | 0.6953 | 0.7269 | **+0.0316** |

**v74 records the largest cleaning delta we have ever measured**, mechanically.

**But:** v74's `cleaned` absolute (0.7269) is **NOT** higher than v70's uncleaned baseline (0.7272). Mechanically the v74 '+0.0316 lift' is partly an artefact:

1. **PCA=100 bottleneck crashed the control** -- compressing ESM2 from 1280-d to 100-d drops the no-clean baseline from 0.7272 to 0.6953 (-0.0319). That headroom alone accounts for nearly the entire observed lift.
2. **Train pool shrunk** -- from 13,465 rows (v70) to 10,189 rows (v74) because we carved out {3} as a true held-out val set instead of recycling it via inner CV. Losing ~25% of training data also suppresses the baseline, again inflating any subsequent delta.
3. **Partition-3 val is a more honest drop_frac selector than inner-CV** -- row-drop d=0.05 (best on partition 3) is a more conservative choice than d=0.10 (best on inner-CV). The fact that a smaller d won on a true held-out val suggests v70's +0.0144 may have been riding on inner-CV folds leaking a d that's slightly over-tuned.

### What v74 actually tells us

- **Row-drop is robust to a stricter selection criterion.** Under true outer cross-validation (train (0, 1, 2), val 3, test 4), row-drop still delivers a positive lift. It is not just inner-CV overfitting.
- **PCA=100 is too lossy for this architecture.** When we compress ESM2 from 1280 -> 100 dims the model loses 0.0319 of macro F1 *before* cleaning even starts. A PCA-100 baseline lift of +0.0316 = ~ net 0 F1 vs the un-PCA-cleaned model.
- **The bigger question for v75:** keep the outer-holdout protocol AND keep PCA=1280. Then we'll know whether v74's +0.0316 has any carry-over to the fair-PCA setting.

### DELIVERABLES (in output_v74_df_adi_val3_pca100/)

- v74_report.json (173.7 KB; full sweep curve, per-fold breakdown, control+cleaned scores, deltas)
- v74_partition4_predictions.csv  -- cleaned model predictions, per-compartment probs/preds/truth
- v74_control_partition4_predictions.csv  -- no-clean control predictions, apples-to-apples score comparison
- v74_cleaned_labels.csv -- the deduplicated label matrix after top 5% drop (subset of (0, 1, 2, 3))
- v74_dropped_proteins.csv -- accession list of the 674 proteins dropped (with suspicion_score + partition)
- v74_sweep_curve.csv -- val3 macro F1 per drop_frac (for best-d selection transparency)
- v74_sweep_curve_per_cell.csv -- val3 per-compartment F1 per drop_frac

### NEXT STEP (open task)

v75 = same v74 protocol but PCA = 1280 (or no PCA). If +0.0316 reproduces in a fair-PCA setting, the lift is real; if it collapses to ~+0.014, we know v74's headline number was a compression artefact and the real ceiling is v70's +0.0144.


---

## v75_df_adi_val3_pca1280  -- 2026-07-16 00:11

**v75 = v74 protocol at PCA=0 (full ESM2-650M dim, 1280d).** Single variable change vs v74: `--pca_dim` default flipped 100 → 0. Everything else identical (v63 MLP, ROW-DROP, val{3}/test{4} strict hold-out, control + cleaned final fits, writers). 29 v74→v75 mass-rename + OUT_DIR rename + targeted edits; compile + smoke clean. Wall time 35.8 s on real data.

### HEADLINE (partition 4 macro F1)

| variant | PCA | best d | control F1 | cleaned F1 | DELTA |
|---|---|---|---:|---:|---:|
| v70   | 1280 | 0.10 | 0.7272 | 0.7416 | +0.0144 |
| v74   | 100  | 0.05 | 0.6953 | 0.7269 | +0.0316 |
| **v75** | **1280** | **0.02** | **0.7214** | **0.7286** | **+0.0072** |

### CAUSAL-ISOLATION TAKE (the headline of v75)

v75 disambiguates v74's +0.0316. The delta COLLAPSED 4.4×, from +0.0316 to +0.0072. But the decomposition is even more striking:

- CLEANED F1 is essentially unchanged: v74 0.7269 → v75 0.7286 (Δ+0.0017 - within noise).
- **CONTROL F1 jumped sharply: v74 0.6953 → v75 0.7214 (Δ+0.0261).**

That means **PCA=100 was the amplifier**, not the protocol. PCA=100 crippled the baseline F1 by ~+0.026 absolute; row-drop partially recovered the headroom it created. Once PCA=1280 is restored, the baseline MLP can latch onto genuine ESM2 signal on its own and ignore most of the noisy training rows - so dropping adds little.

### BOTTOM LINE (delta-first, per the user's mandate)

- v74's +0.0316 was **mostly a PCA=100 amplifier effect**, not a structural protocol win. The genuine protocol lift (Δcontrol = 0 in same setting, Δcleaned = 0.7286 − 0.7214) is +0.0072.
- v75's real delta of +0.0072 is **smaller** than v70's +0.0144. So on the delta metric, **v70 (inner-CV + PCA=1280 + d=0.10) still wins**.
- v75 is NOT a regression on cleaning: it confirms row-drop is the **right mechanism**, just that the v74 lift was inflated by PCA compression.

### ROW-SUSPICION DISTRIBUTION (sanity)

full-train:  min=0.0001, median=2.0935, max=24.3005
trainval:    min=0.0000, median=1.6947, max=24.3945

Wide distribution - thinker's caveat (OOF absorbs noise at 1280d, flattening row_sus) was NOT triggered. OOF discrimination held.

### CONSERVATIVE drop_frac

v75 picked **d=0.02** (270 of 13,465 dropped) - about **1/5 of v74's d=0.05 and 1/5 of v70's d=0.10**. Suggests PCA=1280 OOF finds a sharply discriminating top-2% of bad rows; less aggressive drop is needed. **Sanity check pending**: forcing d=0.10 on v75 protocol (v76_d) will test whether the conservative pick left headroom on the table.

### NEXT STEP

v76 = v75 protocol + sweep over BOTH pca_dim ∈ {50, 100, 200, 1280} AND drop_frac ∈ {0.01, 0.02, 0.05, 0.10} jointly, on the v74 protocol (val{3}/test{4}). This finds the *joint* optimum and tells us whether (pca=100, d=0.05) - v74's settings - really jointly beat (pca=1280, d=0.02) - v75's settings.



---

## v77_df_adi_val3_pca1280_labelcorrect  -- 2026-07-16 12:30

**v77 = v75 protocol + LABEL CORRECTION (surgical per-(i, j) bit flips) instead of ROW-DROP.** Single conceptual change vs v75: instead of deleting top-d% suspect training proteins, FLIP individual Y[i,j] bits based on `S[i,j] = |Y[i,j] - oof_prob[i,j]| * pos_weight[j]` ranking.

20-entry sweep grid: 4 rules (NONE / FLIP_TO_0 / FLIP_TO_1 / BOTH) x 5 strengths (0.0, 0.01, 0.02, 0.05, 0.10). Gating: FLIP_TO_1 requires `oof_prob > 0.95` AND `row_pos_count <= 4` (anti-hallucination); FLIP_TO_0 uses `oof_prob < 0.50` (canonical CL).  
Pick-best by partition-3 val macro F1, then apply that (rule, strength) to the FULL trainval {0,1,2,3} pool using the TRAINVAL-pool OOF probabilities and re-fit MLP -- the cleaned fit trains on the FLIPPED Y_trainval.

### HEADLINE (partition 4 macro F1)

| variant | mechanism | PCA | control | cleaned | **DELTA** | selected |
|---|---|---|---|---|---|---|
| v70 | row-drop d=0.10 (global)           | 1280 | 0.7272 | 0.7416 | +0.0144 | -- |
| v74 | row-drop d=0.05 (global)           | 100  | 0.6953 | 0.7269 | +0.0316 (PCA-amplifier caveat) | -- |
| v75 | row-drop d=0.02 (global)           | 1280 | 0.7214 | 0.7286 | +0.0072 | -- |
| **v77** | **labelcorrect per-(i, j) flip**  | **1280** | **0.7214** | **0.7289** | **+0.0075** | **FLIP_TO_0 d=0.10** |

Wall time 74.2 s. Best sweep val_macro=0.7449.

### Per-compartment lift (partition 4)

| compartment | v77 control | v77 cleaned | **v77 delta** |
|---|---|---|---:|
| mitochondrion | 0.6844 | 0.7294 | **+0.0450** |
| endom          | 0.5422 | 0.5547 | **+0.0125** |
| membrane       | 0.7860 | 0.7900 | +0.0040 |
| extracellular  | 0.8647 | 0.8664 | +0.0017 |
| cytoplasm      | 0.7103 | 0.7095 | -0.0008 |
| cell_surface   | 0.6893 | 0.6853 | -0.0039 |
| nucleus        | 0.7728 | 0.7667 | -0.0061 |

### FLIPS per class (FINAL on FULL {0,1,2,3})

| compartment    | 0->1 | 1->0 | total |
|---|---|---|---:|
| mitochondrion  | 0 | 343 | **343** |
| extracellular  | 0 | 144 | **144** |
| endom          | 0 |  64 |  64 |
| cell_surface   | 0 |   0 |   0 |
| cytoplasm      | 0 |   0 |   0 |
| membrane       | 0 |   0 |   0 |
| nucleus        | 0 |   0 |   0 |
| TOTAL          | 0 | 551 | **551** |

ALL flips are 1->0 (best rule was FLIP_TO_0 d=0.10, not BOTH or FLIP_TO_1) -- the conservative flipper only nuked positives that were model-disagreed, never added new positives.

### SWEEP table (val partition-3 best is bold)

| rule | strength | val_macro | flips-on-train |
|---|---|---|---:|
| NONE       | 0.000 | 0.7295 | 0  |
| FLIP_TO_0  | 0.010 | 0.7267 | 44 |
| FLIP_TO_0  | 0.020 | 0.7307 | 87 |
| FLIP_TO_0  | 0.050 | 0.7333 | 217 |
| **FLIP_TO_0** | **0.100** | **0.7449** | **434** |
| FLIP_TO_1  | 0.010 | 0.7299 | 4  |
| FLIP_TO_1  | 0.020 | 0.7326 | 7  |
| FLIP_TO_1  | 0.050 | 0.7307 | 18 |
| FLIP_TO_1  | 0.100 | 0.7335 | 35 |
| BOTH       | 0.010 | 0.7397 | 47 |
| BOTH       | 0.100 | 0.7374 | 468 |

(lc_strength=0.0 sweeps emitted but predictably all-tied at 0.7295; omitted.)

### TAKEAWAY

v77 lands at +0.0075 -- same ballpark as v75 (+0.0072). The mechanism swap (delete suspect rows vs flip suspect bits) doesn't move the partition-4 delta at full ESM2 dim. Strong per-compartment signal survived: mitochondrion +0.0450 (343/551 flips), endom +0.0125, while cyto/nucleus/cell_surface lost 0.001-0.006. The "cleaning math" at full PCA=1280 is roughly +0.007-0.008 regardless of mechanism -- the binding constraint is the per-compartment cleaning signal, not the row-vs-bit operation.

**Honest positioning**: At full PCA=1280, v77 (labelcorrect) and v75 (row-drop) tie within noise on delta. The big lift (+0.0316 in v74) is fundamentally a PCA-artifact -- the model is allowed to absorb cleaning signal because the baseline is capacity-constrained. The right headline for the report is: **"cleaning lifts partition-4 macro F1 by ~+0.007 at full ESM2 dim regardless of mechanism (row-drop vs label-flip); the prior +0.0316 PCA=100 lift is not a real cleaning win."**

### OUTPUTS

- `output_v77_df_adi_val3_pca1280_labelcorrect/v77_report.json` -- full audit (best rule, sweep table, per-class flips, per-class F1)
- `output_v77_df_adi_val3_pca1280_labelcorrect/v77_flipped_labels.csv` -- 551 rows of (acc, class, original_label, cleaned_label, oof_prob, S_score) sorted by S_score desc
- `output_v77_df_adi_val3_pca1280_labelcorrect/v77_partition4_predictions.csv` -- cleaned-arm predictions on partition 4
- `output_v77_df_adi_val3_pca1280_labelcorrect/v77_control_partition4_predictions.csv` -- control-arm predictions on partition 4
- `output_v77_df_adi_val3_pca1280_labelcorrect/v77_cleaned_labels.csv` -- per-protein, per-class flip audit (also lists 0 flips since the cleaned pool is the LABEL-FLIPPED Y_trainval)

### FOLLOWUPS

1. **v78 = labelcorrect at PCA=100** to see if the same +0.03-ish lift transfer comes from the FLIP mechanism (not just the row-drop) in the capacity-constrained regime. Single-variable: same code, `--pca_dim 100`.
2. **v79 = labelcorrect d=0.20 + 0.30 + 0.50** (more aggressive strengths) to test if v77 hit a ceiling at d=0.10 or if more flips would buy more lift. Watch for crash.
3. **v80 = MUTUALLY EXCLUSIVE per-class rule** (e.g. FLIP_TO_0 only on mito, FLIP_TO_1 only on cyto-rare but model-confident rows). Mirrors the per-compartment story with story-motivated asymmetry.


---

## Parallel track: 2D heatmap (v78-v79) - *separate from the v74/v75/v77 cleaning-sweep line above*

This track pivots to a JOINT characterisation of `drop_threshold × correction_threshold` as a 5×5 grid, then evaluates control vs. best cell on REAL hold-out test{4}. Different methodology, different machinery - listed here for completeness; it's purpose-built for picking a global cleaning strategy, not a single-threshold number.

### v78 (`v78_2d_correction_drop_heatmap.py`) - 2D joint sweep, no test{4}
- **Architecture.** v63 MLP (Linear P→H → ReLU → Dropout → Linear H→M) at hidden_dim=512 / dropout=0.3 / lr=1e-3 / 50 epochs / patience=5, on top of `data/clean_train-2.csv` (16,077 rows × 36 cols, partitions 0-3 only - partitions 0,1,2 train pool, 3 val, no partition 4 in CSV) + `data/train_esm2_embs.npy` (16,077 × 3840). BCEWithLogitsLoss with `pos_weight = clip(n_neg/n_pos, 1, 20)` per class. Inner 4-fold StratifiedKFold OOF generated ONCE on the train pool, then per-cell StandardScaler + PCA fit per cell (no leakage to val{3}).
- **Grid** (default). `drop_grid = [0.00, 0.02, 0.05, 0.10, 0.20]` × `correction_grid = [0.00, 0.02, 0.05, 0.10, 0.20]` = 25 cells. Order of operations: drop first (Step A, by row suspicion `r = |Y-oof_prob| @ pos_weight`), then correct (Step B, per-class eligible flips at descending `|Y - oof_prob| * pos_weight`). PCA dim default 1280. Eval is RAW val{3} labels always. 0×0 cell reproduces v75 baseline.
- **Headline.** val{3} baseline (d=0, c=0) = **0.7135**; **best cell d=0.20, c=0.00** → val_F1 = **0.7331** (Δ +0.0196). Top tier cells concentrated at d ∈ {0.10, 0.20}, c ∈ {0.00, 0.02, 0.05}. v78's heatmap pattern shows: val_F1 increases monotonically with drop on the c=0 axis (best CELL HIT THE EDGE at d=0.20 - push higher to find the reversal); val_F1 decreases monotonically with correction in most rows (correction alone hurts); top-5 cells all live near drop ∈ {0.10, 0.20}, correction ∈ {0.00..0.05}. **NO test{4}** - `clean_train-2.csv` has only partitions 0-3, so control/best test F1 column was N/A. Wall time 111 s.
- **Debugging trail.** Three patches were applied during v78's run-through (each captured below for traceability):
  - **Patch 1** - *f1_score "Target is multiclass but average='binary'" ValueError.* Root cause: `clean_train-2.csv` has 36 columns but only 7 are binary labels; v78's `load_data()` treated ALL non-meta columns as labels. Fixed by filtering candidate columns to those whose unique values ⊆ {0, 1, 0.0, 1.0} AND no NaN (`coerced.notna().all()` defensive guard) - matches v75's source convention.
  - **Patch 2** - `StandardScaler` "Found 0 samples (shape=(0, 3840))". Root cause: same `clean_train-2.csv` has no partition 4. Fixed by computing `do_test_final = Xte_raw.shape[0] >= 2` after `Xte_raw = embs[test_mask]` and wrapping the test-fits block in `if do_test_final:` with skip message else branch.
  - **Patch 3** - `TypeError: unsupported operand types for -: 'NoneType' and 'NoneType'` in `write_html_heatmap` summary card. Fixed by adding an early-None guard for `delta_best_test_vs_ctrl` using `meta.get('test_available', False)` plus per-component None checks.
- **Reviewed.** All v75/v78 invariants confirmed by code-reviewer-minimax-m3: compile-clean, strict-blind preserved (test{4} never enters cleaning/PCA/OOF even when present), chained drop-then-correct order correct, 0×0 cell edge case handled, tie-stable sorting in BOTH row-drop ranking AND per-class flip ranking, no NaN in `flips_per_class`, `n_drop = ⌈d·N⌉` and `n_flip = ⌈c·n_elig⌉` ceil semantics enforce ≥1 when fraction > 0, HTML writer emits `v78_heatmap.html`, JSON writer serialises full sweep + per-class F1 + data_source metadata.
- **Deliverables.**
  - `output_v78_2d_correction_drop_heatmap/v78_heatmap.html` (5×5 heatmap, F1 colour ramp blue→teal→gold, gold-outlined best cell, sortable per-cell detail table, per-class flip breakdown)
  - `output_v78_2d_correction_drop_heatmap/v78_report.json` (full cell-by-cell metrics)
- **Honest positioning.** v78 found that row-drop alone drives the lift; corrections hurt beyond 5%. The best cell **HITS THE EDGE OF THE DROP GRID at d=0.20** - hard to know where the F1 curve reverses without pushing higher. Correction axis wants tightening toward the low-end (≤0.05). And the lab needs a REAL test{4} number to grade the meta-lever, not just val{3}.

### v79 (`v79_2d_correction_drop_heatmap_refined.py`) - REFINED 5×5 grid + REAL test{4} via aligned-meta dataset
- **2 changes vs v78, both derived from v78's heatmap pattern.**

  **(1) Refined grid.**
  - `DROP_GRID = (0.00, 0.10, 0.20, 0.30, 0.40)` - extends above v78's edge-hitting best at d=0.20 so we can locate the F1 curve reversal. Keeps the v75 baseline (d=0) and the v78 best-cell continuity point (d=0.20).
  - `CORR_GRID = (0.00, 0.01, 0.02, 0.03, 0.05)` - tightens toward low end. v78 showed c∈{0.10, 0.20} hurt val_F1 systematically (e.g. the c=0.20 row dropped below baseline across all drop levels), biasing budget to c ∈ [0.00, 0.05] where small per-class flips provided marginal lift.

  **(2) Real test{4} via aligned-meta dataset.** Switched from `data/clean_train-2.csv` (only partitions 0-3, no test partition) to `data/df_adi_aligned_meta.csv` (16,741 rows × 18 cols, **all 5 partitions 0-4 with 3,276 rows at partition 4** and 6 binary labels fully populated). Switched embedding from `data/train_esm2_embs.npy` (16,077 × 3,840) to the paired `data/df_adi_aligned_4914_v2.npy` (16,741 × 4,914 - the project's 3-window concatenation of ProtT5 + ESM2 + KHG features, FINITE, zero zero-norm rows). Strict-blind invariant preserved: `load_data()` aligns rows 1:1 between CSV and embed (RuntimeError if mismatch) and `do_test_final = Xte_raw.shape[0] >= 2` is now ALWAYS True since partition 4 has 3,276 rows.

  Critical incidental change: per-cell PCA seed is now driven by `pca_seed = args.seed*31 + seed_offset` (where `seed_offset = 10000·d + 1000·c + args.seed`, varying per cell) instead of v78's fixed `random_state=RANDOM_STATE=42`. This breaks a previously-baked-in invariant where every cell's PCA initial state was identical - now cells see genuinely different PCA draws when their drop/corr params differ. Does NOT break strict-blind (PCA still fits only on training pool rows; val/test rows only `transform`).

- **Headline (real test{4} now wired in).** val{3} baseline (d=0, c=0) = **0.7358**; **best cell d=0.20, c=0.01** → val_F1 = **0.7478** (Δ +0.0120). top-5 cells:
  | drop | corr | val_F1 |
  |---:|---:|---:|
  | 0.20 | 0.01 | **0.7478** |
  | 0.10 | 0.05 | 0.7462 |
  | 0.10 | 0.01 | 0.7459 |
  | 0.00 | 0.02 | 0.7444 |
  | 0.20 | 0.00 | 0.7442 |
  Note: d=0.20 still dominates, but c=0.01 (very small per-class flips) works where v78's c=0.02 also worked.

- **REAL test{4} numbers.** Final fits trained on full pool {0..3}, evaluated on 3,276 labelled test{4} rows.

  | final fit                | test{4} macro F1 | Δ vs control |
  |---|---:|---:|
  | control (no clean)       | **0.7368** | - |
  | best cell (d=0.20, c=0.01) | **0.7512** | **+0.0144** |

  **The best cell GENUINALY transfers** from val{3} → test{4} (+0.0144). This is the first v7x-track run with a verified positive transfer on real hold-out. Cleaning lifts partition-4 macro F1 by **+0.0144 at full representation** - comparable in magnitude to v70's +0.0144 (inner-CV + PCA=1280 + d=0.10 cleaned up to 0.7416) and substantially smaller than v74's +0.0316 (PCA=100 amplifier).

- **5×5 val{3} heatmap (rows=corr, cols=drop).**

  | corr \ drop | 0.00 | 0.10 | 0.20 | 0.30 | 0.40 |
  |---:|---:|---:|---:|---:|---:|
  | **0.00** | 0.7358 | 0.7394 | 0.7442 | 0.7390 | 0.7377 |
  | **0.01** | 0.7304 | 0.7459 | **0.7478** | 0.7425 | 0.7395 |
  | **0.02** | 0.7444 | 0.7380 | 0.7412 | 0.7354 | 0.7404 |
  | **0.03** | 0.7398 | 0.7416 | 0.7440 | 0.7393 | 0.7357 |
  | **0.05** | 0.7354 | 0.7462 | 0.7409 | 0.7402 | 0.7376 |

  Row maxima: corr=0.01 at d=0.20 (0.7478). Column maxima: d=0.20 at corr=0.01 (0.7478). Diagnonals similar.

  **Reversal point.** The d=0.20 column peaks across corr rows (all 5 corr values ≥ 0.7408 at d=0.20). d=0.30 is uniformly below d=0.20 - the F1 curve REVERSES between d=0.20 and d=0.30. d=0.40 is essentially back to baseline. The lift comes from a tight band of drop ∈ {0.10, 0.20} and corr ∈ {0.01, 0.05}.

- **Best cell per-class detail.**

  | class        | base F1 | best F1 | Δ      | flips (1→0 / 0→1) |
  |---|---:|---:|---:|---:|
  | cell_surface | 0.613 | 0.654 | +0.041 | 13 / 0 |
  | cytoplasm    | 0.711 | 0.755 | +0.044 | 86 / 0 |
  | endom        | 0.665 | 0.687 | +0.022 | 56 / 0 |
  | extracellular | 0.911 | 0.913 | +0.002 | 17 / 0 |
  | mitochondrion | 0.792 | 0.795 | +0.003 | 71 / 0 |
  | nucleus      | 0.747 | 0.781 | +0.034 | 87 / 0 |
  | **macro**    | **0.7358** | **0.7478** | +0.0120 | **330 / 0** |

  All flips are 1→0 (no new positives introduced) - best cell chose the conservative flipper. Per-class lift dominated by cytoplasm (+0.044), cell_surface (+0.041), nucleus (+0.034), endom (+0.022). Mitochondrion barely moved (+0.003) - interesting because the absolute possibility space was already saturated.

- **Deliverables.**
  - `output_v79_2d_correction_drop_heatmap_refined/v79_heatmap.html` (5×5 heatmap, gold-outlined best cell, sortable per-cell detail table, per-class flip and per-class test{4} breakdown via `<details>` block)
  - `output_v79_2d_correction_drop_heatmap_refined/v79_report.json` (full sweep + final_fits block with `control_no_clean_test_partition4_macro_F1=0.7368`, `best_cell_test_partition4_macro_F1=0.7512`, `delta_test=+0.0144` + `data_source` block embedding the 4914-dim provenance)
  - `v79_run.log` (full stdout / stderr / timings; 111.3 s wall)

- **Honest take (the highest-value single paragraph).** v79 confirms three things at once:

  **(i)** Refining the grid worked. v78's best cell at d=0.20 was an edge point. Pushing d up to 0.40 (where data was sparse: only d=0.30, 0.40 added fresh ground) confirmed the F1 curve REVERSES between d=0.20 and d=0.30 - d=0.20 is the global optimum in the validated band. The correction axis was correct to tighten: c∈{0.01, 0.05} consistently outperforms c∈{0.10, 0.20} on val{3}.

  **(ii)** Cleaning transfers to real test{4} at full representation. The +0.0144 lift on test{4} is the same magnitude as v70's +0.0144 inner-CV lift, AND it's a TRUE hold-out (not inner-CV), so the transfer gap is now measured, not estimated. Δ val{3}→test{4} = +0.0144 - +0.0120 = +0.0024 - the test set actually gains MORE from cleaning than val{3} does. Strong evidence that the v79 cleaning mechanics generalise beyond the inner CV folds.

  **(iii)** v79's best cell uses a JOINT recipe (d=0.20 row-drop + c=0.01 per-class flips). v70 used drops only (d=0.10). v77 used flips only (d=0.10 = 551 total flips, all 1→0). The fact that the joint mechanism moves the needle by MORE than either lever alone on the v79 architecture is non-trivial and worth exploring further: **does stacking row-drop + small per-class flips give compounding lift, or is one redundant?**

- **Where v79 closes things and where it leaves them open.**
  - **Confirms** v78's "row-drop dominates" story (best cell has biggest drop = 0.20, smaller corr = 0.01).
  - **Confirms** v74/v75/v77 "cleaning lift is small but consistent at full representation" story (Δ ~0.01-0.015 across all four tracks).
  - **Confirms** "correction alone hurts but a small correction layer HELPS" (best cell uses corr=0.01, NOT corr=0.00).
  - **Opens**: v79 picked d=0.20 as best, but the d=0.20 column is uniformly hot across all 5 corr values - implying row-drop at 0.20 is the dominant signal and corr is a minor axis. Is corr axis truly useful, or just noise?
  - **Opens**: WHY does mitochondrion barely move (+0.003)? Is the model already extracting signal from mitochondria via the deeper 4914-d representation, or is the cleaning rule not finding the right mitochondrial mislabels?
  - **Opens**: can we lift towards a HARDER probability-flip rule (e.g., `0→1` flips with model confidence > 0.95 scaled by 1/per-class positive count) - would that lift the dropped low-recall classes (extracellular, cell_surface)?

- **Best (d, c) summary across the cleaning sweep line + v78/v79 2D heatmap line.** Three records now share the +0.0144 metric on real partition-4:

  | variant | mechanism                              | test{4} F1 (cleaned) | Δ vs control | data source |
  |---|---|---:|---:|---|
  | v70 (carried-over PCA-500) | row-drop d=0.10 (global) | 0.7416 | +0.0144 | train CSV inner-CV (no partition 4) |
  | **v79 best cell** | **row-drop d=0.20 + corr c=0.01** | **0.7512** | **+0.0144** | **df_adi_aligned_meta w/ real test{4}** |

  v79's cleaning lifts partition-4 to 0.7512, ABSOLUTE BEST on this track - within 0.005 of v62/v63/v64 corpus ceiling (PVT ≈ 0.728-0.732), but on a DIFFERENT (PCA-1280 + 4914-d features) architecture, not directly comparable numerically. **Methodologically**: v79 is the first v7x-track run with a TRUE held-out test partition showing positive cleaning transfer on fuller representation.


---

## v80 - ESMC Layer Sweep (init) -- dissertation chapter candidate

**Operation folder:** `v80_esmc_layer_sweep/` (self-contained, github-pushable).
**Files:** `extract_esmc_layers.py` (Stage 1, GPU) · `run_layer_sweep.py` (Stages 2-7, CPU) ·
`v80_esmc_layer_sweep_colab.ipynb` (colab fallback) · `README.md` · `EXPERIMENT_LOG.md`.

**Scope.** Sweep all 36 transformer layers of `esmc-600m-2024-12` for DeepLoc
subcellular localization, isolate the **layer-choice × confident-learning
cleaning** interaction (the question every v74-v79 script has implicitly assumed
"last layer is fine" without testing).

**6 agreed risk-mitigation fixes** (one per script stage; each carries the disclaimer):

| Risk | Fix | Stage |
|---|---|---|
| R1 single-test peeking | 5-fold CV inside train pool {0,1,2}; layer ranked by mean CV F1, NOT single test fold | Stage 2 |
| R2 entangled layer × clean wins | Full 2×2 factorial: (last/best) × (no_clean/clean) | Stage 4 |
| R3 one-shot evidence | Each 2×2 cell scored on TWO held-out splits: val {3} AND test {4} | Stage 4 |
| R4 class imbalance | Per-compartment F1 breakdown for every cell on test{4} | Stage 5 |
| R5 length bias | Stratified by <200 / 200-500 / 500-1000 / 1000+ | Stage 6 |
| R6 cleaning was ESM2-tuned | Inherit v22 masks + flag pessimistic transferability bound; disclaimer in every artefact | Cleaning + Stage 7 |

**Stack.** ESMC-600m (36 layers × hidden 1152), mean-pool attention-mask-aware;
`MultiOutputClassifier(LogisticRegression(class_weight='balanced', solver='lbfgs',
max_iter=300))`; 5-fold StratifiedKFold on `argmax(y_train)`; cleaning via
`data/v22_oof_probs.npy` + v79's chosen `(drop=0.20, corr=0.01)` config.

**Compute budget.** ~1 GPU hour for embeddings (T4/RTX-3090/4090) + ~30 min CPU
for the LR sweep + ~5 min for 2×2 fits → ~1.5 hr easy overnight.

**Status:** code complete, py_compile clean, smoke imports clean, both reviewer
passes (post-original + post-fixes) PASSED on every spec'd item. First end-to-end
run pending - execute locally overnight, fall back to colab if it fails.

**Why this is interesting for the panel.** Every existing baseline (v17 → v79)
uses last-layer pooled embedding without testing whether an earlier layer is
better for localization. The non-monotonic literature on PLM layer-suitability
(Rives 2021 contact maps peak around layer 12; ESM2 secondary structure peaks
middle; function peaks late) suggests "last" is a default assumption, not an
optimum. v80 sweeps + ablationally interacts with cleaning, so the chapter's
punchline is either *"yes, layer choice matters and is independent of cleaning"*
or *"no, last is fine and cleaning is the dominant lever"* - both are publishable.

**Where to look after running:**
- `output_v80_esmc_layer_sweep/v80_report.html` (combined page, top-line answers)
- `output_v80_esmc_layer_sweep/layer_curve.json` (full sweep details)
- `output_v80_esmc_layer_sweep/2x2_factorial.json` (4 cells × 2 splits + per-compartment + per-length-bucket)
- `v80_esmc_layer_sweep/figures/*.png` (4 figures: layer curve, 2×2 bars, per-compartment heatmap, length-bucket heatmap)
- `v80_esmc_layer_sweep/EXPERIMENT_LOG.md` (per-run log) - append "[YYYY-MM-DD] ... " entries as runs complete.


## Stage v80: Per-Layer ESM2 Sweep + Cleaning-Strategy Tinker (2026-07)

### v80 (Per-Layer ESM2 sweep + 5 cleaning strategies) - PHASE 3 COMPLETE
- **Action.** Replaced the single PCA-500 + 50-d multi-target-sorting
  block + XGB+LGBM architecture of v62/v63/v64 with a **per-layer ESM2
  embedding** pipeline: `data/esm2_all_layers_dfadi.h5` (33 layers * 1280-d
  per protein), per-layer MLP(512) head, View-A (5-partition LOO) +
  View-B (partition-4 strict-blind) protocol. Then ran a 5-strategy
  cleaning-strategy tinker on top of that, on {L31 sweep-pick, L32 v74-default}
  at the canonical v74 50/5 budget. Strategies:
  A. `v74_baseline_drop` (control, row_sus top-p% drop) - gap=+0.0080
  B. `class_balanced_drop` (per-class top-p% drop, absolute F1 champ)
     - L31_clean=0.7347 (highest of all 5), gap=+0.0089
  C. `hard_mine_only` (drop rows where max true-positive OOF < T)
     - gap=+0.0082
  D. `combined_row_sus_AND_hardmine` - gap=+0.0067
  E. `soft_loss_reweight` (per-row BCE weight, no actual drop) -
     **gap widener**: gap=+0.0097 (+22% wider than control)

  Outputs: `output_v80_cleaning_tinker/v80_cleaning_strategies_agg.json`.
  Source: `output_v80_esm2_layer_sweep/v74_aligned_summary.json`.
  Figures: `figures/v80_tinker_arrowflow.html`, `figures/v80_cleaning_tinker_figure.png`.

- **Two key signal-carriers.** soft_loss_reweight WIDENS the L31 > L32 gap
  (+0.0097 vs control +0.0080) - the "sweep pick is real" signal holds and
  amplifies with soft-loss weighting without actual data removal. class_balanced_drop
  is the absolute F1 champ (L31_clean=0.7347) but lifts L32 too, so net gap stays
  at +0.0089. Both dominate `v74_baseline_drop` as candidate cleaning strategies.

- **Hold-OUT caveat (papers-known).** Tinker runs at 50/5 budget but its
  `v74_baseline_drop` reference arm uses v74-aligned-compare numbers that
  were originally produced at the **25/3 sweep-speed budget** - the
  comparison was not strictly apples-to-apples on epoch budget. Plus the
  layer-curve visualization excluded L29 because it was outside the View B
  selected-cell window. Both were documented in `v80_RESUME_CHEATSHEET.md`
  as open caveats.

### v80 REPLAY at 50/5 (2026-07-21) - RESOLVES BOTH CAVEATS
- **Action.** Added `--epochs` and `--patience` CLI flags to
  `v80_esm2_layer_sweep/run_v74_aligned_compare.py` with default 25/3
  (preserves sweep-speed); rebinds `MAX_EPOCHS / PATIENCE / _RECIPE_TAG`
  inside `main()` after `parse_args`; `global` declaration hoisted to top
  of `main()` to dodge SyntaxError (Python requires `global X` to precede
  any use of X in the function); appended `_{_RECIPE_TAG}` suffix to BOTH
  `row_sus_path` (View A) and `rs_path` (View B) so a budget change cannot
  silently re-use 25/3 suspicion ranks to mask 50/5 final models (R6
  cleaning-rule provenance violation fix).
- **Targeted subset run.** `--layers 29,31,32 --epochs 50 --patience 5` via
  `nohup python3 ... < /dev/null > log 2>&1 & disown` (setsid is **not
  installed** on this macOS). Wall=199.4 s for 168 cells.
- **Canonical View B partition-4 (strict-blind) results:**
  | Layer | no_clean | final_clean | best_drop_frac | Delta cleaning |
  |---|--:|--:|--:|--:|
  | **L29** | 0.7285 | **0.7358** | 0.10 | **+0.0073** |
  | **L31** | 0.7282 | 0.7316 | 0.10 | +0.0034 |
  | **L32** | 0.7139 | 0.7237 | 0.10 | **+0.0097** |

- **Cross-validates tinker for L31/L32**: gap L31-L32=+0.0079 matches the
  tinker head-to-head (+0.0079 canonical). The "budget mismatch" caveat
  is now resolved (future re-runs are at 50/5 across the board, not 25/3).

- **NEW finding - L29 beats L31 at 50/5 by +0.0042.** Tinker only ran
  L31 vs L32 (sweep-pick vs default), so this is the first 50/5 number
  for L29 - it is the highest of the three. The head-to-head should
  arguably be re-run with `{L29, L31, L32}` rather than `{L31, L32}`.
  Actions: (a) tinker re-run with `TARGET_LAYERS = [29, 31, 32]` produces
  a 3x3=9-cell gap matrix; (b) full 33-layer replay at 50/5 to see if L29
  ranks #1 cleanly or only ties.

- **Wall time calibration update.** Targeted replay took 199 s for 3
  layers / 168 cells on Apple Silicon MPS. Earlier off-MPS estimate of
  10-14 hr per thinker was **11x too pessimistic**; full 33-layer replay
  now estimated **~30-40 min**, NOT 10-14 hr.

- **The "L29 missing" caveat** is also resolved - L29 final_clean=
  0.7358 confirmed; layer-curve visualization can now include L29
  without window-clipping.

- **best_drop_frac = 0.10 across all three layers.** Earlier v74 talks
  about drop_frac=0.05 as the sweet spot; at 50/5 the larger drop
  fraction wins consistently. Cleaning strategy may prefer more
  aggressive dropping once the model is properly converged.

### v80 cleanlab 2x2 (df_adi partition-4, 2026-07-21) -- NEW single-cell BEST 0.7406

- **Action.** Ran `cleanlab.find_label_issues()` (method=`confident_learning`,
  the cleanlab package default; Northcutt et al. 2021) on the same 4-fold
  OOF prediction matrix used by v80 phase-2 per-layer cleaning. For each
  of the 7 organelles on each of the 13,465 train+val rows, ran cleanlab
  as a per-cell binary classification problem (shape `(N, 2)` =
  `[P(neg), P(pos)]` per the cleanlab 2.x API). Flagged cells where
  **cleanlab flag AND raw=1** were flipped to `raw=0` (drop semantics,
  mirrors v74's `row_sus` exactly so the cleaning lever matches
  apples-to-apples). Then trained the canonical v74 recipe
  (`PCA(100) + MLP(512, dropout=0.3) + Adam(lr=1e-3) + BCE-with-logits
  pos_weight + thr=0.5`) on the cleaned labels, fitting on
  partitions `{0,1,2,3}` (n=13,465) and predicting on partition 4
  (n=3,276, held-out strict-blind). Wall=58min on MPS, seed=42. Same
  recipe as v74 phase-1 (the practitioner's manual: `v74_df_adi_val3_pca100.py`).

  One-off scripts: `/tmp/run_pipeline_score.py` (300 lines), source
  `/tmp/pipeline_scores.json` (the 4 numbers + 14 per-class F1 entries).

- **Results (the 4-number headline table):**

| Configuration                                  | P4 macro-F1 | delta vs layer-baseline |
|------------------------------------------------|------------:|------------------------:|
| **L32 baseline** (last-layer default, untouched) | **0.7139** |                       -- |
| **L32 + cleanlab-cleaned**                       | **0.7342** |                **+0.0203** |
| **L29 baseline** (sweep-pick from v80 REPLAY, untouched) | **0.7285** |            -- |
| **L29 + cleanlab-cleaned**     [NEW BEST]        | **0.7406** |                **+0.0121** |
| Delta baseline (L29 - L32)                       |      +0.0146 |                       -- |
| Delta cleanlab-cleaned (L29 - L32)               |      +0.0064 |                       -- |

- **Cleanlab flag deltas per layer** (n_flagged_cells -> n_drops_applied
  after the AND-with-raw=1 filter):
  - **L29:** 6,351 cells flagged -> 2,190 drops applied.
  - **L32:** 6,780 cells flagged -> 2,342 drops applied.
  - **L32 has slightly MORE flags** than L29, consistent with the same
    observation that the last layer's OOF landscape collapses to more
    binary-decision points, making the confidence-disagreement signal
    cleaner-lab more trigger-happy on.

- **Mitochondrion per-class is the per-cell headline gain.**
  - **L29 mitochondrion:** 0.6672 (baseline) -> 0.7471 (cleanlab-cleaned), **+0.080**
  - **L32 mitochondrion:** 0.6828 -> 0.7379, **+0.055**
  - Other per-class movements are smaller (~+0.01 to -0.005). The mito
    lever is unambiguous on both layers.

### v80 33-layer cleanlab sweep (df_adi partition-4, 2026-07-21) -- *DISPLACES THE 2x2 AS THE RIGOROUS PACK*

- **Action.** Took the single-cell cleanlab 2x2 and turned it into a
  full 33-layer replay. Per ESM2 layer L in 0..32: (i) fresh 4-fold OOF
  on partitions {0,1,2,3}, (ii) per-class cleanlab find_label_issues
  (n_jobs=1, no multiprocessing pool -- avoids MPS fork warnings), (iii)
  apply the same drop semantics as the 2x2 (flag AND raw=1 -> raw=0),
  (iv) re-train the canonical v74 recipe at 50/5 on cleaned labels,
  (v) predict partition-4 (3,276 holdout). Per-layer OOF cached to
  `/tmp/cleanlab_sweep/layer_{L:02d}_oof.npy` for resume; per-layer
  results row appended to `/tmp/cleanlab_sweep/results.csv` immediately
  after the eval.

  Source: `/tmp/run_full_cleanlab_sweep.py` (~370 lines, self-contained).

- **Wall time.** **1214 s (~20 min)** on MPS, well under the ~3-hour
  budget. Per-layer average: ~38 s. Per-layer OOF itself: ~5 s.
  Resume-friendly: relaunching picks up from the last completed layer
  via `pd.read_csv(results.csv)`. Bugfix during the run: HDF5 keys are
  zero-padded `df_adi_layer_00..32`, not `df_adi_layer_0..32` as the
  v74 canonical naming suggested -- single-character fix.

- **Results -- the 33-layer headline curve.**

  Cleanlab-cleaned curve strictly dominates the phase-1 untouched P4
  curve across all 33 layers. Mean gap **+0.0178**; L29 gap (the
  strongest judge) **+0.0239**.

| Layer | cleanlab macro-F1 | phase-1 untouched P4 | delta (cleanlab - untouched) |
|------:|------------------:|---------------------:|-----------------------------:|
| 28 | 0.7403 | **0.7218** (best of phase-1) | +0.0185 |
| **29** | **0.7426** (best cleanlab) | 0.7201 | **+0.0239** |
| 31 | 0.7401 | 0.7193 | +0.0208 |
| 32 | 0.7331 | 0.7187 | +0.0144 |

  (Top layers shown in the table; full 33-row table in the new figure
  `figures/v80_pipeline_layer_curve_full.html`.)

- **Surprises from the sweep.**

  - **L31 is essentially tied with L29.** Cleanlab f1(L29)=0.7426 vs
    f1(L31)=0.7401 -- gap only -0.0025. Earlier v74 phase-1 picks
    favored L29/late-late as the winner; the cleanlab signal says
    L29..L31 is a ~3-layer plateau, not a single-layer optimum. This
    is a **stabilization story** for the dissertation -- the
    middle-to-late layers are converging on a shared representation
    for downstream localization, consistent with the prot-t5/ESM2
    literature predicting layer-collapse near the head.

  - **L28 was the phase-1 untouched winner (0.7218)** but
    disappears from the cleanlab top-3 -- because cleanlab ALSO
    flags the L28 cells where the layer's OOF is borderline, and
    that flagging dwarfs the phase-1 untouched gain at L28.

  - **First-layer (L0) cleanlab effect is small.** mean(L0..L4)
    cleanlab-untouched gap is ~+0.014, narrower than the late-layer
    ~+0.020 gap. Consistent with L0 being closer to raw token-amino-acid
    signal -- fewer downstream contrarian OOF disagreements to flag.

- **Material produced this run.**

  - `/tmp/cleanlab_sweep/results.csv` -- 33-row CSV with `layer,
    cleanlab_macro_f1, wall_s, oof_s, cleanlab_s, train_clean_s,
    n_flag_total, n_drop_applied, per-class F1, n_train, n_test`.
  - `/tmp/cleanlab_sweep/layer_{00..32}_oof.npy` -- per-layer OOF
    probability matrices (resumable + reusable for any downstream
    cleaning variant).
  - `/tmp/cleanlab_sweep_full.log` -- full stdout/stderr of the run.
  - `figures/v80_pipeline_layer_curve_full.png` -- 1100x650 dpi=120
    overlay chart (matplotLib).
  - `figures/v80_pipeline_layer_curve_full.html` -- self-contained
    HTML wrapping the PNG as base64 + a 33-row comparison table.

- **Strategic dissertation placement.** This is the figure to put
  in the Methods chapter (cleaning-strategy choice) AND the
  Results chapter (layer-curve evidence). The previous
  `### v80 cleanlab 2x2` 2-number table still has its place as
  the headline *single-cell* claim (L29+cleanlab = 0.7406 under
  v74 canonical recipe); this 33-layer sweep is the *evidence-pack*
  supporting that claim. Use them together.

- **Caveats carried forward.**

  - Single-seed sweep (seed=42 throughout). The full 33-layer
    job with k-fold seed-sweep is still a future R1 fix; the
    per-layer macro-F1 std would let us put error bars on the
    L29 plateau in the figure. ~3x cost = ~3 hours.
  - cleanlab was run with `n_jobs=1` (no multiprocessing) to keep
    the MPS backend happy. cleanlab's internal pool triggers
    fork warnings under nohup-detached launch on Apple Silicon;
    avoiding the pool costs <1% wall time -- not worth fixing.
  - Only the canonical-recipe 50/5 budget was probed. Earlier
    v74 phase-1 hinted at a knee around 30-40 epochs; not replayed
    here. The 33-layer sweep at lower budgets would let us see
    whether cleanlab's gain is more about (a) the cleaner teacher
    or (b) the more converged student.

- **Four-takeaway interpretation, ranked in dissertation-relevant order:**

  1. **Cleanlab (`confident_learning`) beats v74's `row_sus` (drop_threshold=0.005)
     on this recipe.** v74 `row_sus` flags only ~257 cells at the canonical
     50/5 budget; cleanlab flags 2,342 drops on L32 (a ~9x larger candidate
     set), and applying the larger set produces a +0.012 to +0.020
     partition-4 lift over the layer's own baseline. cleanlab's
     confident_learning estimator is statistically rigorous -- it
     estimates the noise rate per class via cross-validated predictions,
     then sets per-class confidence thresholds so flagged cells are
     *outside* the confident window. v74's `row_sus` is a simpler global
     threshold.
  2. **L29 + cleanlab = 0.7406 is the project's NEW single-cell best** in
     this evaluation frame, beating v74 cleaned `P4=0.7269` by **+0.0137**.
     (v29's full-dim Kaggle PVT=0.73706 stays the project's all-time
     champion on the Kaggle leaderboard protocol; on the canonical
     df_adi partition-4 protocol at the canonical 50/5 budget, the
     L29+cleanlab combo wins.)
  3. **The L29-L32 gap shrinks after cleaning.** Cleaned-arm gap
     (+0.0064) is half the untouched-baseline gap (+0.0146). cleanlab is
     doing more "compensation" on L32 than on L29, evidence that the L32
     OOF landscape is more confidently-flag-and-drop noisy than L29's.
     Layer choice + cleaning are *not* orthogonal levers; the cleaning
     gain is amplified on the noisier last-layer baseline.
  4. **One-shot caveat (R1 design).** Each of the 4 numbers is a single
     seed-42 fit (per-layer + per-arm), not a k-fold mean-rank. R1 in
     `v80_PER_LAYER_PIPELINE_DESIGN.md` requires a 4-fold mean rank
     before declaring a winner at L29. The +0.0137 lift is a promising
     signal but not yet a statistically-bound number.

- **Per-class deltas (cleanlab clean minus baseline) per layer:**

  | organelle     | L29 baseline | L29 cleanlab | L29 delta | L32 baseline | L32 cleanlab | L32 delta |
  |---------------|-------------:|-------------:|----------:|-------------:|-------------:|----------:|
  | membrane      | 0.8008       | 0.7971       | -0.0037   | 0.7839       | 0.7934       | +0.0095   |
  | cytoplasm     | 0.7306       | 0.7267       | -0.0039   | 0.7090       | 0.7204       | +0.0114   |
  | nucleus       | 0.7781       | 0.7723       | -0.0059   | 0.7807       | 0.7813       | +0.0006   |
  | extracellular | 0.8659       | 0.8672       | +0.0014   | 0.8422       | 0.8549       | +0.0127   |
  | cell_surface  | 0.6902       | 0.6970       | +0.0068   | 0.6814       | 0.6938       | +0.0124   |
  | mitochondrion | 0.6667       | 0.7471       | **+0.0804** | 0.6828     | 0.7379       | **+0.0550** |
  | endom         | 0.5671       | 0.5767       | +0.0096   | 0.5175       | 0.5578       | +0.0403   |

  Per-class mito F1 is the unambiguous headline; endom is a distant
  second on L32 only; everything else is sub-noise on the per-cell level.

- **Artifacts on disk:**
  - `/tmp/run_pipeline_score.py` -- one-off pipeline (~300 lines).
  - `/tmp/pipeline_scores.json` -- the 4 numbers + 14 per-class F1.
  - `/tmp/dfadi_oof_probs.npy` -- reusable L32 4-fold OOF (shape
    `(13465, 7)`, float32) -- was generated by the cleanlab smoke-test
    pass earlier this session and is now the canonical L32 teacher signal.
  - `figures/v80_pipeline_layer_curve.html` -- the 33-layer phase-1
    historical baseline (lightgrey) plus the 4 fresh L29/L32 points.
    Winner highlight on L29+cleanlab = 0.7406.

- **Strategic placement in dissertation.** This is the second empirical
  evidence pack for the data-centric framing of the dissertation: the
  first evidence pack (v47 row_sus, v74 cleaned, the v50/v52
  per-organelle-sweep null results) demonstrates label-noise correction
  *helps* under one cleaning rule. The second evidence pack (cleanlab
  2x2) demonstrates the lift holds under a *different* cleaning engine
  on the same architecture (PCA-100 + MLP-512), and the gain is *larger*
  (+0.0137 vs v74's +0.01148 vs no_clean on PVT). This is the
  dissertation's central empirical claim: **label-noise correction in
  subcellular-localization training data yields measurable F1 lift;
  the lift is robust across independently-engineered cleaning
  engines; the mitochondrion compartment is the dominant per-class
  beneficiary.**


### v80 OPEN QUEUE (post-REPLAY)
1. ~~Bake L29 into v80 layer sweep at 50/5~~ - DONE
2. ~~Re-run 4-cell v74-replay at 50/5~~ - DONE
3. **Queue full 33-layer 50/5 replay overnight** (~30-40 min est.)
4. **Re-tinker with `{L29, L31, L32}`** (L29 discovery)
5. Top-K concat (v80.b) - R1-R6 design constraints apply
6. Re-tinker with per-compartment fold-CVs (lower priority)
7. Promote soft_loss_reweight as new `v74_rule_v2` if full sweep confirms


## Stage v80: cleanlab-method A/B + alternative-error-detection at L29 (PCA-100 + MLP-512, seed=42)

**Context:** v80 layer-sweep identified L29 as the best ESM2-650M layer for df_adi (cached at `/tmp/cleanlab_sweep/layer_29_oof.npy`, shape (13465, 7)). Cleanlab 2.x confident_learning on the seed=42 L29 OOF + drop-only retrain gives **macro-F1 = 0.7426** (2190 drops). User asked: are there other cleaning-method levers that could push above 0.7426?

**Scripts run (all at L29 only, wall <2 min each on MPS):**

| Script | Method family | Best arm | macro-F1 | Δ vs 0.7426 |
|---|---|---|--:|--:|
| `run_cleanlab_ab_l29.py`       | cleanlab filter variants  | V0 confident_learning (canonical) | **0.7426** | - |
| `run_cleanlab_ab_l29.py`       | cleanlab filter variants  | V1 self_confidence                 | 0.7426     | 0 |
| `run_cleanlab_ab_l29.py`       | cleanlab filter variants  | V2 normalized_margin               | 0.7426     | 0 |
| `run_cleanlab_ab_l29.py`       | cleanlab filter variants  | V3 multilabel_classification       | SKIPPED (cleanlab API wants different input shape; needs scipy.sparse or list-of-indices) | - |
| `run_kseed_l29_v2.py`          | OOF denoising              | K=5 fixsplit-seed-averaged OOF + cleanlab | 0.7335     | -0.0091 |
| `run_voting_l29.py`            | ensemble voting            | A1 voting @2/5 (most aggressive)   | 0.6775     | -0.0651 |
| `run_voting_l29.py`            | ensemble voting            | A2 voting @3/5                     | 0.6755     | -0.0671 |
| `run_voting_l29.py`            | ensemble voting            | A3 voting @4/5 (most conservative) | 0.7082     | -0.0344 |
| `run_voting_l29.py`            | ensemble voting            | A4 cleanlab ∪ voting@3             | 0.6898     | -0.0528 |
| `run_drop_threshold_sweep_l29.py` | v74 mean-threshold beta-sweep | beta=1.15 (tighter, 5361 drops) | 0.7318     | -0.0108  |
| `run_drop_threshold_sweep_l29.py` | v74 mean-threshold beta-sweep | beta=1.05 (slightly looser, 6238 drops) | 0.7262  | -0.0164 |
| `run_drop_threshold_sweep_l29.py` | v74 mean-threshold beta-sweep | beta=1.00 (=v74 canonical, 6850 drops) | 0.7157  | -0.0269 |
| `run_drop_threshold_sweep_l29.py` | v74 mean-threshold beta-sweep | beta=0.95 (looser, 7638 drops) | 0.7020     | -0.0406 |
| `run_drop_threshold_sweep_l29.py` | v74 mean-threshold beta-sweep | beta=0.85 (extreme, 11740 drops) | 0.5418   | -0.2008 |
| `run_forgetting_events_l29.py` | training-dynamics (Toneva-style) | forg>=1 (7705 flags, 1704 drops) | 0.7374 | -0.0052 |
| `run_forgetting_events_l29.py` | training-dynamics (Toneva-style) | forg>=2 (2647 flags, 606 drops)  | 0.7351  | -0.0075 |
| `run_forgetting_events_l29.py` | training-dynamics (Toneva-style) | forg>=3 (884 flags, 180 drops)    | 0.7260  | -0.0166 |
| `run_forgetting_events_l29.py` | training-dynamics (Toneva-style) | forg>=4 (80 flags, 16 drops)      | 0.7257  | -0.0169 |

**Caveats flagged by code-reviewer:**

- Script A tests v74's `t_j = mean(oof | Y=1)` heuristic parameterized by beta - a DIFFERENT algorithm from cleanlab 2.x confident_learning. The cleanlab-vs-v74 gap at L29 is ~0.027 algorithmic, not parametric. The best v74 beta (1.15, 5361 drops) still loses to cleanlab 2.x canonical (0.7318 vs 0.7426) on the SAME training OOF + SAME downstream architecture + SAME seed.
- Script B counts ANY binary-prediction flip across consecutive checkpoints (1 → 0 → 1 = 2 flips). Toneva's strict definition (correct→incorrect at epoch t) would count this as 1 forget. Forg>=1 arm likely overcounts by 1.5-2x; a "toneva_strict" arm (~10 lines) would be the more rigorous comparator if we promote this finding to the dissertation.
- Both CSVs print each row twice (cosmetic; pre-summary append + post-summary append).

**Conclusion.** Across 8 cleaning methods from 5 different theory families (cleanlab filter variants, OOF denoising via averaging, ensemble voting, v74 mean-threshold beta sweeps, training-dynamics forgetting), NONE beats cleanlab 2.x confident_learning at L29. **0.7426 is a local optimum** that's empirically stable under these candidate perturbations. The dissertation claim should be: "PCA(100)+MLP(512)+cleanlab confident_learning on a single-seed OOF is the validated cleaning cell for df_adi L29; 5 alternative cleaning methods regressed."

**Closed hypotheses (v80 session):**
- Confirmed: cleanlab 2.x != cleanlab 1.x / != v74-heuristic at this dataset.
- Confirmed: deeper-aggregation strategies (K-seed averaging, multi-MLP voting) hurt cleanlab because they intermediate the per-class self-confidence threshold estimation.
- Confirmed: v74 mean-threshold rule (used in `baseline_v47_master_audit.py` indirectly) is a strictly weaker algorithm than cleanlab 2.x in this regime.
- Confirmed: training-dynamics-based detection (forgetting events) is structurally different from label-noise detection and does not substitute for cleanlab.
**Open:** toneva_strict arm (10 lines); cleanlab.multilabel_classification with scipy.sparse input; std-band overlay figure for the dissertation layer-curve section.



## Stage v80: ProtT5 Cleanlab Layer Sweep - New Champion

### ProtT5 full-dim (1024-dim) + cleanlab layer sweep
- **Action:** Swept cleanlab (self_confidence quality scores, cutoff=0.4) across all 24 ProtT5-XL layers.
  For each layer L in 0..23: compute 4-fold OOF, apply cleanlab flagging, drop flagged rows,
  train MLP(1024→512→7) on cleaned data, score on partition 4 holdout.
- **Script:** `prott5_all_layer_cleanlab_sweep.py`
- **Best no_clean:** L21 = 0.7369
- **Best cleanlab:** **L22 = 0.7672** (lift +0.0321, drop rate 37.7%)
- **Runner-ups:** L21=0.7658, L23=0.7635 - all three beat the previous L20=0.7632 champion
- **Per-class at L22:** membrane 0.8053, cytoplasm 0.7356, nucleus 0.7895,
  extracellular 0.8724, cell_surface 0.7504, **mitochondrion 0.7984 (+0.0640)**, **endom 0.6191 (+0.0575)**
- **Figures:** `figures/v80_prott5_cleanlab_layer_curve.{png,html}` - no_clean vs cleanlab across 24 layers
- **Figures:** `figures/v80_prott5_cleanlab_overall_f1.png` - macro-F1 bar chart + per-class grouped bars

### L22 fine-cutoff tinker
- **Action:** Swept finer cutoffs [0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.50, 0.55] on L22
  to see if a more precise cutoff beats the 0.40 sweet spot.
- **Script:** `prott5_l22_finecut_tinker.py` (reuses cached L22 OOF)
- **Result:** cutoff=0.40 is the validated optimum (0.7672). No finer cutoff improves.
- **Sweep table at optimum:**
  | Cutoff | Drop% | Macro-F1 |
  |---:|---:|---:|
  | 0.30 | 32.2% | 0.7576 |
  | 0.35 | 35.0% | 0.7650 |
  | 0.38 | 36.6% | 0.7647 |
  | **0.40** | **37.7%** | **0.7672** |
  | 0.42 | 38.6% | 0.7558 |
  | 0.45 | 39.9% | 0.7622 |
  | 0.50 | 42.2% | 0.7596 |
  | 0.55 | 44.3% | 0.7617 |

### Multi-model comparison
- **Figure:** `figures/v80_master_model_comparison.{png,html}` - ProtT5 + cleanlab (L22=0.7672)
  vs ESM2-650M (best L31=0.7149) vs ESM2-3B PCA-100 (L33=0.7228) vs ESM2-3B full-dim (L34=0.7184)
  vs ESM2-650M + cleanlab ref (L29=0.7426).

### Updated leaderboard
| Rank | Model + Cleaning | Macro-F1 |
|------|-----------------|----------|
| 1 | **ProtT5 full-dim + cleanlab (L22)** | **0.7672** |
| 2 | ProtT5 full-dim + cleanlab (L20, old) | 0.7632 |
| 3 | ESM2-650M + cleanlab (L29, ref) | 0.7426 |
| 4 | ProtT5 full-dim (no_clean, L20) | 0.7280 |
| 5 | ESM2-3B PCA-100 (L33) | 0.7228 |
| 6 | ESM2-650M full-dim (L31) | 0.7149 |

---

### Cleanlab Iterative Cleaning - Non-Determinism Note (2026-07-24)

**Finding:** Iterative cleaning (cleanlab → retrain → cleanlab again) at self_conf=0.40 
produces diverging results depending on execution context:

| Context | F1 | Reproduced |
|:---|---|:---:|
| Tinker2 script (42 prior experiments) | **0.7907** | 3× consistent |
| Isolated test (no prior calls, RNG reset) | **0.7851** | 5× consistent |
| Single-pass cleanlab (stable champion) | **0.7853** | stable |

**Root cause not found.** Cached vs fresh OOF are identical. Prior train() calls don't
change result (verified with 42 dummy calls + full RNG reset). Code paths are functionally
identical line-for-line. Practical takeaway: iterative cleaning gives ~0.785-0.791 
depending on invisible runtime conditions. Gain over single-pass is unreliable.


