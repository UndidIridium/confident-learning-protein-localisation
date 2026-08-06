#!/usr/bin/env python3
"""champion_cleaning_sweep.py

3-way comparison of cleaning methods on the multi-layer attn champion:

  cleanlab  — current champion: cleanlab 2-pass self_confidence @ 0.40 (per-row)
  v47       — historical best: per-cell OOF<0.005 → flip Y=1→0 (no cleanlab)
  hybrid    — cleanlab 2-pass + v47 per-cell drops + v41 per-cell corrections

All use: multi-layer attn L20-23 (4096d) + SPACE (512d) + aux (2d) = 4610d
         MLP 4610→512→7, dropout=0.5, early stopping

Usage:
  python3 champion_cleaning_sweep.py --mode cleanlab   (≈40 min)
  python3 champion_cleaning_sweep.py --mode v47        (≈40 min)
  python3 champion_cleaning_sweep.py --mode hybrid     (≈50 min — extra per-cell fix step)

  # Run all 3:
  for m in cleanlab v47 hybrid; do python3 champion_cleaning_sweep.py --mode $m; done

Output:
  output_cleaning_sweep/cleaning_sweep_{mode}.json   (per-fold results)
  output_cleaning_sweep/cleaning_sweep_summary.json   (3-way comparison table)
"""

import json, os, time, warnings, argparse, sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cleanlab.multilabel_classification.rank import get_label_quality_scores

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.resolve()
SRC_CSV     = PROJ / "data" / "df_adi.csv"
ATTN_H5     = str(PROJ / "data" / "prott5_attn_all_layers.h5")
SPACE_EMB   = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK  = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS   = PROJ / "data" / "df_adi_aux_features.npy"

ATTN_LAYERS = [20, 21, 22, 23]
ATTN_PREFIX = "attn_layer_"

HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5

# Default cleanlab cutoff
CL_CUTOFF = 0.40
# v47 drop threshold: OOF < DROP_THR and Y=1 → flip to 0
V47_DROP_THR = 0.005
# v41 correction threshold: OOF > CORR_THR and Y=0 → flip to 1
V41_CORR_THR = 0.98

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]


class MLP(nn.Module):
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, outdim))
    def forward(self, x):
        return self.net(x)


def posw(Y):
    pw = np.ones(M, dtype=np.float32)
    for j in range(M):
        pos = float(Y[:, j].sum()); neg = float(Y.shape[0]) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg / pos)
    return np.clip(pw, 1.0, 20.0)


def train_mlp(Xtr, Ytr, Xte, Yte, seed=42, return_model=False):
    """Train 1-layer MLP, return (macro_f1, per_class_f1s, test_probs)."""
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr).astype(np.float32)
    Xtes = sc.transform(Xte).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    ti, ei = train_test_split(np.arange(len(Xts)), test_size=ES_FRAC, random_state=seed)
    pw = posw(Ytr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw.astype(np.float32)))
    model = MLP(Xts.shape[1], HIDDEN, M, DROPOUT)
    opt = optim.Adam(model.parameters(), lr=LR)
    Xt = torch.from_numpy(Xts); Yt = torch.from_numpy(Ytr.astype(np.float32))
    Xe = torch.from_numpy(Xts[ei]); Ye = Ytr[ei]
    best_f1, best_state, stall = -1.0, None, 0
    for ep in range(1, MAX_EP+1):
        model.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), BATCH_SIZE):
            ix = perm[s:s+BATCH_SIZE]
            criterion(model(Xt[ix]), Yt[ix]).backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): ep_ = torch.sigmoid(model(Xe)).numpy()
        ef = float(np.mean([f1_score(Ye[:,j].astype(int), (ep_[:,j]>=THR).astype(int), zero_division=0) for j in range(M)]))
        if ef > best_f1 + 1e-6:
            best_f1 = ef; stall = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
            if stall >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    pr = (tp >= THR).astype(int)
    pc = [float(f1_score(Yte[:,j].astype(int), pr[:,j], zero_division=0)) for j in range(M)]
    if return_model:
        return float(np.mean(pc)), pc, tp, model, sc
    return float(np.mean(pc)), pc, tp


def gen_oof(X, Y, n_folds=4, seed=42):
    """4-fold out-of-fold predictions."""
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    fs = n // n_folds
    for f in range(n_folds):
        vs = f*fs; ve = n if f == n_folds-1 else (f+1)*fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
        _, _, tp = train_mlp(X[ti], Y[ti], X[vi], Y[vi], seed=seed+f)
        oof[vi] = tp
        f1_f = float(np.mean([f1_score(Y[vi][:,j].astype(int), (tp[:,j]>=THR).astype(int), zero_division=0) for j in range(M)]))
        print(f"        [OOF Fold {f+1}] F1={f1_f:.4f}", flush=True)
    return oof


def cleanlab_step(Y, oof, cutoff):
    """Per-row cleanlab pruning (current champion method)."""
    labs = [list(np.where(Y[i]==1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"        Cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)")
    return keep


def v47_per_cell_fix(Y, oof, drop_thr=V47_DROP_THR):
    """v47-style per-cell drop: flip Y[i,j]=1→0 where OOF[i,j] < drop_thr.
    
    Returns corrected Y. Does NOT drop rows — fixes individual cells only.
    """
    Y_fixed = Y.copy()
    drop_mask = (Y == 1) & (oof < drop_thr)
    n_drops = int(drop_mask.sum())
    Y_fixed[drop_mask] = 0
    per_org_drops = [int(drop_mask[:, j].sum()) for j in range(M)]
    print(f"        v47 per-cell drop (OOF<{drop_thr}): {n_drops} cells flipped 1→0")
    for j, c in enumerate(COMPARTMENTS):
        if per_org_drops[j] > 0:
            print(f"          {c:>15s}: {per_org_drops[j]} drops")
    return Y_fixed


def v41_per_cell_correct(Y, oof, corr_thr=V41_CORR_THR):
    """v41-style per-cell correction: flip Y[i,j]=0→1 where OOF[i,j] > corr_thr.
    
    Returns corrected Y. Does NOT drop rows — fixes individual cells only.
    """
    Y_fixed = Y.copy()
    corr_mask = (Y == 0) & (oof > corr_thr)
    n_corrs = int(corr_mask.sum())
    Y_fixed[corr_mask] = 1
    per_org_corrs = [int(corr_mask[:, j].sum()) for j in range(M)]
    print(f"        v41 per-cell correction (OOF>{corr_thr}): {n_corrs} cells flipped 0→1")
    for j, c in enumerate(COMPARTMENTS):
        if per_org_corrs[j] > 0:
            print(f"          {c:>15s}: {per_org_corrs[j]} corrections")
    return Y_fixed


def load_data():
    """Load and return (X_all, Y_all, parts, src)."""
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    
    # Multi-layer attn ProtT5 (4096d)
    layers = []
    with h5py.File(ATTN_H5, "r") as f:
        for lyr in ATTN_LAYERS:
            key = f"{ATTN_PREFIX}{lyr:02d}"
            arr = f[key][:].astype(np.float32)
            layers.append(arr)
    prot5_multi = np.concatenate(layers, axis=1)
    print(f"      Multi-layer attn: {prot5_multi.shape}  (layers {ATTN_LAYERS})")
    
    # SPACE + aux
    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    aux_feats = np.load(AUX_FEATS)
    assert aux_feats.shape[0] == len(src), f"AUX row count {aux_feats.shape[0]} != {len(src)}"
    
    X_all = np.concatenate([prot5_multi, net_filled, aux_feats], axis=1).astype(np.float32)
    print(f"      Feature matrix: {X_all.shape}  "
          f"(4096 attn + 512 SPACE + 2 aux = 4610d)")
    return X_all, Y_all, parts, src


def run_fold_cleanlab(X_tr, Y_tr, X_te, Y_te, holdout):
    """Current champion: cleanlab 2-pass self_confidence (per-row pruning)."""
    print(f"\n  ── CLEANLAB MODE (current champion) ──")
    
    # Baseline
    print(f"  Baseline ({len(Y_tr)} train, {len(Y_te)} test)...")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)
    
    # R1 OOF + cleanlab
    print(f"  Round 1 OOF...")
    oof_r1 = gen_oof(X_tr, Y_tr)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    
    # R2 OOF + cleanlab
    print(f"  Round 2 OOF...")
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    
    # Final
    print(f"  Final ({len(Y_r2)} train)...")
    final_f1, final_pc, _ = train_mlp(X_r2, Y_r2, X_te, Y_te)
    
    print(f"  Result: Baseline={base_f1:.4f}  Champion={final_f1:.4f}  "
          f"Gain={final_f1-base_f1:+.4f}")
    return {
        "holdout": holdout,
        "cleaning": "cleanlab",
        "n_train": int(len(Y_tr)),
        "n_after_r1": int(len(Y_r1)),
        "n_after_r2": int(len(Y_r2)),
        "baseline_f1": round(base_f1, 4),
        "champion_f1": round(final_f1, 4),
        "gain": round(final_f1 - base_f1, 4),
        "baseline_per_class": [round(x, 4) for x in base_pc],
        "champion_per_class": [round(x, 4) for x in final_pc],
    }


def run_fold_v47(X_tr, Y_tr, X_te, Y_te, holdout):
    """v47-style: 2-pass per-cell drop (OOF<0.005 → flip Y=1→0), no cleanlab.
    
    Pass 1: train on raw → per-cell drop on OOF
    Pass 2: train on fixed → per-cell drop again
    Final: train on twice-fixed labels
    """
    print(f"\n  ── V47 MODE (per-cell drop, no cleanlab) ──")
    
    # Baseline
    print(f"  Baseline ({len(Y_tr)} train, {len(Y_te)} test)...")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)
    
    # R1 OOF + v47 per-cell drop
    print(f"  Round 1 OOF...")
    oof_r1 = gen_oof(X_tr, Y_tr)
    Y_r1 = v47_per_cell_fix(Y_tr, oof_r1)
    n_drops_r1 = int((Y_tr != Y_r1).any(axis=1).sum())
    print(f"        Rows touched: {n_drops_r1}  (cells: {(Y_tr != Y_r1).sum()})")
    # Keep all rows (v47 is per-cell, not per-row)
    
    # R2 OOF + v47 per-cell drop again
    print(f"  Round 2 OOF...")
    oof_r2 = gen_oof(X_tr, Y_r1)  # re-OOF on same rows with fixed labels
    Y_r2 = v47_per_cell_fix(Y_r1, oof_r2)
    n_drops_r2 = int((Y_r1 != Y_r2).any(axis=1).sum())
    print(f"        Rows touched (round 2): {n_drops_r2}  (cells: {(Y_r1 != Y_r2).sum()})")
    
    # Final on twice-fixed labels (all rows kept)
    print(f"  Final ({len(Y_tr)} train, v47-fixed labels)...")
    final_f1, final_pc, _ = train_mlp(X_tr, Y_r2, X_te, Y_te)
    
    print(f"  Result: Baseline={base_f1:.4f}  Champion={final_f1:.4f}  "
          f"Gain={final_f1-base_f1:+.4f}")
    return {
        "holdout": holdout,
        "cleaning": "v47",
        "n_train": int(len(Y_tr)),
        "n_drops_r1": int((Y_tr != Y_r1).any(axis=1).sum()),
        "n_drops_r2": int((Y_r1 != Y_r2).any(axis=1).sum()),
        "n_cells_dropped_r1": int((Y_tr != Y_r1).sum()),
        "n_cells_dropped_r2": int((Y_r1 != Y_r2).sum()),
        "baseline_f1": round(base_f1, 4),
        "champion_f1": round(final_f1, 4),
        "gain": round(final_f1 - base_f1, 4),
        "baseline_per_class": [round(x, 4) for x in base_pc],
        "champion_per_class": [round(x, 4) for x in final_pc],
    }


def run_fold_hybrid(X_tr, Y_tr, X_te, Y_te, holdout):
    """Hybrid: cleanlab 2-pass (per-row prune) + v47 per-cell drops + v41 per-cell corrections.
    
    1. Cleanlab r1 + r2 (same as cleanlab mode) → clean survivors
    2. v47 per-cell drop on survivors (fix remaining FPs)
    3. v41 per-cell correction on survivors (fix remaining FNs)
    4. Final train on surgically-fixed labels
    """
    print(f"\n  ── HYBRID MODE (cleanlab + per-cell fixes) ──")
    
    # Baseline
    print(f"  Baseline ({len(Y_tr)} train, {len(Y_te)} test)...")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)
    
    # ── Stage 1: cleanlab 2-pass (same as champion) ──
    print(f"  [Stage 1] Cleanlab r1 OOF...")
    oof_r1 = gen_oof(X_tr, Y_tr)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    
    print(f"  [Stage 1] Cleanlab r2 OOF...")
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    print(f"        After cleanlab: {len(Y_r2)} rows kept ({len(Y_tr) - len(Y_r2)} pruned)")
    
    # ── Stage 2: v47 per-cell drop + v41 per-cell correction on survivors ──
    # Single OOF on cleanlab-clean labels; v47 drop (Y=1→0) and v41 corr (Y=0→1)
    # operate on disjoint cells, so one OOF is safe for both steps.
    print(f"  [Stage 2] v47 per-cell drop + v41 correction on cleanlab survivors...")
    oof_r3 = gen_oof(X_r2, Y_r2)
    Y_r3 = v47_per_cell_fix(Y_r2, oof_r3)
    n_v47_drops = int((Y_r3 != Y_r2).sum())
    print(f"        v47 drop cells: {n_v47_drops}")
    Y_r4 = v41_per_cell_correct(Y_r3, oof_r3)  # same OOF — disjoint cells
    n_v41_corrs = int((Y_r4 != Y_r3).sum())
    print(f"        v41 correction cells: {n_v41_corrs}")
    
    # Final on surgically-fixed labels
    print(f"  Final ({len(Y_r2)} train, surgically-fixed labels)...")
    final_f1, final_pc, _ = train_mlp(X_r2, Y_r4, X_te, Y_te)
    
    print(f"  Result: Baseline={base_f1:.4f}  Champion={final_f1:.4f}  "
          f"Gain={final_f1-base_f1:+.4f}")
    return {
        "holdout": holdout,
        "cleaning": "hybrid",
        "n_train": int(len(Y_tr)),
        "n_after_cleanlab": int(len(Y_r2)),
        "n_v47_drop_cells": int(n_v47_drops),
        "n_v41_corr_cells": int(n_v41_corrs),
        "baseline_f1": round(base_f1, 4),
        "champion_f1": round(final_f1, 4),
        "gain": round(final_f1 - base_f1, 4),
        "baseline_per_class": [round(x, 4) for x in base_pc],
        "champion_per_class": [round(x, 4) for x in final_pc],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="all",
                    choices=["cleanlab", "v47", "hybrid", "all"],
                    help="Cleaning mode to run (default: all 3 sequentially)")
    args = ap.parse_args()
    
    t0 = time.time()
    
    # Load data once (all modes share the same features)
    print("=" * 72)
    print("  CLEANING METHOD SWEEP — Multi-layer attn champion")
    print("=" * 72)
    X_all, Y_all, parts, _ = load_data()
    
    modes_to_run = ["cleanlab", "v47", "hybrid"] if args.mode == "all" else [args.mode]
    
    for mode in modes_to_run:
        print(f"\n{'#'*72}")
        print(f"#  MODE: {mode.upper()}")
        print(f"{'#'*72}")
        
        results = []
        for holdout in range(5):
            train_mask = (parts != holdout); test_mask = (parts == holdout)
            X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
            X_te, Y_te = X_all[test_mask], Y_all[test_mask]
            
            print(f"\n  {'='*50}")
            print(f"  HOLD-OUT PARTITION {holdout}  ({Y_tr.shape[0]} train, {Y_te.shape[0]} test)")
            print(f"  {'='*50}")
            
            if mode == "cleanlab":
                r = run_fold_cleanlab(X_tr, Y_tr, X_te, Y_te, holdout)
            elif mode == "v47":
                r = run_fold_v47(X_tr, Y_tr, X_te, Y_te, holdout)
            elif mode == "hybrid":
                r = run_fold_hybrid(X_tr, Y_tr, X_te, Y_te, holdout)
            
            results.append(r)
        
        # Summary for this mode
        print(f"\n  {'='*50}")
        print(f"  5-FOLD CV SUMMARY — {mode.upper()}")
        print(f"  {'='*50}")
        print(f"  {'Holdout':>8}  {'Baseline':>9}  {'Champion':>9}  {'Gain':>8}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}")
        baselines = []; champions = []
        for r in results:
            baselines.append(r["baseline_f1"])
            champions.append(r["champion_f1"])
            print(f"  P{r['holdout']:>7}  {r['baseline_f1']:>9.4f}  {r['champion_f1']:>9.4f}  "
                  f"{r['gain']:>+8.4f}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}")
        print(f"  {'Mean':>8}  {np.mean(baselines):>9.4f}  {np.mean(champions):>9.4f}  "
              f"{np.mean(champions)-np.mean(baselines):>+8.4f}")
        print(f"  {'Std':>8}  {np.std(baselines):>9.4f}  {np.std(champions):>9.4f}")
        
        print(f"\n  Per-class champion F1 (mean ± std):")
        for j, c in enumerate(COMPARTMENTS):
            vals = [r["champion_per_class"][j] for r in results]
            print(f"    {c:>15s}:  {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        
        # Save mode result
        out_dir = PROJ / "output_cleaning_sweep"
        out_dir.mkdir(exist_ok=True)
        report = {
            "mode": mode,
            "per_fold": results,
            "baseline_mean": round(float(np.mean(baselines)), 4),
            "baseline_std": round(float(np.std(baselines)), 4),
            "champion_mean": round(float(np.mean(champions)), 4),
            "champion_std": round(float(np.std(champions)), 4),
            "overall_gain": round(float(np.mean(champions) - np.mean(baselines)), 4),
        }
        report_path = out_dir / f"cleaning_sweep_{mode}.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\n  Report saved: {report_path}")
    
    # Final 3-way comparison
    print(f"\n{'='*72}")
    print("  3-WAY COMPARISON")
    print(f"{'='*72}")
    print(f"  {'Mode':<12s}  {'Mean F1':>8s}  {'Std':>6s}  {'Baseline':>9s}  {'Gain':>8s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*8}")
    
    all_reports = {}
    for mode in ["cleanlab", "v47", "hybrid"]:
        rp = PROJ / "output_cleaning_sweep" / f"cleaning_sweep_{mode}.json"
        if rp.exists():
            rep = json.loads(rp.read_text())
            all_reports[mode] = rep
            print(f"  {mode:<12s}  {rep['champion_mean']:>8.4f}  {rep['champion_std']:>6.4f}  "
                  f"{rep['baseline_mean']:>9.4f}  {rep['overall_gain']:>+8.4f}")
        else:
            print(f"  {mode:<12s}  (not run)")
    
    # Per-compartment comparison across modes
    if len(all_reports) > 1:
        print(f"\n  Per-class champion F1 (mean):")
        print(f"  {'Compartment':>15s}", end="")
        for mode in all_reports:
            print(f"  {mode:>10s}", end="")
        print()
        print(f"  {'-'*15}", end="")
        for _ in all_reports:
            print(f"  {'-'*10}", end="")
        print()
        for j, c in enumerate(COMPARTMENTS):
            print(f"  {c:>15s}", end="")
            for mode in all_reports:
                vals = [r["champion_per_class"][j] for r in all_reports[mode]["per_fold"]]
                print(f"  {np.mean(vals):>10.4f}", end="")
            print()
    
    print(f"\n  Total wall time: {time.time()-t0:.1f}s")
    
    # Save 3-way comparison summary
    summary_out = PROJ / "output_cleaning_sweep" / "cleaning_sweep_summary.json"
    summary_out.write_text(json.dumps({
        "config": f"attn_layers_{ATTN_LAYERS[0]}-{ATTN_LAYERS[-1]}+SPACE+aux",
        "mlp_config": {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR},
        "cleanlab_cutoff": CL_CUTOFF,
        "v47_drop_threshold": V47_DROP_THR,
        "v41_corr_threshold": V41_CORR_THR,
        "modes": {mode: {
            "champion_mean": all_reports[mode]["champion_mean"],
            "champion_std": all_reports[mode]["champion_std"],
            "baseline_mean": all_reports[mode]["baseline_mean"],
            "overall_gain": all_reports[mode]["overall_gain"],
        } for mode in all_reports},
    }, indent=2))
    print(f"  Summary saved: {summary_out}")


if __name__ == "__main__":
    main()
