#!/usr/bin/env python3
"""champion_xgb_p4.py

XGBoost champion on partition 4 - replaces our MLP(a) with XGBoost.
Trains 7 binary XGBoost classifiers (1 per compartment) with:
  - scale_pos_weight for class imbalance
  - Early stopping via held-out validation
  - Cleanlab 2-round self-confidence pruning (same as MLP champion)

Usage:
  python3 champion_xgb_p4.py

Output: output_champion_xgb_p4.json + stdout report
"""

import json, os, time, warnings
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from cleanlab.multilabel_classification.rank import get_label_quality_scores

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
PROT5_H5 = str(PROJ / "data" / "prott5_all_layers_dfadi-3.h5")
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"

LAYER = 22
THR = 0.5
CL_CUTOFF = 0.40
HOLDOUT = 4  # partition 4 as held-out test

# XGBoost hyperparams
XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    verbosity=0,
    n_jobs=4,
    early_stopping_rounds=20,
)

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]


def train_xgb(Xtr, Ytr, Xte, Yte, seed=42):
    """Train 7 binary XGBoost classifiers, one per compartment.
    
    Each classifier gets its own scale_pos_weight from training data.
    Returns: (mean-F1, per-class-F1-list, test-probs (n_te x 7))
    """
    np.random.seed(seed)
    n_tr = len(Xtr)
    probs = np.zeros((len(Xte), M), dtype=np.float32)
    
    for j in range(M):
        # Compute scale_pos_weight = neg/pos
        pos = float(Ytr[:, j].sum())
        neg = float(n_tr - pos)
        spw = max(1.0, min(20.0, neg / max(pos, 1.0)))
        
        # Split off validation set for early stopping
        tr_ix, va_ix = train_test_split(
            np.arange(n_tr), test_size=0.10, random_state=seed + j,
            stratify=Ytr[:, j] if pos > 0 else None
        )
        
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=seed + j,
            **XGB_PARAMS,
        )
        model.fit(
            Xtr[tr_ix], Ytr[tr_ix, j],
            eval_set=[(Xtr[va_ix], Ytr[va_ix, j])],
            verbose=False,
        )
        # Predict probabilities on test
        prob_j = model.predict_proba(Xte)[:, 1].astype(np.float32)
        probs[:, j] = prob_j
    
    # Evaluate
    preds = (probs >= THR).astype(int)
    pc = [float(f1_score(Yte[:, j].astype(int), preds[:, j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc, probs


def gen_oof_xgb(X, Y, n_folds=4, seed=42):
    """Generate OOF predictions using 4-fold cross-validation with XGBoost."""
    n = len(X)
    oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    fs = n // n_folds
    
    for f in range(n_folds):
        vs = f * fs
        ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]
        ti = np.concatenate([idx[:vs], idx[ve:]])
        
        _, _, tp = train_xgb(X[ti], Y[ti], X[vi], Y[vi], seed=seed + f)
        oof[vi] = tp
        
        f1_f = np.mean([
            f1_score(Y[vi][:, j].astype(int), (tp[:, j] >= THR).astype(int), zero_division=0)
            for j in range(M)
        ])
        print(f"        [Fold {f + 1}] F1={f1_f:.4f}", flush=True)
    
    return oof


def cleanlab_step(Y, oof, cutoff):
    """Same 2-round cleanlab as champion_5fold_cv.py - self-confidence scoring."""
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(
        labels=labs, pred_probs=oof.astype(np.float64),
        method="self_confidence", adjust_pred_probs=True
    )
    keep = scores >= cutoff
    print(f"        Cleanlab: {int(keep.sum())} kept, "
          f"{int((~keep).sum())} dropped ({100 * (~keep).sum() / len(Y):.1f}%)")
    return keep


def main():
    print("=" * 70)
    print("  XGBOOST CHAMPION - PARTITION 4 (vs MLP champion 0.8011)")
    print("  7 binary XGBoost classifiers + 2-round cleanlab")
    print("=" * 70)
    
    # Load data
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    
    train_mask = (parts != HOLDOUT)
    test_mask = (parts == HOLDOUT)
    n_tr = train_mask.sum()
    n_te = test_mask.sum()
    
    print(f"\n  Data: {n_tr} train, {n_te} test (partition {HOLDOUT} holdout)")
    
    # Embeddings
    with h5py.File(PROT5_H5, "r") as f:
        prot5 = f[f"df_adi_layer_{LAYER:02d}"][:].astype(np.float32)
    
    net_emb = np.load(SPACE_EMB)
    net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy()
    net_filled[~net_mask] = 0.0
    
    aux_feats = np.load(AUX_FEATS)
    assert aux_feats.shape[0] == len(src)
    
    X_all = np.concatenate([prot5, net_filled, aux_feats], axis=1).astype(np.float32)
    FEAT_DIM = X_all.shape[1]
    print(f"  Features: {FEAT_DIM}-d (ProtT5 L22 {prot5.shape[1]} + SPACE {net_filled.shape[1]} + aux {aux_feats.shape[1]})")
    
    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]
    
    t0 = time.time()
    
    # === BASELINE (no cleanlab) ===
    print(f"\n  Baseline XGBoost ({n_tr} train)...")
    base_f1, base_pc, _ = train_xgb(X_tr, Y_tr, X_te, Y_te)
    print(f"  Baseline F1 = {base_f1:.4f}")
    
    # === ROUND 1: OOF + cleanlab ===
    print(f"\n  Round 1 OOF (4-fold CV)...")
    oof_r1 = gen_oof_xgb(X_tr, Y_tr)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    print(f"  R1 kept: {len(Y_r1)}/{n_tr} ({100 * len(Y_r1) / n_tr:.1f}%)")
    
    # === ROUND 2: OOF + cleanlab ===
    print(f"\n  Round 2 OOF (4-fold CV)...")
    oof_r2 = gen_oof_xgb(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    print(f"  R2 kept: {len(Y_r2)}/{len(Y_r1)} ({100 * len(Y_r2) / len(Y_r1):.1f}%)")
    
    # === FINAL ===
    print(f"\n  Final XGBoost ({len(Y_r2)} train)...")
    final_f1, final_pc, final_tp = train_xgb(X_r2, Y_r2, X_te, Y_te)
    gain = final_f1 - base_f1
    
    dt = time.time() - t0
    
    # === REPORT ===
    print("\n" + "=" * 65)
    print("  XGBoost Champion - P4 Results")
    print("=" * 65)
    print(f"  {'Metric':>20}  {'Score':>8}")
    print(f"  {'-'*20}  {'-'*8}")
    print(f"  {'Baseline':>20}  {base_f1:>8.4f}")
    print(f"  {'Champion (cleanlab)':>20}  {final_f1:>8.4f}")
    print(f"  {'Gain':>20}  {gain:>+8.4f}")
    print(f"  {'Wall time':>20}  {dt:.0f}s")
    
    print(f"\n  Per-compartment champion F1:")
    print(f"  {'Compartment':>15s}  {'XGBoost':>8s}  {'MLP*':>8s}  {'Δ':>8s}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")
    
    # MLP reference from champion_5fold_cv (partition 4 known numbers)
    mlp_ref = {
        "Membrane": 0.8344, "Cytoplasm": 0.7656, "Nucleus": 0.8260,
        "Extracell": 0.8886, "Cell_surf": 0.7570, "Mito": 0.8432, "Endom": 0.6926,
    }
    
    for j, c in enumerate(COMPARTMENTS):
        mlp_v = mlp_ref.get(c, 0.0)
        delta = final_pc[j] - mlp_v
        marker = " " if final_pc[j] > mlp_v + 0.005 else (" " if mlp_v > final_pc[j] + 0.005 else "")
        print(f"  {c:>15s}  {final_pc[j]:>8.4f}  {mlp_v:>8.4f}  {delta:>+8.4f}{marker}")
    
    print(f"\n  {'Overall':>15s}  {final_f1:>8.4f}  {0.8011:>8.4f}  {final_f1 - 0.8011:>+8.4f}")
    
    # Comparison table
    print("\n" + "=" * 65)
    print("  HEAD-TO-HEAD (Partition 4)")
    print("=" * 65)
    print(f"  {'Model':>30s}  {'F1-macro':>9s}")
    print(f"  {'-'*30}  {'-'*9}")
    print(f"  {'  XGBoost champion (this run)':>30s}  {final_f1:>9.4f}")
    print(f"  {'    Baseline XGBoost (no cleanlab)':>30s}  {base_f1:>9.4f}")
    print(f"  {'  MLP champion (ProtT5)':>30s}  {0.8011:>9.4f}")
    print(f"  {'    DeepLoc Accurate (ProtT5-XL)':>30s}  {0.7674:>9.4f}")
    print(f"  {'    DeepLoc Fast (ESM-1b)':>30s}  {0.7491:>9.4f}")
    
    # Save
    report = {
        "model": "XGBoost (7 binary classifiers)",
        "features": f"ProtT5 L22 + SPACE + aux = {FEAT_DIM}d",
        "holdout": HOLDOUT,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "n_after_r1": int(len(Y_r1)),
        "n_after_r2": int(len(Y_r2)),
        "baseline_f1": round(base_f1, 4),
        "champion_f1": round(final_f1, 4),
        "gain": round(gain, 4),
        "baseline_per_class": [round(x, 4) for x in base_pc],
        "champion_per_class": [round(x, 4) for x in final_pc],
        "xgb_params": XGB_PARAMS,
        "cl_cutoff": CL_CUTOFF,
        "wall_time_s": round(dt, 1),
        "comparison": {
            "mlp_champion": 0.8011,
            "deeploc_accurate": 0.7674,
            "deeploc_fast": 0.7491,
            "vs_mlp": round(final_f1 - 0.8011, 4),
            "vs_deeploc_accurate": round(final_f1 - 0.7674, 4),
            "vs_deeploc_fast": round(final_f1 - 0.7491, 4),
        },
    }
    
    report_path = PROJ / "output_champion_xgb_p4.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
