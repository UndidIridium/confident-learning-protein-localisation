#!/usr/bin/env python3
"""champion_deep_ensemble.py

Deep Ensemble champion on partition 4.
Trains 5 independent MLPs (seeds 0-4), averages their predictions.
Cleanlab 2-round confidence pruning uses averaged OOF across all 5 seeds.

Pipeline:
  Baseline: train 5 MLPs on full data → average test predictions
  Champion: 4-fold OOF × 5 seeds → average OOF → cleanlab → retrain 5 MLPs → average test predictions

Usage:
  python3 champion_deep_ensemble.py

Output: output_champion_deep_ensemble.json + stdout report
"""

import json, os, time, warnings
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
SRC_CSV = PROJ / "data" / "df_adi.csv"
PROT5_H5 = str(PROJ / "data" / "prott5_all_layers_dfadi-3.h5")
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"

LAYER = 22
HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5; CL_CUTOFF = 0.40
HOLDOUT = 4
N_SEEDS = 5
SEEDS = list(range(N_SEEDS))

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]


class MLP(nn.Module):
    """Same 1-layer MLP as champion: indim → hdim → outdim with dropout."""
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


def train_mlp(Xtr, Ytr, Xte, Yte, seed=42):
    """Train MLP, return (mean-F1, per-class-F1-list, test-probs)."""
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
    for ep in range(1, MAX_EP + 1):
        model.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), BATCH_SIZE):
            ix = perm[s:s + BATCH_SIZE]
            criterion(model(Xt[ix]), Yt[ix]).backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            ep_ = torch.sigmoid(model(Xe)).numpy()
        ef = float(np.mean([f1_score(Ye[:, j].astype(int), (ep_[:, j] >= THR).astype(int), zero_division=0) for j in range(M)]))
        if ef > best_f1 + 1e-6:
            best_f1 = ef; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stall = 0
        else:
            stall += 1
            if stall >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    preds = (tp >= THR).astype(int)
    pc = [float(f1_score(Yte[:, j].astype(int), preds[:, j], zero_division=0)) for j in range(M)]
    f1_mean = float(np.mean(pc))
    return f1_mean, pc, tp


def gen_oof(X, Y, seed=42):
    """Generate OOF predictions for one seed using 4-fold CV.
    Returns: (n x 7) OOF probability array.
    """
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    n_folds = 4
    fs = n // n_folds
    for f in range(n_folds):
        vs = f * fs; ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
        _, _, tp = train_mlp(X[ti], Y[ti], X[vi], Y[vi], seed=seed + f)
        oof[vi] = tp
    return oof


def load_data():
    """Load and return (X_all, Y_all, train_mask, test_mask, FEAT_DIM)."""
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != HOLDOUT)
    test_mask = (parts == HOLDOUT)

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
    return X_all, Y_all, train_mask, test_mask, FEAT_DIM


def main():
    print("=" * 70)
    print("  DEEP ENSEMBLE CHAMPION - PARTITION 4")
    print(f"  {N_SEEDS} independent MLPs + averaged predictions + 2-round cleanlab")
    print("=" * 70)

    t0 = time.time()

    # Load data
    X_all, Y_all, train_mask, test_mask, FEAT_DIM = load_data()
    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]
    n_tr = len(X_tr); n_te = len(X_te)
    print(f"\n  Data: {n_tr} train, {n_te} test | Features: {FEAT_DIM}-d")

    # ===================================================================
    # BASELINE ENSEMBLE: train 5 MLPs on full data, average test probs
    # ===================================================================
    print(f"\n  {'='*60}")
    print(f"  BASELINE ENSEMBLE ({N_SEEDS} seeds, no cleanlab)")
    print(f"  {'='*60}")
    baseline_probs = np.zeros((n_te, M), dtype=np.float32)
    base_pc_all = []
    for i, seed in enumerate(SEEDS):
        t_s = time.time()
        f1, pc, tp = train_mlp(X_tr, Y_tr, X_te, Y_te, seed=seed)
        baseline_probs += tp
        base_pc_all.append(pc)
        dt_s = time.time() - t_s
        print(f"    Seed {seed}: F1={f1:.4f}  ({dt_s:.0f}s)")

    # Average and evaluate
    baseline_probs /= N_SEEDS
    base_preds = (baseline_probs >= THR).astype(int)
    base_pc_ens = [float(f1_score(Y_te[:, j].astype(int), base_preds[:, j], zero_division=0)) for j in range(M)]
    base_f1_ens = float(np.mean(base_pc_ens))
    base_pc_std = [float(np.std([p[j] for p in base_pc_all])) for j in range(M)]

    print(f"\n    {'Ensemble average':30s}: {base_f1_ens:.4f}")
    print(f"    {'Individual seeds (mean)':30s}: {np.mean([np.mean(p) for p in base_pc_all]):.4f}")
    print(f"    {'Ensemble gain vs mean':30s}: +{base_f1_ens - np.mean([np.mean(p) for p in base_pc_all]):.4f}")

    # ===================================================================
    # ROUND 1: OOF from all seeds, averaged, then cleanlab
    # ===================================================================
    print(f"\n  {'='*60}")
    print(f"  ROUND 1 - OOF × {N_SEEDS} seeds → averaged → cleanlab")
    print(f"  {'='*60}")

    # Sum OOF across seeds (will divide later)
    oof_sum = np.zeros((n_tr, M), dtype=np.float64)
    for i, seed in enumerate(SEEDS):
        t_s = time.time()
        oof_s = gen_oof(X_tr, Y_tr, seed=seed)
        oof_sum += oof_s.astype(np.float64)
        dt_s = time.time() - t_s
        print(f"    Seed {seed}: OOF done ({dt_s:.0f}s)")

    oof_r1 = (oof_sum / N_SEEDS).astype(np.float32)

    # Cleanlab on averaged OOF
    labs = [list(np.where(Y_tr[i] == 1)[0]) for i in range(n_tr)]
    scores_r1 = get_label_quality_scores(
        labels=labs, pred_probs=oof_r1.astype(np.float64),
        method="self_confidence", adjust_pred_probs=True
    )
    keep_r1 = scores_r1 >= CL_CUTOFF
    n_kept_r1 = int(keep_r1.sum())
    print(f"\n    Cleanlab R1: {n_kept_r1}/{n_tr} kept ({100 * n_kept_r1 / n_tr:.1f}%)")
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]

    # ===================================================================
    # ROUND 2: OOF on R1-kept rows, averaged, then cleanlab
    # ===================================================================
    print(f"\n  {'='*60}")
    print(f"  ROUND 2 - OOF × {N_SEEDS} seeds on R1-kept → averaged → cleanlab")
    print(f"  {'='*60}")

    oof_sum_r2 = np.zeros((len(Y_r1), M), dtype=np.float64)
    for i, seed in enumerate(SEEDS):
        t_s = time.time()
        oof_s = gen_oof(X_r1, Y_r1, seed=seed)
        oof_sum_r2 += oof_s.astype(np.float64)
        dt_s = time.time() - t_s
        print(f"    Seed {seed}: OOF done ({dt_s:.0f}s)")

    oof_r2 = (oof_sum_r2 / N_SEEDS).astype(np.float32)

    # Cleanlab on averaged OOF (R2)
    labs_r2 = [list(np.where(Y_r1[i] == 1)[0]) for i in range(len(Y_r1))]
    scores_r2 = get_label_quality_scores(
        labels=labs_r2, pred_probs=oof_r2.astype(np.float64),
        method="self_confidence", adjust_pred_probs=True
    )
    keep_r2 = scores_r2 >= CL_CUTOFF
    n_kept_r2 = int(keep_r2.sum())
    print(f"\n    Cleanlab R2: {n_kept_r2}/{len(Y_r1)} kept ({100 * n_kept_r2 / len(Y_r1):.1f}%)")
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]

    # ===================================================================
    # FINAL ENSEMBLE: retrain 5 MLPs on R2-kept, average test probs
    # ===================================================================
    print(f"\n  {'='*60}")
    print(f"  FINAL ENSEMBLE ({N_SEEDS} seeds, R2 clean data)")
    print(f"  {'='*60}")
    champion_probs = np.zeros((n_te, M), dtype=np.float32)
    champ_pc_all = []
    for i, seed in enumerate(SEEDS):
        t_s = time.time()
        f1, pc, tp = train_mlp(X_r2, Y_r2, X_te, Y_te, seed=seed)
        champion_probs += tp
        champ_pc_all.append(pc)
        dt_s = time.time() - t_s
        print(f"    Seed {seed}: F1={f1:.4f}  ({dt_s:.0f}s)")

    # Average and evaluate
    champion_probs /= N_SEEDS
    champ_preds = (champion_probs >= THR).astype(int)
    champ_pc_ens = [float(f1_score(Y_te[:, j].astype(int), champ_preds[:, j], zero_division=0)) for j in range(M)]
    champ_f1_ens = float(np.mean(champ_pc_ens))
    champ_pc_std = [float(np.std([p[j] for p in champ_pc_all])) for j in range(M)]
    gain = champ_f1_ens - base_f1_ens

    dt = time.time() - t0

    # ===================================================================
    # REPORT
    # ===================================================================
    print("\n" + "=" * 65)
    print("  DEEP ENSEMBLE - PARTITION 4 RESULTS")
    print("=" * 65)
    print(f"  {'Metric':>30s}  {'Score':>8s}")
    print(f"  {'-'*30}  {'-'*8}")
    print(f"  {'Baseline (single MLP mean)':>30s}  {np.mean([np.mean(p) for p in base_pc_all]):>8.4f}")
    print(f"  {'Baseline (ensemble avg)':>30s}  {base_f1_ens:>8.4f}")
    print(f"  {'Champion (ensemble + cleanlab)':>30s}  {champ_f1_ens:>8.4f}")
    print(f"  {'Gain (ensemble vs single)':>30s}  {base_f1_ens - np.mean([np.mean(p) for p in base_pc_all]):>+8.4f}")
    print(f"  {'Gain (cleanlab)':>30s}  {gain:>+8.4f}")
    print(f"  {'Wall time':>30s}  {dt:.0f}s ({dt/60:.1f}m)")

    print(f"\n  Per-compartment ensemble champion F1:")
    print(f"  {'Compartment':>15s}  {'Ensemble':>9s}  {'±std':>7s}  {'MLP*':>8s}  {'Δ':>8s}")
    print(f"  {'-'*15}  {'-'*9}  {'-'*7}  {'-'*8}  {'-'*8}")

    # MLP reference (single MLP champion)
    mlp_ref = {
        "Membrane": 0.8344, "Cytoplasm": 0.7656, "Nucleus": 0.8260,
        "Extracell": 0.8886, "Cell_surf": 0.7570, "Mito": 0.8432, "Endom": 0.6926,
    }

    for j, c in enumerate(COMPARTMENTS):
        mlp_v = mlp_ref.get(c, 0.0)
        delta = champ_pc_ens[j] - mlp_v
        marker = " " if champ_pc_ens[j] > mlp_v + 0.005 else (" " if mlp_v > champ_pc_ens[j] + 0.005 else "")
        print(f"  {c:>15s}  {champ_pc_ens[j]:>9.4f}  {champ_pc_std[j]:>7.4f}  {mlp_v:>8.4f}  {delta:>+8.4f}{marker}")

    print(f"\n  {'Overall':>15s}  {champ_f1_ens:>9.4f}  {'':>7s}  {0.8011:>8.4f}  {champ_f1_ens - 0.8011:>+8.4f}")

    # HEAD-TO-HEAD TABLE
    print("\n" + "=" * 65)
    print("  HEAD-TO-HEAD (Partition 4)")
    print("=" * 65)
    print(f"  {'Model':>35s}  {'F1-macro':>9s}")
    print(f"  {'-'*35}  {'-'*9}")
    print(f"  {'  Deep Ensemble champion (this run)':>35s}  {champ_f1_ens:>9.4f}")
    print(f"  {'    Baseline ensemble (no cleanlab)':>35s}  {base_f1_ens:>9.4f}")
    print(f"  {'    Single MLP individual mean':>35s}  {np.mean([np.mean(p) for p in base_pc_all]):>9.4f}")
    print(f"  {'  Single MLP champion (0.8011)':>35s}  {0.8011:>9.4f}")
    print(f"  {'    DeepLoc Accurate (ProtT5-XL)':>35s}  {0.7674:>9.4f}")
    print(f"  {'    DeepLoc Fast (ESM-1b)':>35s}  {0.7491:>9.4f}")

    # Save report
    report = {
        "model": f"Deep Ensemble ({N_SEEDS} MLPs, averaged predictions)",
        "features": f"ProtT5 L22 + SPACE + aux = {FEAT_DIM}d",
        "holdout": HOLDOUT,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "n_after_r1": n_kept_r1,
        "n_after_r2": n_kept_r2,
        "mlp_config": {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "max_ep": MAX_EP, "patience": PATIENCE},
        "cl_cutoff": CL_CUTOFF,
        "baseline_single_mean": round(float(np.mean([np.mean(p) for p in base_pc_all])), 4),
        "baseline_ensemble": round(base_f1_ens, 4),
        "baseline_per_class": [round(x, 4) for x in base_pc_ens],
        "champion_ensemble": round(champ_f1_ens, 4),
        "champion_per_class": [round(x, 4) for x in champ_pc_ens],
        "champion_per_class_std": [round(x, 4) for x in champ_pc_std],
        "gain_vs_single": round(base_f1_ens - np.mean([np.mean(p) for p in base_pc_all]), 4),
        "cleanlab_gain": round(gain, 4),
        "wall_time_s": round(dt, 1),
        "comparison": {
            "single_mlp_champion": 0.8011,
            "deeploc_accurate": 0.7674,
            "deeploc_fast": 0.7491,
            "vs_single_mlp": round(champ_f1_ens - 0.8011, 4),
            "vs_deeploc_accurate": round(champ_f1_ens - 0.7674, 4),
            "vs_deeploc_fast": round(champ_f1_ens - 0.7491, 4),
        },
    }

    # Per-seed details
    report["per_seed_baseline"] = [round(float(np.mean(p)), 4) for p in base_pc_all]
    report["per_seed_champion"] = [round(float(np.mean(p)), 4) for p in champ_pc_all]

    report_path = PROJ / "output_champion_deep_ensemble.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_path}\n")


if __name__ == "__main__":
    main()
