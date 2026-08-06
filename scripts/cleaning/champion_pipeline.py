#!/usr/bin/env python3
"""champion_pipeline.py

Standalone end-to-end champion pipeline for subcellular localization.
  ProtT5 L22 + SPACE → iterative cleanlab → MLP → 0.7907 Macro F1

Usage:
  python3 champion_pipeline.py

Requires:
  data/df_adi.csv
  data/prott5_all_layers_dfadi-3.h5  (ProtT5 all-layer embeddings)
  data/space_network_embeddings.npy   (SPACE 512-d embeddings)
  data/space_network_mask.npy         (SPACE coverage mask)
"""

import json, os, time, warnings
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cleanlab.multilabel_classification.rank import get_label_quality_scores

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────
PROJ = Path(__file__).parent.resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
PROT5_H5 = str(PROJ / "data" / "prott5_all_layers_dfadi-3.h5")
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"

# ── Hyperparameters (sweep-optimised) ──────────────────────
LAYER = 22              # ProtT5 layer to use
HIDDEN = 512            # MLP hidden units
DROPOUT = 0.5           # Sweep-best: 0.5 (was 0.3)
LR = 1e-4               # Sweep-best: 1e-4 (was 1e-3)
MAX_EP = 50
PATIENCE = 5
BATCH_SIZE = 256
ES_FRAC = 0.10          # Early-stopping validation fraction
THRESHOLD = 0.5         # Binary prediction threshold
CL_CUTOFF = 0.40        # Cleanlab self-confidence cutoff

LABEL_COLS = ["membrane", "cytoplasm", "nucleus", "extracellular",
              "cell_surface", "mitochondrion", "endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane", "Cytoplasm", "Nucleus", "Extracell",
                "Cell_surf", "Mito", "Endom"]


# ── MLP ────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, hdim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hdim, outdim),
        )

    def forward(self, x):
        return self.net(x)


def compute_pos_weight(Y):
    """Per-class positive weight for BCE loss."""
    pw = np.ones(M, dtype=np.float32)
    for j in range(M):
        pos = float(Y[:, j].sum())
        neg = float(Y.shape[0]) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg / pos)
    return np.clip(pw, 1.0, 20.0)


def train_mlp(Xtr, Ytr, Xte, Yte, seed=42):
    """
    Train MLP with early stopping. Returns macro F1, per-class F1, and test probs.
    """
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr).astype(np.float32)
    Xte_s = sc.transform(Xte).astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Validation split
    idx_all = np.arange(len(Xtr_s))
    ti, ei = train_test_split(idx_all, test_size=ES_FRAC, random_state=seed)

    pw = compute_pos_weight(Ytr)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pw.astype(np.float32))
    )

    model = MLP(Xtr_s.shape[1], HIDDEN, M, DROPOUT)
    optimiser = optim.Adam(model.parameters(), lr=LR)

    X_t = torch.from_numpy(Xtr_s)
    Y_t = torch.from_numpy(Ytr.astype(np.float32))
    X_e = torch.from_numpy(Xtr_s[ei])
    Y_e = Ytr[ei]

    best_f1 = -1.0
    best_state = None
    stall = 0

    for epoch in range(1, MAX_EP + 1):
        model.train()
        perm = torch.randperm(len(ti))
        for start in range(0, len(ti), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            loss = criterion(model(X_t[idx]), Y_t[idx])
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        # Early stopping check
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(X_e)).numpy()
        val_f1 = float(np.mean([
            f1_score(Y_e[:, j].astype(int),
                     (preds[:, j] >= THRESHOLD).astype(int),
                     zero_division=0)
            for j in range(M)
        ]))

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= PATIENCE:
                break

    # Restore best state and predict on test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(
            model(torch.from_numpy(Xte_s))
        ).numpy().astype(np.float32)

    test_preds = (test_probs >= THRESHOLD).astype(int)
    per_class = [
        float(f1_score(Yte[:, j].astype(int), test_preds[:, j],
                        zero_division=0))
        for j in range(M)
    ]
    macro = float(np.mean(per_class))
    return macro, per_class, test_probs


# ── OOF generation ─────────────────────────────────────────
def generate_oof(X, Y, n_folds=4, seed=42):
    """Generate 4-fold out-of-fold predictions."""
    n = len(X)
    oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    fold_size = n // n_folds

    for fold in range(n_folds):
        vs = fold * fold_size
        ve = n if fold == n_folds - 1 else (fold + 1) * fold_size
        vi = idx[vs:ve]
        ti = np.concatenate([idx[:vs], idx[ve:]])
        _, _, tp = train_mlp(X[ti], Y[ti], X[vi], Y[vi], seed=seed + fold)
        oof[vi] = tp
        print(f"      [Fold {fold + 1}/{n_folds}] F1 = {np.mean([
            f1_score(Y[vi][:, j].astype(int),
                     (tp[:, j] >= THRESHOLD).astype(int),
                     zero_division=0)
            for j in range(M)
        ]):.4f}", flush=True)

    return oof


# ── Cleanlab filtering ─────────────────────────────────────
def cleanlab_filter(Y, oof, cutoff):
    """Return keep mask from cleanlab self-confidence scores."""
    labels = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(
        labels=labels,
        pred_probs=oof.astype(np.float64),
        method="self_confidence",
        adjust_pred_probs=True,
    )
    keep = scores >= cutoff
    n_kept = int(keep.sum())
    n_dropped = int((~keep).sum())
    print(f"      Cleanlab: {n_kept} kept, {n_dropped} dropped "
          f"({100 * n_dropped / len(Y):.1f}%)")
    return keep


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 70)
    print("  CHAMPION PIPELINE — Subcellular Localisation")
    print("  ProtT5 L22 + SPACE → Iterative Cleanlab → MLP")
    print("=" * 70)

    # ── 1. Load data ──
    print(f"\n  [1/7] Loading labels...")
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != 4)
    test_mask = (parts == 4)
    print(f"      Total: {len(Y_all)}  Train: {train_mask.sum()}  "
          f"Test: {test_mask.sum()}")

    # ── 2. Load ProtT5 L22 ──
    print(f"  [2/7] Loading ProtT5 L{LAYER}...")
    with h5py.File(PROT5_H5, "r") as f:
        prot5 = f[f"df_adi_layer_{LAYER:02d}"][:].astype(np.float32)
    print(f"      Shape: {prot5.shape}")

    # ── 3. Load SPACE + concat ──
    print(f"  [3/7] Loading SPACE network embeddings...")
    net_emb = np.load(SPACE_EMB)
    net_mask = np.load(SPACE_MASK)
    # Zero-pad missing embeddings
    net_filled = net_emb.copy()
    net_filled[~net_mask] = 0.0
    X_all = np.concatenate([prot5, net_filled], axis=1).astype(np.float32)
    n_missing = int((~net_mask).sum())
    print(f"      Concat: {X_all.shape}  "
          f"(zero-padded {n_missing} / {100 * n_missing / len(net_mask):.1f}%)")

    X_tr = X_all[train_mask]
    Y_tr = Y_all[train_mask]
    X_te = X_all[test_mask]
    Y_te = Y_all[test_mask]

    # ── 4. Baseline ──
    print(f"  [4/7] Training baseline (no cleaning)...")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)
    print(f"      Baseline Macro F1 = {base_f1:.4f}")
    print(f"      Per-class: {', '.join(f'{c}={v:.4f}' for c, v in zip(COMPARTMENTS, base_pc))}")

    # ── 5. Round 1: OOF + cleanlab ──
    print(f"  [5/7] Round 1: 4-fold OOF + cleanlab...")
    oof_r1 = generate_oof(X_tr, Y_tr)
    keep_r1 = cleanlab_filter(Y_tr, oof_r1, CL_CUTOFF)

    X_r1 = X_tr[keep_r1]
    Y_r1 = Y_tr[keep_r1]

    # ── 6. Round 2: Fresh OOF + cleanlab again ──
    print(f"  [6/7] Round 2: Fresh OOF + cleanlab again...")
    oof_r2 = generate_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_filter(Y_r1, oof_r2, CL_CUTOFF)

    X_r2 = X_r1[keep_r2]
    Y_r2 = Y_r1[keep_r2]
    print(f"      Cumulative: {len(Y_tr)} → {len(Y_r1)} → {len(Y_r2)} "
          f"({100 * (len(Y_tr) - len(Y_r2)) / len(Y_tr):.1f}% total drop)")

    # ── 7. Final training + evaluation ──
    print(f"  [7/7] Training final model on cleaned set...")
    final_f1, final_pc, _ = train_mlp(X_r2, Y_r2, X_te, Y_te)
    gain = final_f1 - base_f1

    # ── Report ──
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  {'Metric':>30s}  {'Value':>10}")
    print(f"  {'-' * 30}  {'-' * 10}")
    print(f"  {'Baseline (no cleaning)':>30s}  {base_f1:>10.4f}")
    print(f"  {'Champion (iterative cleanlab)':>30s}  {final_f1:>10.4f}")
    print(f"  {'Gain from cleaning':>30s}  {gain:>+10.4f}")
    print(f"  {'Training samples (final)':>30s}  {len(Y_r2):>10}")
    print(f"  {'Drop rate':>30s}  {100 * (len(Y_tr) - len(Y_r2)) / len(Y_tr):>9.1f}%")
    print()
    print(f"  Per-class F1 (champion):")
    for c, v in zip(COMPARTMENTS, final_pc):
        gain_pc = v - base_pc[COMPARTMENTS.index(c)]
        print(f"    {c:>15s}:  {v:.4f}  ({gain_pc:+.4f})")

    print(f"\n  Wall time: {time.time() - t0:.1f}s")
    print("=" * 70)

    # ── Save report ──
    report = {
        "pipeline": "ProtT5 L22 + SPACE → iterative cleanlab → MLP",
        "baseline_macro_f1": round(base_f1, 4),
        "champion_macro_f1": round(final_f1, 4),
        "gain": round(gain, 4),
        "n_train_initial": int(len(Y_tr)),
        "n_after_r1": int(len(Y_r1)),
        "n_after_r2": int(len(Y_r2)),
        "n_test": int(len(Y_te)),
        "cleanlab_cutoff": CL_CUTOFF,
        "mlp_params": {
            "lr": LR, "dropout": DROPOUT, "hidden": HIDDEN,
            "max_epochs": MAX_EP, "patience": PATIENCE,
        },
        "per_class_f1_baseline": [round(x, 4) for x in base_pc],
        "per_class_f1_champion": [round(x, 4) for x in final_pc],
        "compartments": LABEL_COLS,
    }
    report_path = PROJ / "output_champion_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
