#!/usr/bin/env python3
"""champion_esm2_sweep_cl.py

Sweeps cleanlab self-confidence cutoff for ESM2 champion pipeline.
Generates OOF predictions ONCE (expensive), then sweeps cutoffs cheaply.
"""

import warnings, json, time, sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings("ignore")

PROJ = Path("/Volumes/BOMBOCLAT/project_JL")
DATA = PROJ / "data"
DEVICE = "cpu"
RSEED = 42

HIDDEN = 256
DROPOUT = 0.5
LR = 1e-4
MAX_EPOCHS = 50
PATIENCE = 5
LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
CUTOFFS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN, out=M, drop=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_tr, Y_tr, X_va, Y_va, in_dim, verbose=False):
    torch.manual_seed(RSEED)
    m = MLP(in_dim).to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=LR)
    pos = (Y_tr.sum(0) / len(Y_tr)).clip(0.05, 0.95)
    pw = torch.from_numpy(((1 - pos) / pos).clip(1, 20).astype(np.float32)).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    X_tt = torch.from_numpy(X_va.astype(np.float32)).to(DEVICE)
    Y_tt = torch.from_numpy(Y_va.astype(np.float32)).to(DEVICE)
    best_f1, best_ep, best_sd = -1, 0, None
    for ep in range(MAX_EPOCHS):
        perm = torch.randperm(len(X_tr))
        m.train()
        for i in range(0, len(X_tr), 256):
            idx = perm[i:i+256]
            bx = torch.from_numpy(X_tr[idx].astype(np.float32)).to(DEVICE)
            by = torch.from_numpy(Y_tr[idx].astype(np.float32)).to(DEVICE)
            opt.zero_grad(); crit(m(bx), by).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            logits = m(X_tt)
            probs = torch.sigmoid(logits).cpu().numpy()
        f1 = float(np.mean([f1_score(Y_va[:, j], (probs[:, j] >= 0.5).astype(int), zero_division=0) for j in range(M)]))
        if f1 > best_f1:
            best_f1, best_ep, best_sd = f1, ep, {k: v.detach().cpu() for k, v in m.state_dict().items()}
        if ep - best_ep > PATIENCE:
            break
    m.load_state_dict(best_sd)
    return m, best_f1


def gen_oof(X, y, folds, in_dim):
    torch.manual_seed(RSEED)
    oof = np.zeros((len(X), M), dtype=np.float32)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RSEED)
    for tr_idx, va_idx in skf.split(X, y[:, 0] + 2*y[:, 2]):
        m, _ = train_mlp(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], in_dim)
        with torch.no_grad():
            oof[va_idx] = torch.sigmoid(m(torch.from_numpy(X[va_idx].astype(np.float32)).to(DEVICE))).cpu().numpy()
    return oof


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("ESM2 CLEANLAB CUTOFF SWEEP")
print("=" * 60)
t0 = time.time()

# 1. Load labels + partition
adi = pd.read_csv(DATA / "df_adi.csv")
Y_all = adi[LABEL_COLS].values.astype(np.float32)
parts = adi["partition"].values
print(f"df_adi: {len(adi)} proteins")

# 2. Load ESM2 last-layer features
with h5py.File(DATA / "esm2_all_layers_dfadi.h5", "r") as h:
    prot5_esm2 = h["df_adi_layer_32"][:].astype(np.float32)
print(f"ESM2 L32: {prot5_esm2.shape}")

# 3. Load SPACE
net_emb = np.load(DATA / "space_network_embeddings.npy")
net_mask = np.load(DATA / "space_network_mask.npy")
net_filled = net_emb.copy()
net_filled[~net_mask] = 0.0

# 4. Load aux
aux_feats = np.load(DATA / "df_adi_aux_features.npy")

# 5. Concat
X_all = np.concatenate([prot5_esm2, net_filled, aux_feats], axis=1).astype(np.float32)
FEAT_DIM = X_all.shape[1]
print(f"Features: {FEAT_DIM} ({prot5_esm2.shape[1]} ESM2 + {net_filled.shape[1]} SPACE + {aux_feats.shape[1]} aux)")

# 6. Train/test split
tr_mask = parts != 4
te_mask = parts == 4
X_tr, Y_tr = X_all[tr_mask], Y_all[tr_mask]
X_te, Y_te = X_all[te_mask], Y_all[te_mask]
n_tr = int(tr_mask.sum())
n_te = int(te_mask.sum())
print(f"Train: {n_tr}  Test: {n_te}")

# 7. Baseline
print(f"\nBaseline...")
baseline_model, baseline_f1 = train_mlp(X_tr, Y_tr, X_te, Y_te, FEAT_DIM)
print(f"  Baseline F1: {baseline_f1:.4f}")

# 8. OOF (expensive - done once)
print(f"\nGenerating OOF predictions (5-fold, expensive)...")
oof_r1 = gen_oof(X_tr, Y_tr, 5, FEAT_DIM)
print(f"  OOF done.")

# 9. Sweep cutoffs
print(f"\n{'='*60}")
print(f"{'Cutoff':>8s}  {'Keep':>8s}  {'Champ F1':>10s}  {'Gain':>8s}  {'Per-comp F1':>40s}")
print(f"{'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*40}")
results = []
for cut in CUTOFFS:
    # R1 cleaning
    Y_int = Y_tr.astype(int)
    conf_r1 = np.where(Y_int == 1, oof_r1, 1 - oof_r1).mean(axis=1)
    keep_r1 = conf_r1 >= cut

    # Train on R1-kept, get OOF for R2
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    oof_r2 = gen_oof(X_r1, Y_r1, 5, FEAT_DIM)

    # R2 cleaning
    conf_r2 = np.where(Y_r1.astype(int) == 1, oof_r2, 1 - oof_r2).mean(axis=1)
    keep_r2 = conf_r2 >= cut

    # Final model - train once, unpack both F1 and model
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    final_model, champ_f1 = train_mlp(X_r2, Y_r2, X_te, Y_te, FEAT_DIM)
    final_model.eval()
    with torch.no_grad():
        te_probs = torch.sigmoid(final_model(torch.from_numpy(X_te.astype(np.float32)).to(DEVICE))).cpu().numpy()
    per_f1 = [f1_score(Y_te[:, j].astype(int), (te_probs[:, j] >= 0.5).astype(int), zero_division=0) for j in range(M)]

    gain = champ_f1 - baseline_f1
    n_keep = int(keep_r2.sum())
    pct_kept = 100 * keep_r2.mean()
    pc_str = " ".join(f"{x:.3f}" for x in per_f1)
    print(f"{cut:>8.2f}  {n_keep:>4d}/{n_tr:<3d}  {champ_f1:>10.4f}  {gain:>+8.4f}  {pc_str}")
    results.append({
        "cutoff": cut,
        "n_initial": n_tr,
        "n_after_r1": int(keep_r1.sum()),
        "n_after_r2": n_keep,
        "pct_kept": round(pct_kept, 1),
        "champion_f1": round(champ_f1, 4),
        "gain": round(gain, 4),
        "per_class_f1": [round(float(x), 4) for x in per_f1],
    })

# Best
best = max(results, key=lambda r: r["champion_f1"])
print(f"\n{'='*60}")
print(f"  BEST: cutoff={best['cutoff']:.2f}  champion F1={best['champion_f1']:.4f}  gain={best['gain']:+.4f}")
print(f"  Kept {best['n_after_r2']}/{best['n_initial']} ({best['pct_kept']:.0f}%)")
print(f"\n  Per-compartment at best cutoff:")
for j, c in enumerate(LABEL_COLS):
    print(f"    {c:>18s}: {best['per_class_f1'][j]:.4f}")

# Save
with open(PROJ / "output_champion_esm2_sweep_cl.json", "w") as f:
    json.dump({"baseline_f1": round(baseline_f1, 4), "results": results, "best": best}, f, indent=2)
print(f"\nSaved: output_champion_esm2_sweep_cl.json")
print(f"Wall time: {time.time() - t0:.0f}s")
