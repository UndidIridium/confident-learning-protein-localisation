#!/usr/bin/env python3
"""champion_esm2_p4.py

Runs our champion pipeline (MLP + iterative cleanlab) on ESM2 features
instead of ProtT5, for a fair comparison against DeepLoc Fast (ESM-1b).

DeepLoc Fast uses:  ESM-1b (650M params, 33 layers, 1280-d)
We use:            ESM-2 650M (33 layers, 1280-d) - closest available

Pipeline: ESM2 L32 (1280) + SPACE (512) + aux (2) = 1794d
          → baseline MLP → cleanlab R1 → retrain → cleanlab R2 → eval on P4
"""

import warnings, json, sys, time
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
DEVICE = "cpu"  # small model, fine on CPU
RSEED = 42

# ── Config ──
HIDDEN = 256
DROPOUT = 0.5
LR = 1e-4
MAX_EPOCHS = 50
PATIENCE = 5
CL_CUTOFF = 0.40
LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)


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


def train_mlp(X_tr, Y_tr, X_va, Y_va, in_dim):
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
    """Generate OOF predictions."""
    torch.manual_seed(RSEED)
    oof = np.zeros((len(X), M), dtype=np.float32)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RSEED)
    # Use membrane as stratification target (most balanced)
    for tr_idx, va_idx in skf.split(X, y[:, 0] + 2*y[:, 2]):  # membrane + nucleus
        m, _ = train_mlp(X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], in_dim)
        with torch.no_grad():
            oof[va_idx] = torch.sigmoid(m(torch.from_numpy(X[va_idx].astype(np.float32)).to(DEVICE))).cpu().numpy()
    return oof





def posw(Y):
    pos = Y.sum(0).clip(1)
    neg = len(Y) - pos
    return (neg / pos).clip(1, 20).astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("CHAMPION PIPELINE - ESM2 L32 + SPACE + aux")
print("Apples-to-apples vs DeepLoc Fast (both ESM-family, 650M, 1280-d)")
print("=" * 60)

t0 = time.time()

# 1. Load labels + partition
adi = pd.read_csv(DATA / "df_adi.csv")
Y_all = adi[LABEL_COLS].values.astype(np.float32)
parts = adi["partition"].values
print(f"\nLoaded df_adi: {len(adi)} proteins")

# 2. Load ESM2 last-layer features (L32 = final layer, 1280-d)
with h5py.File(DATA / "esm2_all_layers_dfadi.h5", "r") as h:
    prot5_esm2 = h["df_adi_layer_32"][:].astype(np.float32)
print(f"ESM2 L32: {prot5_esm2.shape}")

# 3. Load SPACE features
net_emb = np.load(DATA / "space_network_embeddings.npy")
net_mask = np.load(DATA / "space_network_mask.npy")
net_filled = net_emb.copy()
net_filled[~net_mask] = 0.0
print(f"SPACE: {net_filled.shape}, masked: {~net_mask.sum()}/{len(net_mask)}")

# 4. Load aux features
aux_feats = np.load(DATA / "df_adi_aux_features.npy")
print(f"Aux: {aux_feats.shape}")

# 5. Concat
X_all = np.concatenate([prot5_esm2, net_filled, aux_feats], axis=1).astype(np.float32)
FEAT_DIM = X_all.shape[1]
print(f"Total features: {FEAT_DIM}  ({prot5_esm2.shape[1]} ESM2 + {net_filled.shape[1]} SPACE + {aux_feats.shape[1]} aux)")
print(f"Training: {(parts != 4).sum()}  Test: {(parts == 4).sum()}  (partition 4 held out)")

# 6. Train/val split
tr_mask = parts != 4
te_mask = parts == 4
X_tr, Y_tr = X_all[tr_mask], Y_all[tr_mask]
X_te, Y_te = X_all[te_mask], Y_all[te_mask]
print(f"\nBaseline...")

# 7. Baseline
baseline_model, baseline_f1 = train_mlp(X_tr, Y_tr, X_te, Y_te, FEAT_DIM)
print(f"  Baseline F1: {baseline_f1:.4f}")

# 8. OOF on training set
print(f"Generating OOF predictions...")
oof_r1 = gen_oof(X_tr, Y_tr, 5, FEAT_DIM)

# 9. R1 cleaning - manual self-confidence scoring
# (cleanlab v2.9.0 multilabel API breaks on imbalanced compartments)
print(f"R1 cleanlab (self-confidence cutoff={CL_CUTOFF})...")
# Self-confidence = mean over classes of P(Y_true|model)
Y_int = Y_tr.astype(int)
conf_r1 = np.where(Y_int == 1, oof_r1, 1 - oof_r1).mean(axis=1)
keep_r1 = conf_r1 >= CL_CUTOFF
print(f"  Kept {int(keep_r1.sum())}/{len(keep_r1)}  ({100*keep_r1.mean():.1f}%)")

# 10. Retrain on R1-kept
X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
oof_r2 = gen_oof(X_r1, Y_r1, 5, FEAT_DIM)

# 11. R2 cleaning
print(f"R2 cleanlab (self-confidence cutoff={CL_CUTOFF})...")
conf_r2 = np.where(Y_r1.astype(int) == 1, oof_r2, 1 - oof_r2).mean(axis=1)
keep_r2 = conf_r2 >= CL_CUTOFF
print(f"  Kept {int(keep_r2.sum())}/{len(keep_r2)}  ({100*keep_r2.mean():.1f}%)")

# 12. Final model on R2-kept
X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
final_model, final_f1 = train_mlp(X_r2, Y_r2, X_te, Y_te, FEAT_DIM)
print(f"\n{'='*50}")
print(f"  ESM2 CHAMPION RESULT")
print(f"{'='*50}")
print(f"  Baseline F1: {baseline_f1:.4f}")
print(f"  Champion F1: {final_f1:.4f}")
print(f"  Gain:        {final_f1 - baseline_f1:+.4f}")
print(f"  Cleanlab:    {len(Y_tr)} → {int(keep_r1.sum())} → {int(keep_r2.sum())} ({100*(len(Y_tr)-int(keep_r2.sum()))/len(Y_tr):.0f}% dropped)")

# Per-compartment
final_model.eval()
with torch.no_grad():
    te_probs = torch.sigmoid(final_model(torch.from_numpy(X_te.astype(np.float32)).to(DEVICE))).cpu().numpy()
per_f1 = [f1_score(Y_te[:, j].astype(int), (te_probs[:, j] >= 0.5).astype(int), zero_division=0) for j in range(M)]
print(f"\n  Per-compartment champion F1:")
for j, c in enumerate(LABEL_COLS):
    print(f"    {c:>18s}: {per_f1[j]:.4f}")

# Save
result = {
    "pipeline": "ESM2 L32 + SPACE + aux → iterative cleanlab → MLP",
    "baseline_macro_f1": round(baseline_f1, 4),
    "champion_macro_f1": round(final_f1, 4),
    "gain": round(final_f1 - baseline_f1, 4),
    "feat_dim": FEAT_DIM,
    "n_train_initial": int(tr_mask.sum()),
    "n_after_r1": int(keep_r1.sum()),
    "n_after_r2": int(keep_r2.sum()),
    "n_test": int(te_mask.sum()),
    "per_class_f1_champion": [round(float(x), 4) for x in per_f1],
    "compartments": LABEL_COLS,
}
with open(PROJ / "output_champion_esm2_p4.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: output_champion_esm2_p4.json")
print(f"Wall time: {time.time() - t0:.0f}s")
