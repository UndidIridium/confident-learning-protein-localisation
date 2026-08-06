#!/usr/bin/env python3
"""save_cleaned_data.py — Phase 1: run 2-round cleanlab once, save to disk."""

import os, warnings, time
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)

import h5py, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch.nn as nn, torch.optim as optim
from cleanlab.multilabel_classification.rank import get_label_quality_scores

PROJ = Path(__file__).parent.resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
PROT5_H5 = str(PROJ / "data" / "prott5_all_layers_dfadi-3.h5")
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"

LAYER = 22; HOLDOUT = 4; CL_CUTOFF = 0.40; M = 7; SEED = 42
HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4; MAX_EP = 50; PATIENCE = 5
BS = 256; ES_FRAC = 0.10

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]

class MLP(nn.Module):
    def __init__(self, i, h, o, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i,h), nn.ReLU(True), nn.Dropout(d), nn.Linear(h,o))
    def forward(self, x): return self.net(x)

def posw(Y):
    pw = np.ones(M, np.float32)
    for j in range(M):
        pos = float(Y[:,j].sum()); neg = float(Y.shape[0]) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg/pos)
    return np.clip(pw, 1.0, 20.0)

def train_mlp(Xtr, Ytr, Xte, Yte, seed=42):
    sc = StandardScaler(); Xts = sc.fit_transform(Xtr).astype(np.float32); Xtes = sc.transform(Xte).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    ti, ei = train_test_split(np.arange(len(Xts)), test_size=ES_FRAC, random_state=seed)
    pw = posw(Ytr); model = MLP(Xts.shape[1], HIDDEN, M, DROPOUT)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw.astype(np.float32)))
    Xt = torch.from_numpy(Xts); Yt = torch.from_numpy(Ytr.astype(np.float32))
    Xe = torch.from_numpy(Xts[ei]); Ye = Ytr[ei]
    best_f1, best_state, stall = -1.0, None, 0
    for ep in range(1, MAX_EP+1):
        model.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), BS):
            ix = perm[s:s+BS]; crit(model(Xt[ix]), Yt[ix]).backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): ep_ = torch.sigmoid(model(Xe)).numpy()
        ef = float(np.mean([f1_score(Ye[:,j].astype(int),(ep_[:,j]>=0.5).astype(int),zero_division=0) for j in range(M)]))
        if ef > best_f1 + 1e-6: best_f1 = ef; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stall = 0
        else: stall += 1
        if stall >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    pc = [float(f1_score(Yte[:,j].astype(int),(tp[:,j]>=0.5).astype(int),zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc, tp

def gen_oof(X, Y, n_folds=4, seed=42, label=""):
    n = len(X); oof = np.zeros((n,M), np.float32)
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx); fs = n//n_folds
    for f in range(n_folds):
        vs = f*fs; ve = n if f==n_folds-1 else (f+1)*fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs],idx[ve:]])
        t0f = time.time()
        _, _, tp = train_mlp(X[ti], Y[ti], X[vi], Y[vi], seed=seed+f)
        f1_f = float(np.mean([f1_score(Y[vi][:,j].astype(int),(tp[:,j]>=0.5).astype(int),zero_division=0) for j in range(M)]))
        print(f"    {label}[Fold {f+1}/{n_folds}] F1={f1_f:.4f}  [{time.time()-t0f:.0f}s]", flush=True)
        oof[vi] = tp
    return oof

def cleanlab_step(Y, oof, cutoff):
    labs = [list(np.where(Y[i]==1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"    Cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)", flush=True)
    return keep

t0 = time.time()
print("=" * 60, flush=True)
print("  PHASE 1: Save cleaned training data", flush=True)
print("=" * 60, flush=True)

# Load
src = pd.read_csv(SRC_CSV)
Y_all = src[LABEL_COLS].values.astype(np.int64)
parts = src["partition"].to_numpy()
train_mask = (parts != HOLDOUT); test_mask = (parts == HOLDOUT)
n_tr = int(train_mask.sum()); n_te = int(test_mask.sum())
print(f"Train: {n_tr} (P0-P3)  Test: {n_te} (P4)", flush=True)

with h5py.File(PROT5_H5, "r") as f:
    prot5 = f[f"df_adi_layer_{LAYER:02d}"][:].astype(np.float32)
net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
aux = np.load(AUX_FEATS)
X_all = np.concatenate([prot5, net_filled, aux], axis=1).astype(np.float32)
X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
X_te, Y_te = X_all[test_mask], Y_all[test_mask]

# 2-round cleanlab
print("\nRound 1 OOF...", flush=True)
oof_r1 = gen_oof(X_tr, Y_tr, label="R1 ")
keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
print(f"R1: {len(Y_r1)} kept ({100*len(Y_r1)/n_tr:.1f}%)", flush=True)

print("\nRound 2 OOF...", flush=True)
oof_r2 = gen_oof(X_r1, Y_r1, label="R2 ")
keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
print(f"R2: {len(Y_r2)} kept ({100*len(Y_r2)/len(Y_r1):.1f}%) — "
      f"total {100*len(Y_r2)/n_tr:.1f}%", flush=True)

# Save
out_dir = PROJ / "cleaned_data"
out_dir.mkdir(exist_ok=True)
np.save(out_dir / "X_train_cleaned.npy", X_r2)
np.save(out_dir / "Y_train_cleaned.npy", Y_r2)
np.save(out_dir / "X_test.npy", X_te)
np.save(out_dir / "Y_test.npy", Y_te)
print(f"\nSaved to {out_dir}/", flush=True)
print(f"  X_train: {X_r2.shape}  Y_train: {Y_r2.shape}", flush=True)
print(f"  X_test:  {X_te.shape}  Y_test:  {Y_te.shape}", flush=True)
print(f"Wall time: {(time.time()-t0)/60:.1f}m", flush=True)
