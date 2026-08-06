#!/usr/bin/env python3
"""champion_per_comp.py — Per-compartment binary MLPs on PCA 100d.

Instead of one shared 100→256→7 multi-label MLP, trains 7 separate
100→32→1 binary classifiers — one per compartment.

Each compartment gets its own feature transformation, pos_weight,
and early stopping. This avoids the shared-representation bottleneck:
weak compartments (Cell_surf, Endom, Mito) no longer compete with
dominant ones (Nucleus, Extracellular) for hidden-layer capacity.

Pipeline: ProtT5 attn L20-23 + SPACE + aux → PCA 100d → per-comp cleanlab
          2-pass on reconstructed multi-label OOF → 7 binary MLPs → macro F1

Usage:
  python3 champion_per_comp.py

Output:
  output_cleaning_sweep/per_comp_result.json
"""

import os, time, warnings, json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cleanlab.multilabel_classification.rank import get_label_quality_scores

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.resolve()
SRC_CSV    = PROJ / "data" / "df_adi.csv"
PROTT5_H5  = str(PROJ / "data" / "prott5_attn_all_layers.h5")
SPACE_EMB  = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS  = PROJ / "data" / "df_adi_aux_features.npy"

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]

# Per-comp MLP (smaller — no shared hidden)
HIDDEN_PER = 32   # per-compartment hidden dim
DROPOUT = 0.3     # slightly lower dropout for smaller net
LR = 1e-3         # higher lr for faster per-comp convergence
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5; CL_CUTOFF = 0.40
N_COMPONENTS = 100


class BinaryMLP(nn.Module):
    """Tiny binary classifier: indim → 32 → 1."""
    def __init__(self, indim, hdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)  # (N,)


def train_binary(Xtr, ytr, Xte, yte, seed=42):
    """Train binary MLP for one compartment. Returns (f1, probs)."""
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr).astype(np.float32)
    Xtes = sc.transform(Xte).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    ti, ei = train_test_split(np.arange(len(Xts)), test_size=ES_FRAC, random_state=seed)
    
    # Per-class pos_weight for this binary task
    pos = float(ytr.sum()); neg = float(len(ytr)) - pos
    pw = 1.0 if pos <= 0 else min(20.0, neg / pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw))
    
    model = BinaryMLP(Xts.shape[1], HIDDEN_PER, DROPOUT)
    opt = optim.Adam(model.parameters(), lr=LR)
    Xt = torch.from_numpy(Xts); yt = torch.from_numpy(ytr.astype(np.float32))
    Xe = torch.from_numpy(Xts[ei]); ye = ytr[ei]
    best_f1, best_state, stall = -1.0, None, 0
    for ep in range(1, MAX_EP+1):
        model.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), BATCH_SIZE):
            ix = perm[s:s+BATCH_SIZE]
            criterion(model(Xt[ix]), yt[ix]).backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            ep_ = torch.sigmoid(model(Xe)).numpy()
        ef = float(f1_score(ye.astype(int), (ep_ >= THR).astype(int), zero_division=0))
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
    f1_v = float(f1_score(yte.astype(int), pr, zero_division=0))
    return f1_v, tp


def train_shared(Xtr, Ytr, Xte, Yte):
    """Train shared multi-label MLP (baseline comparison).
    Same architecture as original champion: 100→256→7.
    """
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr).astype(np.float32)
    Xtes = sc.transform(Xte).astype(np.float32)
    torch.manual_seed(42); np.random.seed(42)
    ti, ei = train_test_split(np.arange(len(Xts)), test_size=ES_FRAC, random_state=42)
    
    pw = np.ones(M, dtype=np.float32)
    for j in range(M):
        pos = float(Ytr[:, j].sum()); neg = float(len(Ytr)) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg / pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw.astype(np.float32)))
    model = nn.Sequential(nn.Linear(Xts.shape[1], 256), nn.ReLU(True),
                          nn.Dropout(0.5), nn.Linear(256, M))
    opt = optim.Adam(model.parameters(), lr=1e-4)
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
        else: stall += 1
        if stall >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    pr = (tp >= THR).astype(int)
    pc = [float(f1_score(Yte[:,j].astype(int), pr[:,j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc, tp


def gen_oof_per_comp(X, Y):
    """Generate per-compartment OOF using 4-fold CV for each binary classifier.
    
    Returns: oof of shape (n, M) — reconstructed from 7 per-comp OOF vectors.
    """
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(42); idx = np.arange(n); rng.shuffle(idx)
    fs = n // 4
    for j in range(M):
        yj = Y[:, j]
        oof_j = np.zeros(n, dtype=np.float32)
        print(f"        OOF [{COMPARTMENTS[j]:>12s}]...", end="", flush=True)
        for f in range(4):
            vs = f*fs; ve = n if f == 3 else (f+1)*fs
            vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
            _, tp = train_binary(X[ti], yj[ti], X[vi], yj[vi], seed=42+f)
            oof_j[vi] = tp
        oof[:, j] = oof_j
        f1_j = float(f1_score(yj.astype(int), (oof_j >= THR).astype(int), zero_division=0))
        print(f"  F1={f1_j:.4f}")
    return oof


def cleanlab_step(Y, oof, cutoff):
    labs = [list(np.where(Y[i]==1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"        Cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)")
    return keep


def load_data():
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    
    # ProtT5 attn L20-23 (4096d)
    pt_layers = []
    with h5py.File(PROTT5_H5, "r") as f:
        for lyr in [20, 21, 22, 23]:
            pt_layers.append(f[f"attn_layer_{lyr:02d}"][:].astype(np.float32))
    prot5 = np.concatenate(pt_layers, axis=1)
    
    # SPACE + aux
    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    aux = np.load(AUX_FEATS)
    
    X_all = np.concatenate([prot5, net_filled, aux], axis=1).astype(np.float32)
    print(f"      Features: {X_all.shape}  ({X_all.shape[1]}d)")
    return X_all, Y_all, parts


def run_fold(X_tr, Y_tr, X_te, Y_te, holdout):
    print(f"\n  {'='*50}\n  HOLD-OUT PARTITION {holdout}\n  {'='*50}")
    
    # PCA
    pca = PCA(n_components=N_COMPONENTS)
    X_tr = pca.fit_transform(X_tr).astype(np.float32)
    X_te = pca.transform(X_te).astype(np.float32)
    print(f"      PCA → {N_COMPONENTS}d (var: {pca.explained_variance_ratio_.sum():.4f})")
    
    # ── Shared baseline (for comparison) ──
    print(f"  Shared baseline ({len(Y_tr)} train, {len(Y_te)} test)...")
    base_f1, base_pc, _ = train_shared(X_tr, Y_tr, X_te, Y_te)
    print(f"      Shared macro F1: {base_f1:.4f}")
    
    # ── Per-compartment baseline ──
    print(f"  Per-comp baseline...")
    base_pc_comp = []
    for j in range(M):
        f1_j, _ = train_binary(X_tr, Y_tr[:, j], X_te, Y_te[:, j], seed=42)
        base_pc_comp.append(f1_j)
    base_f1_comp = float(np.mean(base_pc_comp))
    print(f"      Per-comp macro F1: {base_f1_comp:.4f}")
    for j, c in enumerate(COMPARTMENTS):
        print(f"        {c:>12s}: {base_pc_comp[j]:.4f}")
    
    # ── Round 1 OOF + cleanlab (per-compartment OOF) ──
    print(f"  Round 1 OOF (per-comp)...")
    oof_r1 = gen_oof_per_comp(X_tr, Y_tr)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    
    # ── Round 2 OOF + cleanlab ──
    print(f"  Round 2 OOF (per-comp)...")
    oof_r2 = gen_oof_per_comp(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    
    # ── Final: per-compartment models on cleaned data ──
    print(f"  Final per-comp ({len(Y_r2)} train)...")
    final_pc = []
    for j in range(M):
        f1_j, _ = train_binary(X_r2, Y_r2[:, j], X_te, Y_te[:, j], seed=42)
        final_pc.append(f1_j)
    final_f1 = float(np.mean(final_pc))
    
    print(f"\n  Shared champion F1: {base_f1:.4f}")
    print(f"  Per-comp champion F1: {final_f1:.4f}")
    print(f"  Per-comp improvement vs shared: {final_f1-base_f1:+.4f}")
    
    return {"holdout": holdout, "shared_baseline": base_f1, "baseline": base_f1_comp,
            "champion_f1": round(final_f1, 4),
            "baseline_per_class": [round(x,4) for x in base_pc_comp],
            "champion_per_class": [round(x,4) for x in final_pc],
            "shared_baseline_per_class": [round(x,4) for x in base_pc]}


def main():
    t0 = time.time()
    print("=" * 72)
    print("  PER-COMPARTMENT CHAMPION — PCA 100d")
    print("  7 separate binary MLPs (100→32→1) vs 1 shared (100→256→7)")
    print("=" * 72)
    
    X_all, Y_all, parts = load_data()
    
    results = []
    for holdout in range(5):
        train_mask = (parts != holdout); test_mask = (parts == holdout)
        r = run_fold(X_all[train_mask], Y_all[train_mask],
                     X_all[test_mask], Y_all[test_mask], holdout)
        results.append(r)
    
    # Summary
    print(f"\n  {'='*50}\n  COMPARISON: Per-comp vs Shared\n  {'='*50}")
    print(f"  {'Holdout':>8}  {'Shared F1':>10s}  {'PerComp F1':>11s}  {'Δ':>8s}")
    shared_mean = 0; percomp_mean = 0
    for r in results:
        shared_mean += r["shared_baseline"]; percomp_mean += r["champion_f1"]
        print(f"  P{r['holdout']:>7}  {r['shared_baseline']:>10.4f}  {r['champion_f1']:>11.4f}  "
              f"{r['champion_f1']-r['shared_baseline']:>+8.4f}")
    shared_mean /= 5; percomp_mean /= 5
    print(f"  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*8}")
    print(f"  {'Mean':>8}  {shared_mean:>10.4f}  {percomp_mean:>11.4f}  "
          f"{percomp_mean-shared_mean:>+8.4f}")
    
    print(f"\n  Per-compartment final F1 (mean across folds):")
    print(f"  {'Compartment':>15s}  {'Shared':>8s}  {'Per-comp':>9s}  {'Δ':>8s}")
    for j, c in enumerate(COMPARTMENTS):
        svals = [r["shared_baseline_per_class"][j] for r in results]
        pvals = [r["champion_per_class"][j] for r in results]
        sm = np.mean(svals); pm = np.mean(pvals)
        print(f"  {c:>15s}  {sm:>8.4f}  {pm:>9.4f}  {pm-sm:>+8.4f}")
    
    print(f"\n  Wall time: {time.time()-t0:.1f}s")
    
    report = {"features": "ProtT5_attn_L20-23+SPACE+aux_PCA100d",
              "architecture": "per_compartment_binary_mlp",
              "per_fold": results,
              "shared_mean": round(float(shared_mean), 4),
              "per_comp_mean": round(float(percomp_mean), 4),
              "improvement": round(float(percomp_mean - shared_mean), 4)}
    
    out_dir = PROJ / "output_cleaning_sweep"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "per_comp_result.json").write_text(json.dumps(report, indent=2))
    print(f"\n  Saved: output_cleaning_sweep/per_comp_result.json")


if __name__ == "__main__":
    main()
