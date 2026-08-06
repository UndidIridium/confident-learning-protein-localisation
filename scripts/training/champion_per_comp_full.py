#!/usr/bin/env python3
"""champion_per_comp_full.py — Per-compartment binary MLPs on FULL 4610d.

Tests whether separate feature representations help weak compartments
(Mito, Endom, Cell_surf) on the full-dimensional features.

Architecture: 7 × BinaryMLP(4610→128→1) vs 1 × SharedMLP(4610→512→7)

Pipeline: ProtT5 attn L20-23 + SPACE + aux (4610d)
  → per-compartment OOF (4-fold × 7 compartments)
  → cleanlab 2-pass on reconstructed multi-label OOF
  → 7 binary MLPs on clean data → macro F1

Expected runtime: ~70 min (7× more MLPs than shared champion).

Usage:
  python3 champion_per_comp_full.py

Output:
  output_cleaning_sweep/per_comp_full_result.json
  output_cleaning_sweep/per_comp_full_vs_shared.json  (comparison table)
"""

import os, time, warnings, json
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
SRC_CSV    = PROJ / "data" / "df_adi.csv"
PROTT5_H5  = str(PROJ / "data" / "prott5_attn_all_layers.h5")
SPACE_EMB  = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS  = PROJ / "data" / "df_adi_aux_features.npy"

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]

# Per-compartment MLP config
HIDDEN_PER = 128     # per-compartment hidden dim (smaller for speed)
DROPOUT = 0.3
LR = 1e-3
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 512; ES_FRAC = 0.10
THR = 0.5; CL_CUTOFF = 0.40

# Shared MLP config (for comparison baseline)
SHARED_HIDDEN = 512; SHARED_DROPOUT = 0.5; SHARED_LR = 1e-4


class BinaryMLP(nn.Module):
    """Per-compartment binary classifier: indim → hidden → 1."""
    def __init__(self, indim, hdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


class SharedMLP(nn.Module):
    """Shared multi-label MLP (champion architecture)."""
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, outdim))
    def forward(self, x):
        return self.net(x)


def train_binary(Xtr, ytr, Xte, yte, seed=42):
    """Train binary MLP for one compartment. Returns (f1, probs)."""
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr).astype(np.float32)
    Xtes = sc.transform(Xte).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    ti, ei = train_test_split(np.arange(len(Xts)), test_size=ES_FRAC, random_state=seed)
    
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
        else: stall += 1
        if stall >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    pr = (tp >= THR).astype(int)
    f1_v = float(f1_score(yte.astype(int), pr, zero_division=0))
    return f1_v, tp


def train_shared(Xtr, Ytr, Xte, Yte):
    """Shared multi-label MLP (baseline comparison)."""
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
    model = SharedMLP(Xts.shape[1], SHARED_HIDDEN, M, SHARED_DROPOUT)
    opt = optim.Adam(model.parameters(), lr=SHARED_LR)
    Xt = torch.from_numpy(Xts); Yt = torch.from_numpy(Ytr.astype(np.float32))
    Xe = torch.from_numpy(Xts[ei]); Ye = Ytr[ei]
    best_f1, best_state, stall = -1.0, None, 0
    for ep in range(1, MAX_EP+1):
        model.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), 256):
            ix = perm[s:s+256]
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


def gen_oof_per_comp(X, Y, prefix=""):
    """Per-compartment 4-fold OOF. Returns (N, M) OOF array."""
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(42); idx = np.arange(n); rng.shuffle(idx)
    fs = n // 4
    for j in range(M):
        yj = Y[:, j]
        oof_j = np.zeros(n, dtype=np.float32)
        print(f"        {prefix}OOF [{COMPARTMENTS[j]:>12s}]...", end="", flush=True)
        t0_j = time.time()
        for f in range(4):
            vs = f*fs; ve = n if f == 3 else (f+1)*fs
            vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
            _, tp = train_binary(X[ti], yj[ti], X[vi], yj[vi], seed=42+f)
            oof_j[vi] = tp
        oof[:, j] = oof_j
        f1_j = float(f1_score(yj.astype(int), (oof_j >= THR).astype(int), zero_division=0))
        dt = time.time() - t0_j
        print(f"  F1={f1_j:.4f}  ({dt:.0f}s)")
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
    print(f"      Features: {X_all.shape}  (4610d)")
    return X_all, Y_all, parts


def run_fold(X_tr, Y_tr, X_te, Y_te, holdout):
    print(f"\n  {'='*50}\n  HOLD-OUT PARTITION {holdout}  ({len(Y_tr)} train, {len(Y_te)} test)\n  {'='*50}")
    
    # ── Shared baseline (no cleanlab) ──
    t0_s = time.time()
    print(f"  Shared baseline...")
    shared_base_f1, shared_base_pc, _ = train_shared(X_tr, Y_tr, X_te, Y_te)
    print(f"      Shared macro F1: {shared_base_f1:.4f}  ({time.time()-t0_s:.0f}s)")
    
    # ── Per-compartment baseline (no cleanlab) ──
    t0_p = time.time()
    print(f"  Per-comp baseline...")
    base_pc_comp = []
    for j in range(M):
        f1_j, _ = train_binary(X_tr, Y_tr[:, j], X_te, Y_te[:, j], seed=42)
        base_pc_comp.append(f1_j)
        print(f"      [{COMPARTMENTS[j]:>12s}] F1={f1_j:.4f}")
    base_comp_f1 = float(np.mean(base_pc_comp))
    print(f"      Per-comp macro F1: {base_comp_f1:.4f}  ({time.time()-t0_p:.0f}s)")
    
    # ── Round 1 OOF + cleanlab ──
    print(f"  Round 1 OOF...")
    oof_r1 = gen_oof_per_comp(X_tr, Y_tr)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    
    # ── Round 2 OOF + cleanlab ──
    print(f"  Round 2 OOF...")
    oof_r2 = gen_oof_per_comp(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    
    # ── Final: per-compartment on cleaned data ──
    t0_f = time.time()
    print(f"  Final per-comp ({len(Y_r2)} train)...")
    final_pc = []
    for j in range(M):
        f1_j, _ = train_binary(X_r2, Y_r2[:, j], X_te, Y_te[:, j], seed=42)
        final_pc.append(f1_j)
        print(f"      [{COMPARTMENTS[j]:>12s}] F1={f1_j:.4f}")
    final_f1 = float(np.mean(final_pc))
    print(f"      Per-comp champion F1: {final_f1:.4f}  ({time.time()-t0_f:.0f}s)")
    
    # ── Also train shared champion on cleanlab survivors for comparison ──
    print(f"  Shared champion on same clean data...")
    shared_champ_f1, shared_champ_pc, _ = train_shared(X_r2, Y_r2, X_te, Y_te)
    print(f"      Shared champion F1: {shared_champ_f1:.4f}")
    
    delta = final_f1 - shared_champ_f1
    print(f"\n  Per-comp vs shared champion: {final_f1:.4f} vs {shared_champ_f1:.4f}  (Δ={delta:+.4f})")
    
    return {
        "holdout": holdout,
        "shared_baseline": shared_base_f1,
        "shared_baseline_per_class": [round(x,4) for x in shared_base_pc],
        "per_comp_baseline": base_comp_f1,
        "per_comp_baseline_per_class": [round(x,4) for x in base_pc_comp],
        "per_comp_champion": round(final_f1, 4),
        "per_comp_champion_per_class": [round(x,4) for x in final_pc],
        "shared_champion": round(shared_champ_f1, 4),
        "shared_champion_per_class": [round(x,4) for x in shared_champ_pc],
        "per_comp_gain_vs_shared": round(delta, 4),
    }


def main():
    t0 = time.time()
    print("=" * 72)
    print("  PER-COMPARTMENT CHAMPION — FULL 4610d")
    print("  7 × BinaryMLP(4610→128→1) vs 1 × SharedMLP(4610→512→7)")
    print("  Cleanlab 2-pass on reconstructed multi-label OOF")
    print("=" * 72)
    
    X_all, Y_all, parts = load_data()
    
    results = []
    for holdout in range(5):
        train_mask = (parts != holdout); test_mask = (parts == holdout)
        r = run_fold(X_all[train_mask], Y_all[train_mask],
                     X_all[test_mask], Y_all[test_mask], holdout)
        results.append(r)
    
    # Summary
    print(f"\n{'='*72}")
    print("  COMPARISON: Per-Comp vs Shared (both after cleanlab 2-pass)")
    print(f"{'='*72}")
    print(f"  {'Holdout':>8s}  {'Shared':>8s}  {'PerComp':>8s}  {'Δ':>8s}  "
          f"{'Shared Base':>11s}  {'PerComp Base':>12s}")
    shared_champs = []; per_comp_champs = []; shared_bases = []; per_comp_bases = []
    for r in results:
        shared_champs.append(r["shared_champion"])
        per_comp_champs.append(r["per_comp_champion"])
        shared_bases.append(r["shared_baseline"])
        per_comp_bases.append(r["per_comp_baseline"])
        d = r["per_comp_gain_vs_shared"]
        print(f"  P{r['holdout']:>7d}  {r['shared_champion']:>8.4f}  {r['per_comp_champion']:>8.4f}  "
              f"{d:>+8.4f}  {r['shared_baseline']:>11.4f}  {r['per_comp_baseline']:>12.4f}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*11}  {'-'*12}")
    sm = np.mean(shared_champs); pm = np.mean(per_comp_champs)
    print(f"  {'Mean':>8s}  {sm:>8.4f}  {pm:>8.4f}  {pm-sm:>+8.4f}  "
          f"{np.mean(shared_bases):>11.4f}  {np.mean(per_comp_bases):>12.4f}")
    print(f"  {'Std':>8s}  {np.std(shared_champs):>8.4f}  {np.std(per_comp_champs):>8.4f}")
    
    print(f"\n  Per-compartment F1 (both after cleanlab):")
    print(f"  {'Compartment':>15s}  {'Shared':>8s}  {'PerComp':>9s}  {'Δ':>8s}")
    for j, c in enumerate(COMPARTMENTS):
        svals = [r["shared_champion_per_class"][j] for r in results]
        pvals = [r["per_comp_champion_per_class"][j] for r in results]
        print(f"  {c:>15s}  {np.mean(svals):>8.4f}  {np.mean(pvals):>9.4f}  "
              f"{np.mean(pvals)-np.mean(svals):>+8.4f}")
    
    print(f"\n  Wall time: {time.time()-t0:.0f}s ({((time.time()-t0)/60):.1f}m)")
    
    report = {
        "architecture": "per_compartment_binary_mlp_full_4610d",
        "per_fold": results,
        "shared_champion_mean": round(float(sm), 4),
        "shared_champion_std": round(float(np.std(shared_champs)), 4),
        "per_comp_mean": round(float(pm), 4),
        "per_comp_std": round(float(np.std(per_comp_champs)), 4),
        "improvement_over_shared_champion": round(float(pm - sm), 4),
        "wall_time_min": round((time.time()-t0)/60, 1),
    }
    
    out_dir = PROJ / "output_cleaning_sweep"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "per_comp_full_result.json").write_text(json.dumps(report, indent=2))
    
    # Save comparison vs the canonical shared champion
    try:
        shared_champ = json.load(open(PROJ / "output_champion_multilayer_attn.json"))
        comparison = {
            "shared_champion_5fold_mean": shared_champ["champion_mean"],
            "shared_champion_5fold_std": shared_champ["champion_std"],
            "per_comp_5fold_mean": pm,
            "per_comp_5fold_std": float(np.std(per_comp_champs)),
            "delta_vs_canonical_shared": round(pm - shared_champ["champion_mean"], 4),
        }
        (out_dir / "per_comp_full_vs_shared.json").write_text(json.dumps(comparison, indent=2))
        print(f"\n  Canonical shared champion: {shared_champ['champion_mean']}")
        print(f"  Per-comp vs canonical: {pm:.4f} vs {shared_champ['champion_mean']:.4f} "
              f"(Δ={pm-shared_champ['champion_mean']:+.4f})")
    except:
        pass
    
    print(f"\n  Saved: output_cleaning_sweep/per_comp_full_result.json")


if __name__ == "__main__":
    main()
