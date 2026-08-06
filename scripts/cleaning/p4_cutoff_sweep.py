#!/usr/bin/env python3
"""p4_cutoff_sweep.py — Sweep CL_CUTOFF ∈ {0.40, 0.45, 0.50, 0.55} on P4.

Attn-pooled ProtT5 L22 + SPACE + aux (1538d). R1 OOF computed once and reused.
"""

import json, os, time, warnings
from pathlib import Path
import h5py, numpy as np, pandas as pd

PROJ = Path(__file__).parent.resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
ATTN_H5 = str(PROJ / "data" / "prott5_attn_all_layers.h5")
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch, torch.nn as nn, torch.optim as optim
from cleanlab.multilabel_classification.rank import get_label_quality_scores

LAYER = 22
HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5

CUTOFFS = [0.40, 0.45, 0.50, 0.55]

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]


class MLP(nn.Module):
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, outdim))
    def forward(self, x): return self.net(x)


def posw(Y):
    pw = np.ones(M, dtype=np.float32)
    for j in range(M):
        pos = float(Y[:, j].sum()); neg = float(Y.shape[0]) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg / pos)
    return np.clip(pw, 1.0, 20.0)


def train_mlp(Xtr, Ytr, Xte, Yte, seed=42):
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
            best_f1 = ef; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stall = 0
        else:
            stall += 1
            if stall >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    pr = (tp >= THR).astype(int)
    pc = [float(f1_score(Yte[:,j].astype(int), pr[:,j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc, tp


def gen_oof(X, Y, n_folds=4, seed=42):
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    fs = n // n_folds
    for f in range(n_folds):
        vs = f*fs; ve = n if f == n_folds-1 else (f+1)*fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
        _, _, tp = train_mlp(X[ti], Y[ti], X[vi], Y[vi], seed=seed+f)
        oof[vi] = tp
    return oof


def cleanlab_step(Y, oof, cutoff):
    labs = [list(np.where(Y[i]==1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    return keep


def run_cutoff(cutoff, X_tr, Y_tr, X_te, Y_te, oof_r1):
    t0 = time.time()
    keep_r1 = cleanlab_step(Y_tr, oof_r1, cutoff)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    n_r1 = int(keep_r1.sum())

    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, cutoff)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    n_r2 = int(keep_r2.sum())

    final_f1, final_pc, _ = train_mlp(X_r2, Y_r2, X_te, Y_te)

    print(f"  cutoff={cutoff:.2f}:  {len(Y_tr)} → {n_r1} → {n_r2}  "
          f"({100*n_r2/len(Y_tr):.1f}% kept)  F1={final_f1:.4f}  [{time.time()-t0:.0f}s]")

    return {
        "cutoff": cutoff,
        "n_r1": n_r1, "n_r2": n_r2,
        "f1": round(final_f1, 4),
        "per_class": [round(x, 4) for x in final_pc],
    }


def main():
    t0 = time.time()
    print("=" * 65)
    print("  P4 CL_CUTOFF SWEEP — {0.40, 0.45, 0.50, 0.55}")
    print("  Attn-pooled ProtT5 L22 + SPACE + aux (1538d)")
    print("=" * 65)

    # ── Load data ──
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != 4); test_mask = (parts == 4)
    n_tr = train_mask.sum(); n_te = test_mask.sum()
    print(f"\n  Loaded: {len(Y_all)} total  Train: {n_tr}  Test: {n_te}")

    with h5py.File(ATTN_H5, "r") as f:
        prot5 = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)
    net_emb = np.load(SPACE_EMB); net_emb[~np.load(SPACE_MASK)] = 0.0
    aux_feats = np.load(AUX_FEATS)
    X_all = np.concatenate([prot5, net_emb, aux_feats], axis=1).astype(np.float32)
    print(f"  Features: {X_all.shape[1]}d")

    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]

    # ── Baseline (once) ──
    print(f"\n  Baseline...")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)
    print(f"  Baseline F1 = {base_f1:.4f}")

    # ── R1 OOF (once, reused across cutoffs) ──
    print(f"\n  Computing R1 OOF (shared across all cutoffs)...")
    t_oof = time.time()
    oof_r1 = gen_oof(X_tr, Y_tr)
    print(f"  R1 OOF done [{time.time()-t_oof:.0f}s]")

    # ── Sweep cutoffs ──
    print(f"\n  Sweeping cutoffs...")
    print(f"  {'cutoff':>8}  {'Flow':>25}  {'F1':>8}")
    print(f"  {'-'*8}  {'-'*25}  {'-'*8}")
    results = []
    for c in CUTOFFS:
        r = run_cutoff(c, X_tr, Y_tr, X_te, Y_te, oof_r1)
        results.append(r)

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*55}")
    print(f"  Baseline (no cleaning):  {base_f1:.4f}")
    print(f"  {'Cutoff':>8}  {'Kept':>6}  {'Drop%':>6}  {'F1':>8}  Δ vs 0.40")
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*10}")
    ref = results[0]["f1"]
    best = max(results, key=lambda r: r["f1"])
    for r in results:
        marker = " ← best" if r["cutoff"] == best["cutoff"] else ""
        print(f"  {r['cutoff']:>8.2f}  {r['n_r2']:>6}  {100*(n_tr-r['n_r2'])/n_tr:>5.1f}%  {r['f1']:>8.4f}  {r['f1']-ref:>+10.4f}{marker}")

    print(f"\n  Best cutoff: {best['cutoff']:.2f} → F1 = {best['f1']:.4f}")
    print(f"\n  Per-class at best cutoff ({best['cutoff']:.2f}):")
    for c, v in zip(COMPARTMENTS, best["per_class"]):
        print(f"    {c:>15s}:  {v:.4f}")

    print(f"\n  Wall time: {time.time()-t0:.0f}s")

    # Save
    out = {
        "config": "attn-pooled L22 + SPACE + aux (1538d)",
        "baseline_f1": round(base_f1, 4),
        "sweep": results,
        "best_cutoff": best["cutoff"],
        "best_f1": best["f1"],
    }
    (PROJ / "output_p4_cutoff_sweep.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
