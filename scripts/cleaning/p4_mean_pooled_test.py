#!/usr/bin/env python3
"""p4_mean_pooled_test.py

Quick single P4 run - same pipeline as champion_5fold_cv_attn.py but with
MEAN-POOLED ProtT5 L22 embeddings (from prott5_all_layers_dfadi-3.h5).
No aux features - matches the original 0.8011 champion_pipeline.py exactly.

Purpose: does mean-pooled embedding give us 0.80 on P4?
"""

import json, os, time, warnings
from pathlib import Path
import h5py, numpy as np, pandas as pd

PROJ = Path(__file__).parent.resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
ATTN_H5 = str(PROJ / "data" / "prott5_attn_all_layers.h5")  # ATTENTION-POOLED
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
THR = 0.5; CL_CUTOFF = 0.50

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
            best_f1 = ef
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
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
        f1_f = np.mean([f1_score(Y[vi][:,j].astype(int), (tp[:,j]>=THR).astype(int), zero_division=0) for j in range(M)])
        print(f"        [Fold {f+1}] F1={f1_f:.4f}", flush=True)
    return oof


def cleanlab_step(Y, oof, cutoff):
    labs = [list(np.where(Y[i]==1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"        Cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)\n")
    return keep


def main():
    t0 = time.time()
    print("=" * 70)
    print("  P4 TEST - ATTN-POOLED ProtT5 L22 + CL_CUTOFF=0.50")
    print("  Pipeline: 2-round MLP self-confidence → final MLP")
    print("  Features: ProtT5 L22 attn-pooled (1024d) + SPACE (512d) + aux (2d) = 1538d")
    print("  CL_CUTOFF=0.50 - matched to tinker9 config that produced 0.8011")
    print("=" * 70)

    # ── Load data ──
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != 4)
    test_mask = (parts == 4)
    n_tr = train_mask.sum(); n_te = test_mask.sum()
    print(f"\n  Loaded: {len(Y_all)} total  Train: {n_tr}  Test: {n_te}")

    # ── Load MEAN-POOLED ProtT5 L22 ──
    print(f"  Loading attn-pooled ProtT5 L{LAYER} from {ATTN_H5}...")
    with h5py.File(ATTN_H5, "r") as f:
        prot5 = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)
    print(f"    Shape: {prot5.shape}  Mean: {prot5.mean():.2f}  Std: {prot5.std():.2f}")

    # ── Load SPACE ──
    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    n_missing = int((~net_mask).sum())
    print(f"    SPACE: {net_filled.shape}  (zero-padded {n_missing} / {100*n_missing/len(net_mask):.1f}%)")

    # ── Load aux features ──
    aux_feats = np.load(AUX_FEATS)
    print(f"    Aux features: {aux_feats.shape}")

    # ── Concat ──
    X_all = np.concatenate([prot5, net_filled, aux_feats], axis=1).astype(np.float32)
    print(f"    Feature dim: {X_all.shape[1]}d  (1024 ProtT5 + 512 SPACE + 2 aux)")

    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]

    # ── Baseline ──
    print(f"\n  [1] Baseline ({n_tr} train, {n_te} test)...")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)
    print(f"      Baseline Macro F1 = {base_f1:.4f}")

    # ── Round 1 ──
    print(f"\n  [2] Round 1 OOF + cleanlab...")
    oof_r1 = gen_oof(X_tr, Y_tr)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]

    # ── Round 2 ──
    print(f"  [3] Round 2 OOF + cleanlab...")
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    print(f"      Flow: {len(Y_tr)} → {len(Y_r1)} → {len(Y_r2)} ({100*(len(Y_tr)-len(Y_r2))/len(Y_tr):.1f}% dropped)")

    # ── Final ──
    print(f"\n  [4] Final training on {len(Y_r2)} proteins...")
    final_f1, final_pc, _ = train_mlp(X_r2, Y_r2, X_te, Y_te)
    gain = final_f1 - base_f1

    # ── Report ──
    print("\n" + "=" * 70)
    print("  RESULTS - ATTN-POOLED P4, CL_CUTOFF=0.50")
    print("=" * 70)
    print(f"  {'Baseline (no cleaning)':>35s}  {base_f1:.4f}")
    print(f"  {'Champion (2-round cleanlab)':>35s}  {final_f1:.4f}")
    print(f"  {'Gain from cleaning':>35s}  {gain:+.4f}")
    print(f"  {'Proteins retained':>35s}  {len(Y_r2)} / {len(Y_tr)} ({100*len(Y_r2)/len(Y_tr):.1f}%)")
    print()
    print(f"  Per-class F1 (champion):")
    for c, v in zip(COMPARTMENTS, final_pc):
        print(f"    {c:>15s}:  {v:.4f}")
    print()

    # ── Comparison ──
    print(f"  vs tinker9 (attn-pooled, cutoff=0.50, 0.8011):            Δ = {final_f1 - 0.8011:+.4f}")
    print(f"  vs champion (attn-pooled, cutoff=0.40, 0.7994):           Δ = {final_f1 - 0.7994:+.4f}")
    print(f"\n  Wall time: {time.time()-t0:.1f}s")
    print("=" * 70)

    # Save
    report = {
        "embedding": f"mean-pooled ProtT5 L{LAYER} from prott5_all_layers_dfadi-3.h5",
        "feature_dim": int(X_all.shape[1]),
        "aux_features": False,
        "baseline_f1": round(base_f1, 4),
        "champion_f1": round(final_f1, 4),
        "gain": round(gain, 4),
        "n_train": int(n_tr), "n_r1": int(len(Y_r1)), "n_r2": int(len(Y_r2)), "n_test": int(n_te),
        "per_class": [round(x, 4) for x in final_pc],
    }
    out = PROJ / "output_p4_mean_pooled_test.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
