#!/usr/bin/env python3
"""champion_5fold_cv_attn.py

5-fold cross-validation of the manual champion pipeline with ATTENTION-POOLED ProtT5.
Same as champion_5fold_cv.py but uses prott5_attn_all_layers.h5 (attn_layer_22).

Pipeline: 2-round MLP self-confidence cleanlab → final MLP
Compares apples-to-apples with hybrid_5fold_cv.py and alternating_5fold_cv_attn.py.

Usage:
  python3 champion_5fold_cv_attn.py 2>&1 | tee champion_5fold_attn.log
  tail -f champion_5fold_attn.log
"""

import json, os, time, warnings
from pathlib import Path
import h5py, numpy as np, pandas as pd

PROJ = Path(__file__).parent.resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
ATTN_H5 = str(PROJ / "data" / "prott5_attn_all_layers.h5")
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"

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
THR = 0.5; CL_CUTOFF = 0.40

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


def run_fold(holdout):
    print(f"\n  {'='*50}")
    print(f"  HOLD-OUT PARTITION {holdout}")
    print(f"  {'='*50}")

    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != holdout)
    test_mask = (parts == holdout)
    n_tr = train_mask.sum(); n_te = test_mask.sum()

    with h5py.File(ATTN_H5, "r") as f:
        prot5 = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)

    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    aux_feats = np.load(AUX_FEATS)
    X_all = np.concatenate([prot5, net_filled, aux_feats], axis=1).astype(np.float32)

    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]

    # Baseline
    print(f"  Baseline ({n_tr} train, {n_te} test)...")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)

    # Round 1 OOF + cleanlab
    print(f"  Round 1 OOF...")
    oof_r1 = gen_oof(X_tr, Y_tr)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]

    # Round 2 OOF + cleanlab
    print(f"  Round 2 OOF...")
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]

    # Final train
    print(f"  Final ({len(Y_r2)} train)...")
    final_f1, final_pc, _ = train_mlp(X_r2, Y_r2, X_te, Y_te)

    print(f"  Result: Baseline={base_f1:.4f}  Champion={final_f1:.4f}  Gain={final_f1-base_f1:+.4f}")

    return {
        "holdout": holdout,
        "n_train": int(n_tr), "n_test": int(n_te),
        "n_after_r1": int(len(Y_r1)), "n_after_r2": int(len(Y_r2)),
        "baseline_f1": round(base_f1, 4),
        "champion_f1": round(final_f1, 4),
        "gain": round(final_f1 - base_f1, 4),
        "baseline_per_class": [round(x, 4) for x in base_pc],
        "champion_per_class": [round(x, 4) for x in final_pc],
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print(f"  MANUAL CHAMPION — 5-FOLD CV — ATTENTION-POOLED")
    print(f"  Pipeline: 2-round MLP self-confidence → final MLP")
    print(f"  Embeddings: attn-pooled ProtT5 L22 + SPACE + aux (1538-d)")
    print("=" * 70)

    results = []
    for holdout in range(5):
        r = run_fold(holdout)
        results.append(r)

    # Summary
    print("\n" + "=" * 65)
    print("  5-FOLD CV SUMMARY — MANUAL (ATTN-POOLED)")
    print("=" * 65)
    print(f"  {'Holdout':>8}  {'Baseline':>9}  {'Champion':>9}  {'Gain':>8}  n_train → n_R2")
    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*18}")

    baselines = []; champions = []
    for r in results:
        baselines.append(r["baseline_f1"]); champions.append(r["champion_f1"])
        print(f"  P{r['holdout']:>7}  {r['baseline_f1']:>9.4f}  {r['champion_f1']:>9.4f}  "
              f"{r['gain']:>+8.4f}  {r['n_train']} → {r['n_after_r2']}")

    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*18}")
    print(f"  {'Mean':>8}  {np.mean(baselines):>9.4f}  {np.mean(champions):>9.4f}  "
          f"{np.mean(champions)-np.mean(baselines):>+8.4f}")
    print(f"  {'Std':>8}  {np.std(baselines):>9.4f}  {np.std(champions):>9.4f}")

    # Per-class
    print(f"\n  Per-class champion F1 (mean ± std across 5 folds):")
    for j, c in enumerate(COMPARTMENTS):
        vals = [r["champion_per_class"][j] for r in results]
        print(f"    {c:>15s}:  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    print(f"\n  Wall time: {time.time()-t0:.1f}s")

    # Save
    report = {
        "embedding": "attention-pooled ProtT5 L22",
        "per_fold": results,
        "baseline_mean": round(float(np.mean(baselines)), 4),
        "baseline_std": round(float(np.std(baselines)), 4),
        "champion_mean": round(float(np.mean(champions)), 4),
        "champion_std": round(float(np.std(champions)), 4),
        "overall_gain": round(float(np.mean(champions) - np.mean(baselines)), 4),
    }
    report_path = PROJ / "output_champion_5fold_cv_attn.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved: {report_path}")


if __name__ == "__main__":
    main()
