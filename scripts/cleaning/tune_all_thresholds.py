#!/usr/bin/env python3
"""tune_all_thresholds.py

Matches champion_5fold_cv.py EXACTLY — same train_mlp, MLP, posw, gen_oof,
cleanlab_step, hyperparams, and aux features.

Runs 4 variants on P4 with threshold tuning:
  A: T5 only (1026-d: prot5 L22 + aux)
  B: T5 + cleanlab (2-round)
  C: SPACE only (514-d: SPACE mean-impute + aux)
  D: SPACE + cleanlab (2-round)

Champion (T5 + SPACE + cleanlab) = 0.8011 is already known from champion_5fold_cv.py.

Usage:
  python3 tune_all_thresholds.py
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
PROT5_H5 = str(PROJ / "data" / "prott5_attn_all_layers.h5")
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"

LAYER = 22
HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5; CL_CUTOFF = 0.40

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]
THR_GRID = np.arange(0.02, 0.96, 0.02)


# ─── Exact copies from champion_5fold_cv.py ────────────────────────────────

class MLP(nn.Module):
    """1-layer MLP: indim → hdim → outdim with dropout (v1 champion config)."""
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, outdim))
    def forward(self, x):
        return self.net(x)


def posw(Y):
    pw = np.ones(M, dtype=np.float32)
    for j in range(M):
        pos = float(Y[:, j].sum())
        neg = float(Y.shape[0]) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg / pos)
    return np.clip(pw, 1.0, 20.0)


def train_mlp(Xtr, Ytr, Xte, Yte, seed=42):
    """Matches champion_5fold_cv.py train_mlp EXACTLY."""
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
    """Matches champion_5fold_cv.py gen_oof EXACTLY."""
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    fs = n // n_folds
    for f in range(n_folds):
        vs = f*fs; ve = n if f == n_folds-1 else (f+1)*fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
        _, _, tp = train_mlp(X[ti], Y[ti], X[vi], Y[vi], seed=seed+f)
        oof[vi] = tp
        f1_f = np.mean([f1_score(Y[vi][:,j].astype(int),(tp[:,j]>=THR).astype(int),zero_division=0) for j in range(M)])
        print(f"        [Fold {f+1}] F1={f1_f:.4f}", flush=True)
    return oof


def cleanlab_step(Y, oof, cutoff=CL_CUTOFF):
    """Matches champion_5fold_cv.py cleanlab_step EXACTLY."""
    labs = [list(np.where(Y[i]==1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"        Cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped "
          f"({100*(~keep).sum()/len(Y):.1f}%)")
    return keep


# ─── Threshold tuning ─────────────────────────────────────────────────────

def tune_thresholds(oof, Y):
    """Sweep THR_GRID 0.02–0.94 on OOF, per-class max F1."""
    best = np.full(M, 0.5, dtype=np.float32)
    for j in range(M):
        cands = np.array([f1_score(Y[:,j].astype(int),(oof[:,j]>=t).astype(int),zero_division=0)
                          for t in THR_GRID])
        best[j] = THR_GRID[int(cands.argmax())]
    return best


def eval_at_thresholds(probs, Y, thresholds):
    preds = (probs >= thresholds).astype(int)
    pc = [float(f1_score(Y[:,j].astype(int), preds[:,j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_variant(name, X_tr, Y_tr, X_te, Y_te, do_cleanlab=False):
    """
    Run one variant: baseline + optional 2-round cleanlab, with threshold tuning.
    """
    print(f"\n  {'='*55}")
    print(f"  {name}")
    print(f"  {'='*55}")

    # --- Baseline ---
    print(f"  Baseline ({len(Y_tr)} train, {len(Y_te)} test)...")
    base_f1, base_pc, tp_base = train_mlp(X_tr, Y_tr, X_te, Y_te)
    print(f"  Baseline @ 0.5:  {base_f1:.4f}")

    print(f"  Baseline OOF...")
    oof_base = gen_oof(X_tr, Y_tr)

    thr_base = tune_thresholds(oof_base, Y_tr)
    base_tuned_f1, base_tuned_pc = eval_at_thresholds(tp_base, Y_te, thr_base)
    print(f"  Baseline @ tuned: {base_tuned_f1:.4f}  (Δ {base_tuned_f1-base_f1:+.4f})")

    result = {
        "name": name,
        "n_train": int(len(Y_tr)),
        "n_test": int(len(Y_te)),
        "baseline_f1_05": round(base_f1, 4),
        "baseline_f1_tuned": round(base_tuned_f1, 4),
        "baseline_thresholds": [round(float(t), 3) for t in thr_base],
        "baseline_per_class_05": [round(x, 4) for x in base_pc],
        "baseline_per_class_tuned": [round(x, 4) for x in base_tuned_pc],
    }

    if not do_cleanlab:
        return result

    # --- 2-round Cleanlab ---
    # Round 1
    print(f"  Round 1 OOF + cleanlab...")
    keep_r1 = cleanlab_step(Y_tr, oof_base)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]

    # Round 2: fresh OOF on kept set
    print(f"  Round 2 OOF + cleanlab...")
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    print(f"  Cumulative: {len(Y_tr)} → {len(Y_r1)} → {len(Y_r2)} "
          f"({100*(len(Y_tr)-len(Y_r2))/len(Y_tr):.1f}% total drop)")

    # Final train
    print(f"  Final ({len(Y_r2)} train)...")
    cl_f1, cl_pc, tp_cl = train_mlp(X_r2, Y_r2, X_te, Y_te)
    print(f"  Cleanlab @ 0.5:  {cl_f1:.4f}")

    # CL threshold tuning: use Round 2 OOF
    thr_cl = tune_thresholds(oof_r2, Y_r1)
    cl_tuned_f1, cl_tuned_pc = eval_at_thresholds(tp_cl, Y_te, thr_cl)
    print(f"  Cleanlab @ tuned: {cl_tuned_f1:.4f}  (Δ {cl_tuned_f1-cl_f1:+.4f})")

    result.update({
        "cleanlab_f1_05": round(cl_f1, 4),
        "cleanlab_f1_tuned": round(cl_tuned_f1, 4),
        "cleanlab_thresholds": [round(float(t), 3) for t in thr_cl],
        "cleanlab_per_class_05": [round(x, 4) for x in cl_pc],
        "cleanlab_per_class_tuned": [round(x, 4) for x in cl_tuned_pc],
        "n_after_r1": int(len(Y_r1)),
        "n_after_r2": int(len(Y_r2)),
    })
    return result


def main():
    t0 = time.time()
    print("=" * 65)
    print("  THRESHOLD TUNING — All Variants (P4)")
    print("  Matches champion_5fold_cv.py exactly")
    print("=" * 65)

    # Load data
    print(f"\n  Loading data...")
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != 4); test_mask = (parts == 4)
    Y_tr = Y_all[train_mask]; Y_te = Y_all[test_mask]
    print(f"  Train: {len(Y_tr)}, Test: {len(Y_te)}")

    # ProtT5
    print(f"  Loading ProtT5 L{LAYER}...")
    with h5py.File(PROT5_H5, "r") as f:
        prot5 = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)

    # SPACE
    print(f"  Loading SPACE...")
    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0

    # Mean-impute SPACE (for SPACE-only variants)
    mean_vec = net_emb[net_mask].mean(axis=0)
    net_mean = net_emb.copy()
    net_mean[~net_mask] = mean_vec

    # Aux features
    aux_feats = np.load(AUX_FEATS)
    assert aux_feats.shape[0] == len(src)

    # Build feature matrices
    X_t5     = np.concatenate([prot5, aux_feats], axis=1).astype(np.float32)         # 1026-d
    X_space  = np.concatenate([net_mean, aux_feats], axis=1).astype(np.float32)       # 514-d
    X_t5s    = np.concatenate([prot5, net_filled, aux_feats], axis=1).astype(np.float32)  # 1538-d

    print(f"  T5 only:  {X_t5.shape[1]}-d")
    print(f"  SPACE only: {X_space.shape[1]}-d (mean-imputed)")
    print(f"  T5+SPACE:  {X_t5s.shape[1]}-d")

    # Train/test splits
    X_t5_tr,    X_t5_te    = X_t5[train_mask], X_t5[test_mask]
    X_space_tr, X_space_te = X_space[train_mask], X_space[test_mask]
    X_t5s_tr,   X_t5s_te   = X_t5s[train_mask], X_t5s[test_mask]

    results = []

    # A: T5 only
    results.append(run_variant("T5 only (1026-d)", X_t5_tr, Y_tr, X_t5_te, Y_te, do_cleanlab=False))

    # B: T5 + cleanlab
    results.append(run_variant("T5 + cleanlab", X_t5_tr, Y_tr, X_t5_te, Y_te, do_cleanlab=True))

    # C: SPACE only
    results.append(run_variant("SPACE only (514-d, mean-impute)", X_space_tr, Y_tr, X_space_te, Y_te, do_cleanlab=False))

    # D: SPACE + cleanlab
    results.append(run_variant("SPACE + cleanlab", X_space_tr, Y_tr, X_space_te, Y_te, do_cleanlab=True))

    # E: T5 + SPACE + cleanlab (champion)
    results.append(run_variant("T5 + SPACE + cleanlab 🏆", X_t5s_tr, Y_tr, X_t5s_te, Y_te, do_cleanlab=True))

    # ─── Final table ───────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  FINAL RESULTS — All Variants with Threshold Tuning (P4)")
    print("  (Matches champion_5fold_cv.py protocol: aux feats, 2-round cleanlab)")
    print("=" * 85)

    hdr = f"  {'Variant':<32s}  {'BL@0.5':>8}  {'BL@tuned':>8}  {'Δ tune':>8}  {'CL@0.5':>8}  {'CL@tuned':>8}  {'Δ CL':>8}"
    sep = f"  {'─'*32}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}"
    print(hdr); print(sep)

    for r in results:
        bl_05 = r["baseline_f1_05"]
        bl_t  = r["baseline_f1_tuned"]
        bl_d  = bl_t - bl_05
        cl_05 = r.get("cleanlab_f1_05")
        cl_t  = r.get("cleanlab_f1_tuned")
        def fmt(v): return f"{v:>8.4f}" if v is not None else f"{'—':>8}"
        def fmt_d(v): return f"{v:>+8.4f}" if v is not None else f"{'—':>8}"
        print(f"  {r['name']:<32s}  {bl_05:>8.4f}  {bl_t:>8.4f}  {bl_d:>+8.4f}  "
              f"{fmt(cl_05)}  {fmt(cl_t)}  {fmt_d(cl_t - cl_05 if cl_t is not None and cl_05 is not None else None)}")

    print(sep)

    # Per-class baseline thresholds
    print(f"\n  Per-class tuned thresholds (baseline):")
    print(f"  {'Compartment':<15}", end="")
    for r in results:
        print(f"  {r['name'][:22]:>22s}", end="")
    print()
    for j, c in enumerate(COMPARTMENTS):
        print(f"  {c:<15}", end="")
        for r in results:
            t = r["baseline_thresholds"][j]
            print(f"  {t:>22.2f}", end="")
        print()

    # Per-class cleanlab thresholds
    cl_results = [r for r in results if "cleanlab_thresholds" in r]
    if cl_results:
        print(f"\n  Per-class tuned thresholds (cleanlab):")
        print(f"  {'Compartment':<15}", end="")
        for r in cl_results:
            print(f"  {r['name'][:22]:>22s}", end="")
        print()
        for j, c in enumerate(COMPARTMENTS):
            print(f"  {c:<15}", end="")
            for r in cl_results:
                t = r["cleanlab_thresholds"][j]
                print(f"  {t:>22.2f}", end="")
            print()

    print(f"\n  Wall time: {time.time()-t0:.1f}s")

    report = {"variants": results, "wall_time_sec": round(time.time()-t0, 1)}
    out = PROJ / "output_tuned_thresholds_all.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
