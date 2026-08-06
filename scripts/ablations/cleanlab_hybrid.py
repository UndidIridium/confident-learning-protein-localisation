#!/usr/bin/env python3
"""cleanlab_hybrid.py

HYBRID CL: CleanLearning noise detection + manual 2-round pipeline with tuned thresholds.

Step 1: CleanLearning per-compartment → label_issues_mask (confident joint)
Step 2: Aggregate: per protein, count how many compartments flagged it
Step 3: Drop if flagged by >= min_flags compartments (sweep [1, 2, 3])
Step 4: Manual 2-round cleanlab (self_confidence, cutoff=0.40, posw MLP) on kept data
Step 5: Per-class threshold tuning on OOF

Runs on P4, all 3 CL configs (T5+CL, SPACE+CL, T5+SPACE+CL).

Usage:
  python3 cleanlab_hybrid.py 2>&1 | tee hybrid.log
  # Then in another terminal: tail -f hybrid.log
"""

import json, os, sys, time, warnings
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

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch, torch.nn as nn, torch.optim as optim
from cleanlab.classification import CleanLearning
from cleanlab.multilabel_classification.rank import get_label_quality_scores

M = 7; LAYER = 22
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]
LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular","cell_surface","mitochondrion","endom"]
HIDDEN = 512; DROP = 0.5; LR = 1e-4; MAX_EP = 50; PAT = 5; BS = 256; ES_FRAC = 0.10
CL_CUTOFF = 0.40
THR_GRID = np.arange(0.02, 0.96, 0.02)
MIN_FLAGS_SWEEP = [1, 2, 3]  # drop if >= this many compartments flag a protein


# ═══════════════════════════════════════════════════════════════════
#  Sklearn-compatible MLP wrapper (for CleanLearning, per-compartment binary)
# ═══════════════════════════════════════════════════════════════════

class SklearnMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, indim=1026, hidden=512, dropout=0.5, lr=1e-4, max_ep=50, patience=5, bs=256):
        self.indim = indim; self.hidden = hidden; self.dropout = dropout
        self.lr = lr; self.max_ep = max_ep; self.patience = patience; self.bs = bs

    def _build(self):
        return nn.Sequential(nn.Linear(self.indim, self.hidden), nn.ReLU(True),
                             nn.Dropout(self.dropout), nn.Linear(self.hidden, 1))

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        self.scaler_ = StandardScaler(); Xs = self.scaler_.fit_transform(X)
        pos = float(y.sum()); neg = float(len(y) - pos)
        pw = torch.tensor([min(20.0, neg / max(pos, 1))], dtype=torch.float32)
        self.model_ = self._build(); opt = optim.Adam(self.model_.parameters(), lr=self.lr)
        fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')
        Xt = torch.from_numpy(Xs); Yt = torch.from_numpy(y)
        sw_t = torch.from_numpy(sample_weight.astype(np.float32)) if sample_weight is not None else None
        torch.manual_seed(42); np.random.seed(42)
        best_loss, best_state, stall = float("inf"), None, 0
        for ep in range(1, self.max_ep + 1):
            self.model_.train(); perm = torch.randperm(len(Xt))
            for s in range(0, len(Xt), self.bs):
                ix = perm[s:s + self.bs]
                loss_per = fn(self.model_(Xt[ix]), Yt[ix]).squeeze()
                if sw_t is not None:
                    loss_per = loss_per * sw_t[ix]
                loss = loss_per.mean()
                loss.backward(); opt.step(); opt.zero_grad()
            self.model_.eval()
            with torch.no_grad():
                val_loss = fn(self.model_(Xt), Yt).squeeze()
                if sw_t is not None:
                    val_loss = (val_loss * sw_t).mean()
                else:
                    val_loss = val_loss.mean()
            avg = val_loss.item()
            if avg < best_loss - 1e-6: best_loss = avg; best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}; stall = 0
            else: stall += 1
            if stall >= self.patience: break
        if best_state: self.model_.load_state_dict(best_state)
        self.model_.eval(); self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32); Xs = self.scaler_.transform(X)
        self.model_.eval()
        with torch.no_grad():
            pos = torch.sigmoid(self.model_(torch.from_numpy(Xs))).numpy().flatten()
        return np.column_stack([1 - pos, pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════════
#  Manual pipeline (multi-output MLP) - from champion_5fold_cv.py
# ═══════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Multi-output MLP: indim → hdim → M with dropout."""
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, outdim))
    def forward(self, x):
        return self.net(x)


def posw(Y):
    pw = np.ones(M, dtype=np.float32)
    for j in range(M):
        pos = float(Y[:, j].sum()); neg = float(Y.shape[0]) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg / pos)
    return np.clip(pw, 1.0, 20.0)


def train_mlp_manual(Xtr, Ytr, Xte, Yte, seed=42):
    """Multi-output MLP with posw, early stopping on val F1. Returns (f1, per_class, test_probs)."""
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr).astype(np.float32)
    Xtes = sc.transform(Xte).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    ti, ei = train_test_split(np.arange(len(Xts)), test_size=ES_FRAC, random_state=seed)
    pw = posw(Ytr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw.astype(np.float32)))
    model = MLP(Xts.shape[1], HIDDEN, M, DROP)
    opt = optim.Adam(model.parameters(), lr=LR)
    Xt = torch.from_numpy(Xts); Yt = torch.from_numpy(Ytr.astype(np.float32))
    Xe = torch.from_numpy(Xts[ei]); Ye = Ytr[ei]
    best_f1, best_state, stall = -1.0, None, 0
    for ep in range(1, MAX_EP + 1):
        model.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), BS):
            ix = perm[s:s + BS]
            criterion(model(Xt[ix]), Yt[ix]).backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): ep_ = torch.sigmoid(model(Xe)).numpy()
        ef = float(np.mean([f1_score(Ye[:, j].astype(int), (ep_[:, j] >= 0.5).astype(int), zero_division=0) for j in range(M)]))
        if ef > best_f1 + 1e-6:
            best_f1 = ef
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= PAT: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    pr = (tp >= 0.5).astype(int)
    pc = [float(f1_score(Yte[:, j].astype(int), pr[:, j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc, tp


def gen_oof_manual(X, Y, n_folds=4, seed=42):
    """4-fold OOF using multi-output MLP."""
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    fs = n // n_folds
    for f in range(n_folds):
        vs = f * fs; ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
        _, _, tp = train_mlp_manual(X[ti], Y[ti], X[vi], Y[vi], seed=seed + f)
        oof[vi] = tp
        f1_f = np.mean([f1_score(Y[vi][:, j].astype(int), (tp[:, j] >= 0.5).astype(int), zero_division=0) for j in range(M)])
        print(f"          [Fold {f+1}/{n_folds}] F1={f1_f:.4f}", flush=True)
    return oof


def cleanlab_step_manual(Y, oof, cutoff):
    """Manual cleanlab: self_confidence → keep mask."""
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    pct = 100 * (~keep).sum() / len(Y)
    print(f"          Manual cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({pct:.1f}%)", flush=True)
    return keep


def tune_thresholds(oof, Y):
    """Per-class threshold tuning on OOF probabilities."""
    best = np.full(M, 0.5, dtype=np.float32)
    for j in range(M):
        cands = np.array([f1_score(Y[:, j].astype(int), (oof[:, j] >= t).astype(int), zero_division=0) for t in THR_GRID])
        best[j] = THR_GRID[int(cands.argmax())]
    return best


def eval_at_thresholds(probs, Y, thresholds):
    preds = (probs >= thresholds).astype(int)
    pc = [float(f1_score(Y[:, j].astype(int), preds[:, j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc


# ═══════════════════════════════════════════════════════════════════
#  Hybrid pipeline
# ═══════════════════════════════════════════════════════════════════

def compute_flag_counts(Xtr, Ytr, indim):
    """Run CleanLearning per compartment once, return flag_counts per protein."""
    n = len(Ytr)
    flag_counts = np.zeros(n, dtype=int)
    for j in range(M):
        t_j = time.time()
        clf = SklearnMLP(indim=indim, hidden=HIDDEN, dropout=DROP, lr=LR, max_ep=MAX_EP, patience=PAT, bs=BS)
        cl = CleanLearning(clf=clf, cv_n_folds=4, seed=42, verbose=False)
        cl.fit(Xtr, Ytr[:, j].astype(int))
        issues = cl.label_issues_mask
        flag_counts += issues.astype(int)
        n_issues = int(issues.sum())
        print(f"      {COMPARTMENTS[j]:>12s}: {n_issues:>5d} flagged ({100*n_issues/n:.1f}%)  [{time.time()-t_j:.1f}s]", flush=True)
    return flag_counts


def run_hybrid_from_flags(flag_counts, Xtr, Ytr, Xte, Yte, indim, min_flags, t0):
    """
    Hybrid CL for a single min_flags value (reuses pre-computed flag_counts).

    2. Per protein: count flagged compartments
    3. Drop if flagged >= min_flags
    4. Manual 2-round cleanlab pipeline on kept data
    5. Per-class threshold tuning
    """
    n = len(Ytr)
    print(f"    {'─'*55}", flush=True)
    print(f"    Hybrid CL  (min_flags={min_flags})", flush=True)
    print(f"    {'─'*55}", flush=True)

    # --- Step 2: Aggregate ---
    n_drop = int((flag_counts >= min_flags).sum())
    keep_mask = flag_counts < min_flags
    print(f"    Step 2/4: Flag counts per protein → drop if >= {min_flags} compartments", flush=True)
    print(f"      Flag distribution: {dict(zip(*np.unique(flag_counts, return_counts=True)))}", flush=True)
    print(f"      Dropped {n_drop}/{n} ({100*n_drop/n:.1f}%)  - kept {int(keep_mask.sum())}", flush=True)

    if int(keep_mask.sum()) < 500:
        print(f"      WARNING:  Too few proteins kept - skipping", flush=True)
        return None

    X_kp, Y_kp = Xtr[keep_mask], Ytr[keep_mask]

    # --- Step 3: Manual 2-round cleanlab ---
    print(f"    Step 3/4: Manual 2-round cleanlab pipeline...", flush=True)

    # Round 1 OOF
    print(f"      Round 1 OOF ({len(Y_kp)} proteins)...", flush=True)
    oof_r1 = gen_oof_manual(X_kp, Y_kp)
    keep_r1 = cleanlab_step_manual(Y_kp, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_kp[keep_r1], Y_kp[keep_r1]

    # Round 2 OOF
    print(f"      Round 2 OOF ({len(Y_r1)} proteins)...", flush=True)
    oof_r2 = gen_oof_manual(X_r1, Y_r1)
    keep_r2 = cleanlab_step_manual(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]

    # Final train
    print(f"      Final train ({len(Y_r2)} proteins)...", flush=True)
    final_f1, final_pc, final_tp = train_mlp_manual(X_r2, Y_r2, Xte, Yte)

    # --- Step 4: Per-class threshold tuning ---
    print(f"    Step 4/4: Per-class threshold tuning...", flush=True)
    # Use OOF from round 2 for tuning
    oof_kept = oof_r2[keep_r2]  # OOF on r2-kept proteins
    thr = tune_thresholds(oof_kept, Y_r2)
    tuned_f1, tuned_pc = eval_at_thresholds(final_tp, Yte, thr)

    print(f"    ─────────────────────────────────────────────", flush=True)
    print(f"    min_flags={min_flags}:  n_drop(CL)={n_drop}  kept={len(Y_r2)}  "
          f"F1@0.5={final_f1:.4f}  F1@tuned={tuned_f1:.4f}", flush=True)
    print(f"    Time: {time.time()-t0:.0f}s", flush=True)

    return {
        "min_flags": min_flags,
        "n_drop_cl": n_drop,
        "n_after_cl": int(keep_mask.sum()),
        "n_after_r1": len(Y_r1),
        "n_after_r2": len(Y_r2),
        "f1_05": round(float(final_f1), 4),
        "f1_tuned": round(float(tuned_f1), 4),
        "per_class_05": [round(float(x), 4) for x in final_pc],
        "per_class_tuned": [round(float(x), 4) for x in tuned_pc],
        "thresholds": [round(float(t), 3) for t in thr],
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 75, flush=True)
    print("  HYBRID CLEANLAB", flush=True)
    print("  Strategy: CleanLearning noise detection → manual 2-round pipeline", flush=True)
    print("  Sweeping min_flags ∈ [1, 2, 3] on all 3 CL configs (P4)", flush=True)
    print("=" * 75, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != 4); test_mask = (parts == 4)
    Y_tr, Y_te = Y_all[train_mask], Y_all[test_mask]
    print(f"  Train: {train_mask.sum()}  Test: {test_mask.sum()}", flush=True)

    with h5py.File(ATTN_H5, "r") as f:
        prot5 = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)
    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    aux = np.load(AUX_FEATS)

    mean_vec = net_emb[net_mask].mean(axis=0)
    net_mean = net_emb.copy(); net_mean[~net_mask] = mean_vec
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0

    X_t5    = np.concatenate([prot5, aux], axis=1).astype(np.float32)
    X_space = np.concatenate([net_mean, aux], axis=1).astype(np.float32)
    X_t5s   = np.concatenate([prot5, net_filled, aux], axis=1).astype(np.float32)

    configs = [
        ("T5 + CL",            X_t5[train_mask],    X_t5[test_mask],    X_t5.shape[1]),
        ("SPACE + CL",         X_space[train_mask], X_space[test_mask], X_space.shape[1]),
        ("T5 + SPACE + CL", X_t5s[train_mask],   X_t5s[test_mask],   X_t5s.shape[1]),
    ]

    all_results = []

    for cfg_name, Xtr, Xte, indim in configs:
        print(f"\n{'='*70}", flush=True)
        print(f"  [{cfg_name}]  {indim}-d", flush=True)
        print(f"{'='*70}", flush=True)

        # Step 1: CleanLearning noise detection (once per config)
        print(f"  Step 1/4: CleanLearning noise detection (per compartment)...", flush=True)
        flag_counts = compute_flag_counts(Xtr, Y_tr, indim)

        res = {"name": cfg_name, "indim": indim, "min_flags_sweep": {}}

        for mf in MIN_FLAGS_SWEEP:
            r = run_hybrid_from_flags(flag_counts, Xtr, Y_tr, Xte, Y_te, indim, mf, t0)
            if r is not None:
                res["min_flags_sweep"][str(mf)] = r

        res["best_min_flags"] = max(
            res["min_flags_sweep"].items(),
            key=lambda kv: kv[1]["f1_tuned"]
        )[0] if res["min_flags_sweep"] else None

        all_results.append(res)

    # ═══ Summary ════════════════════════════════════════════════════════
    print(f"\n{'='*90}", flush=True)
    print(f"  HYBRID CL RESULTS - P4", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"  {'Config':<25s}  {'min_flags':>10}  {'CL Drop':>8}  {'Final N':>8}  {'F1@0.5':>8}  {'F1@tuned':>8}  {'Manual':>8}", flush=True)
    print(f"  {'─'*25}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}", flush=True)

    manual = {"T5 + CL": 0.7809, "SPACE + CL": 0.7054, "T5 + SPACE + CL": 0.7994}

    for r in all_results:
        n = r["name"]
        for mf_str, rd in r["min_flags_sweep"].items():
            print(f"  {n:<25s}  {mf_str:>10}  {rd['n_drop_cl']:>8}  {rd['n_after_r2']:>8}  "
                  f"{rd['f1_05']:>8.4f}  {rd['f1_tuned']:>8.4f}  {manual.get(n,0):>8.4f}", flush=True)
        if r["best_min_flags"]:
            br = r["min_flags_sweep"][r["best_min_flags"]]
            print(f"  {'  * best':<25s}  {r['best_min_flags']:>10}  {br['n_drop_cl']:>8}  {br['n_after_r2']:>8}  "
                  f"{br['f1_05']:>8.4f}  {br['f1_tuned']:>8.4f}", flush=True)

    print(f"\n  Wall time: {time.time()-t0:.1f}s", flush=True)

    out = PROJ / "output_cleanlab_hybrid.json"
    out.write_text(json.dumps({
        "results": all_results,
        "manual": manual,
        "wall_s": round(time.time() - t0, 1),
    }, indent=2))
    print(f"  Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
