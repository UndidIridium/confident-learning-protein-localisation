#!/usr/bin/env python3
"""hybrid_5fold_cv.py

HYBRID CL - 5-FOLD CROSS-VALIDATION on champion config (T5+SPACE+CL, min_flags=3).

Per fold:
  1. CleanLearning per-compartment → flag_counts
  2. Drop proteins flagged by >= 3 compartments
  3. Manual 2-round cleanlab + per-class threshold tuning
  4. Report fold F1

Compare against manual 5-fold CV (mean 0.7696).

Usage:
  python3 hybrid_5fold_cv.py 2>&1 | tee hybrid_5fold.log
  tail -f hybrid_5fold.log
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
CL_CUTOFF = 0.40; MIN_FLAGS = 3
THR_GRID = np.arange(0.02, 0.96, 0.02)


# ═══════════════════════════════════════════════════════════════════
#  Sklearn MLP (per-compartment binary - for CleanLearning)
# ═══════════════════════════════════════════════════════════════════

class SklearnMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, indim=1538, hidden=512, dropout=0.5, lr=1e-4, max_ep=50, patience=5, bs=256):
        self.indim = indim; self.hidden = hidden; self.dropout = dropout
        self.lr = lr; self.max_ep = max_ep; self.patience = patience; self.bs = bs

    def _build(self):
        return nn.Sequential(nn.Linear(self.indim, self.hidden), nn.ReLU(True),
                             nn.Dropout(self.dropout), nn.Linear(self.hidden, 1))

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
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
                if sw_t is not None: loss_per = loss_per * sw_t[ix]
                loss_per.mean().backward(); opt.step(); opt.zero_grad()
            self.model_.eval()
            with torch.no_grad():
                val_loss = fn(self.model_(Xt), Yt).squeeze()
                if sw_t is not None: val_loss = (val_loss * sw_t).mean()
                else: val_loss = val_loss.mean()
            if val_loss.item() < best_loss - 1e-6: best_loss = val_loss.item(); best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}; stall = 0
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
#  Manual pipeline (multi-output MLP)
# ═══════════════════════════════════════════════════════════════════

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


def train_mlp_manual(Xtr, Ytr, Xte, Yte, seed=42):
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr).astype(np.float32); Xtes = sc.transform(Xte).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    ti, ei = train_test_split(np.arange(len(Xts)), test_size=ES_FRAC, random_state=seed)
    pw = posw(Ytr)
    model = MLP(Xts.shape[1], HIDDEN, M, DROP)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw.astype(np.float32)))
    Xt = torch.from_numpy(Xts); Yt = torch.from_numpy(Ytr.astype(np.float32))
    Xe = torch.from_numpy(Xts[ei]); Ye = Ytr[ei]
    best_f1, best_state, stall = -1.0, None, 0
    for ep in range(1, MAX_EP + 1):
        model.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), BS):
            ix = perm[s:s + BS]; criterion(model(Xt[ix]), Yt[ix]).backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): ep_ = torch.sigmoid(model(Xe)).numpy()
        ef = float(np.mean([f1_score(Ye[:, j].astype(int), (ep_[:, j] >= 0.5).astype(int), zero_division=0) for j in range(M)]))
        if ef > best_f1 + 1e-6: best_f1 = ef; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stall = 0
        else: stall += 1
        if stall >= PAT: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    pr = (tp >= 0.5).astype(int)
    pc = [float(f1_score(Yte[:, j].astype(int), pr[:, j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc, tp


def gen_oof_manual(X, Y, n_folds=4, seed=42):
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
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"          Manual cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)", flush=True)
    return keep


def tune_thresholds(oof, Y):
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
#  Per-fold hybrid pipeline
# ═══════════════════════════════════════════════════════════════════

def run_fold(holdout, X_all, Y_all, parts, fold_t0):
    print(f"\n{'='*60}", flush=True)
    print(f"  HOLD-OUT PARTITION {holdout}", flush=True)
    print(f"{'='*60}", flush=True)

    train_mask = (parts != holdout); test_mask = (parts == holdout)
    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]
    n_tr, n_te = len(Y_tr), len(Y_te)
    indim = X_tr.shape[1]
    print(f"  Train: {n_tr}  Test: {n_te}  dim={indim}", flush=True)

    # --- Baseline (multi-output, no cleaning) ---
    print(f"  Baseline...", flush=True)
    t_bl = time.time()
    base_f1, base_pc, _ = train_mlp_manual(X_tr, Y_tr, X_te, Y_te)
    print(f"    Baseline F1@0.5 = {base_f1:.4f}  [{time.time()-t_bl:.1f}s]", flush=True)

    # --- Step 1: CleanLearning noise detection ---
    print(f"  Step 1: CleanLearning per compartment...", flush=True)
    t_cl = time.time()
    flag_counts = np.zeros(n_tr, dtype=int)
    for j in range(M):
        t_j = time.time()
        clf = SklearnMLP(indim=indim, hidden=HIDDEN, dropout=DROP, lr=LR, max_ep=MAX_EP, patience=PAT, bs=BS)
        cl = CleanLearning(clf=clf, cv_n_folds=4, seed=42, verbose=False)
        cl.fit(X_tr, Y_tr[:, j].astype(int))
        flag_counts += cl.label_issues_mask.astype(int)
        n_issues = int(cl.label_issues_mask.sum())
        print(f"    {COMPARTMENTS[j]:>12s}: {n_issues:>5d} flagged ({100*n_issues/n_tr:.1f}%)  [{time.time()-t_j:.1f}s]", flush=True)
    print(f"    Flag distribution: {dict(sorted(zip(*np.unique(flag_counts, return_counts=True))))}", flush=True)

    # --- Step 2: Drop noisy ---
    keep_cl = flag_counts < MIN_FLAGS
    n_drop = int((~keep_cl).sum())
    print(f"  Step 2: Drop if >= {MIN_FLAGS} flags → {n_drop} dropped, {int(keep_cl.sum())} kept", flush=True)
    X_kp, Y_kp = X_tr[keep_cl], Y_tr[keep_cl]

    # --- Step 3: Manual 2-round cleanlab ---
    print(f"  Step 3: Manual 2-round cleanlab ({len(Y_kp)} proteins)...", flush=True)

    print(f"    Round 1 OOF...", flush=True)
    oof_r1 = gen_oof_manual(X_kp, Y_kp)
    keep_r1 = cleanlab_step_manual(Y_kp, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_kp[keep_r1], Y_kp[keep_r1]

    print(f"    Round 2 OOF ({len(Y_r1)} proteins)...", flush=True)
    oof_r2 = gen_oof_manual(X_r1, Y_r1)
    keep_r2 = cleanlab_step_manual(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]

    # --- Step 4: Final train + threshold tuning ---
    print(f"    Final train ({len(Y_r2)} proteins)...", flush=True)
    final_f1, final_pc, final_tp = train_mlp_manual(X_r2, Y_r2, X_te, Y_te)

    # Per-class threshold tuning from round 2 OOF
    oof_kept = oof_r2[keep_r2]
    thr = tune_thresholds(oof_kept, Y_r2)
    tuned_f1, tuned_pc = eval_at_thresholds(final_tp, Y_te, thr)

    print(f"  ───────────────────────────────────────────", flush=True)
    print(f"  P{holdout}:  Baseline={base_f1:.4f}  F1@0.5={final_f1:.4f}  F1@tuned={tuned_f1:.4f}", flush=True)
    print(f"  Δ vs baseline:  +{final_f1-base_f1:+.4f} (0.5)  +{tuned_f1-base_f1:+.4f} (tuned)", flush=True)
    print(f"  Time: {time.time()-fold_t0:.0f}s", flush=True)

    return {
        "holdout": int(holdout),
        "n_train": n_tr, "n_test": n_te,
        "n_drop_cl": n_drop, "n_after_r1": len(Y_r1), "n_after_r2": len(Y_r2),
        "baseline_f1": round(base_f1, 4),
        "hybrid_f1_05": round(float(final_f1), 4),
        "hybrid_f1_tuned": round(float(tuned_f1), 4),
        "baseline_per_class": [round(float(x), 4) for x in base_pc],
        "hybrid_per_class_05": [round(float(x), 4) for x in final_pc],
        "hybrid_per_class_tuned": [round(float(x), 4) for x in tuned_pc],
        "thresholds": [round(float(t), 3) for t in thr],
        "flag_distribution": {str(k): int(v) for k, v in zip(*np.unique(flag_counts, return_counts=True))},
    }


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 75, flush=True)
    print("  HYBRID CL - 5-FOLD CROSS-VALIDATION", flush=True)
    print(f"  Config: T5+SPACE+CL  |  min_flags={MIN_FLAGS}", flush=True)
    print("  Strategy: CL noise detection → manual 2-round + tuned thresholds", flush=True)
    print("=" * 75, flush=True)

    # Load full dataset
    print("\nLoading data...", flush=True)
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)

    with h5py.File(ATTN_H5, "r") as f:
        prot5 = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)
    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    aux = np.load(AUX_FEATS)

    X_all = np.concatenate([prot5, net_filled, aux], axis=1).astype(np.float32)
    parts = src["partition"].to_numpy()
    print(f"  Full dataset: {len(X_all)} proteins × {X_all.shape[1]} features", flush=True)

    results = []
    for holdout in range(5):
        r = run_fold(holdout, X_all, Y_all, parts, t0)
        results.append(r)

    # ═══ Summary ════════════════════════════════════════════════════════
    print(f"\n{'='*75}", flush=True)
    print(f"  5-FOLD CV SUMMARY - HYBRID CL", flush=True)
    print(f"{'='*75}", flush=True)
    print(f"  {'Holdout':>8}  {'Baseline':>9}  {'Hyb@0.5':>9}  {'Hyb@tuned':>9}  {'n_train → n_R2':>16}", flush=True)
    print(f"  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*16}", flush=True)

    baselines, f1_05s, f1_tuneds = [], [], []
    for r in results:
        baselines.append(r["baseline_f1"]); f1_05s.append(r["hybrid_f1_05"]); f1_tuneds.append(r["hybrid_f1_tuned"])
        print(f"  P{r['holdout']:>7}  {r['baseline_f1']:>9.4f}  {r['hybrid_f1_05']:>9.4f}  "
              f"{r['hybrid_f1_tuned']:>9.4f}  {r['n_train']} → {r['n_after_r2']}", flush=True)

    print(f"  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*16}", flush=True)
    print(f"  {'Mean':>8}  {np.mean(baselines):>9.4f}  {np.mean(f1_05s):>9.4f}  "
          f"{np.mean(f1_tuneds):>9.4f}", flush=True)
    print(f"  {'Std':>8}  {np.std(baselines):>9.4f}  {np.std(f1_05s):>9.4f}  "
          f"{np.std(f1_tuneds):>9.4f}", flush=True)

    # Per-class
    print(f"\n  Per-class hybrid F1@tuned (mean ± std across 5 folds):", flush=True)
    for j, c in enumerate(COMPARTMENTS):
        vals = [r["hybrid_per_class_tuned"][j] for r in results]
        print(f"    {c:>15s}:  {np.mean(vals):.4f} ± {np.std(vals):.4f}", flush=True)

    # Compare vs manual 5-fold
    manual_json = PROJ / "output_champion_5fold_cv.json"
    if manual_json.exists():
        manual = json.loads(manual_json.read_text())
        manual_mean = manual.get("champion_mean", 0.7696)
        manual_folds = {r["holdout"]: r["champion_f1"] for r in manual.get("per_fold", [])}
    else:
        manual_mean = 0.7696
        manual_folds = {}

    print(f"\n  ───────────────────────────────────────────────────────", flush=True)
    print(f"  VS MANUAL 5-FOLD CV", flush=True)
    print(f"    {'Fold':>6}  {'Manual':>9}  {'Hybrid':>9}  {'Δ':>8}", flush=True)
    print(f"    {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}", flush=True)
    for r in results:
        h = r["holdout"]
        mf = manual_folds.get(h, None)
        if mf is not None:
            d = r["hybrid_f1_tuned"] - mf
            print(f"    P{h:>5}  {mf:>9.4f}  {r['hybrid_f1_tuned']:>9.4f}  {d:>+8.4f}", flush=True)
    print(f"    {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}", flush=True)
    print(f"    {'Mean':>6}  {manual_mean:>9.4f}  {np.mean(f1_tuneds):>9.4f}  {np.mean(f1_tuneds)-manual_mean:>+8.4f}", flush=True)

    print(f"\n  Wall time: {time.time()-t0:.1f}s ({((time.time()-t0)/3600):.1f}h)", flush=True)

    # Save
    report = {
        "config": "T5+SPACE+CL", "min_flags": MIN_FLAGS,
        "per_fold": results,
        "baseline_mean": round(float(np.mean(baselines)), 4),
        "baseline_std": round(float(np.std(baselines)), 4),
        "hybrid_mean_05": round(float(np.mean(f1_05s)), 4),
        "hybrid_std_05": round(float(np.std(f1_05s)), 4),
        "hybrid_mean_tuned": round(float(np.mean(f1_tuneds)), 4),
        "hybrid_std_tuned": round(float(np.std(f1_tuneds)), 4),
        "manual_5fold_mean": manual_5fold_mean,
        "delta_vs_manual": round(float(np.mean(f1_tuneds) - manual_5fold_mean), 4),
        "wall_s": round(time.time() - t0, 1),
    }
    out = PROJ / "output_hybrid_5fold_cv.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  Report saved: {out}", flush=True)


if __name__ == "__main__":
    main()
