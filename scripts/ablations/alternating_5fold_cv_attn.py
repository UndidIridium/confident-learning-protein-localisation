#!/usr/bin/env python3
"""alternating_5fold_cv.py

5-FOLD CV of the alternating pipeline:
  CL consensus → MLP self-confidence cleanlab → CL consensus → final MLP + tuned thresholds

Compares with:
  Manual: 2-round MLP self-confidence (5-fold mean: 0.7696)
  Hybrid: CL → manual R1 → manual R2 (5-fold mean: 0.7838)

Usage:
  python3 alternating_5fold_cv.py 2>&1 | tee alternating_5fold.log
  tail -f alternating_5fold.log
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


# ═══════════════ Sklearn MLP (per-compartment binary — for CleanLearning) ═══════════════

class SklearnMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, indim=1026, hidden=512, dropout=0.5, lr=1e-4, max_ep=50, patience=5, bs=256):
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


# ═══════════════ Multi-output MLP ═══════════════

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


def gen_oof(X, Y, n_folds=4, seed=42):
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    fs = n // n_folds
    for f in range(n_folds):
        vs = f * fs; ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
        _, _, tp = train_mlp(X[ti], Y[ti], X[vi], Y[vi], seed=seed + f)
        oof[vi] = tp
        f1_f = np.mean([f1_score(Y[vi][:, j].astype(int), (tp[:, j] >= 0.5).astype(int), zero_division=0) for j in range(M)])
        print(f"          [Fold {f+1}/{n_folds}] F1={f1_f:.4f}", flush=True)
    return oof


def cleanlab_self_confidence(Y, oof, cutoff):
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"          Self-confidence: {int(keep.sum())} kept, {int((~keep).sum())} dropped "
          f"({100*(~keep).sum()/len(Y):.1f}%)", flush=True)
    return keep


def cleanlearning_consensus(X, Y, indim):
    """Run CleanLearning per compartment, return consensus keep mask (min_flags)."""
    n = len(Y)
    flag_counts = np.zeros(n, dtype=int)
    for j in range(M):
        t_j = time.time()
        clf = SklearnMLP(indim=indim, hidden=HIDDEN, dropout=DROP, lr=LR, max_ep=MAX_EP, patience=PAT, bs=BS)
        cl = CleanLearning(clf=clf, cv_n_folds=4, seed=42, verbose=False)
        cl.fit(X, Y[:, j].astype(int))
        flag_counts += cl.label_issues_mask.astype(int)
        n_issues = int(cl.label_issues_mask.sum())
        print(f"        {COMPARTMENTS[j]:>12s}: {n_issues:>5d} flagged "
              f"({100*n_issues/n:.1f}%)  [{time.time()-t_j:.1f}s]", flush=True)

    keep = flag_counts < MIN_FLAGS
    n_drop = int((~keep).sum())
    dist = dict(sorted(zip(*np.unique(flag_counts, return_counts=True))))
    print(f"        Flag distribution: {dist}", flush=True)
    print(f"        Consensus (min_flags={MIN_FLAGS}): {n_drop} dropped, "
          f"{int(keep.sum())} kept ({100*int(keep.sum())/n:.1f}%)", flush=True)
    return keep


def tune_thresholds(oof, Y):
    best = np.full(M, 0.5, dtype=np.float32)
    for j in range(M):
        cands = np.array([f1_score(Y[:, j].astype(int), (oof[:, j] >= t).astype(int),
                                   zero_division=0) for t in THR_GRID])
        best[j] = THR_GRID[int(cands.argmax())]
    return best


def eval_at_thresholds(probs, Y, thresholds):
    preds = (probs >= thresholds).astype(int)
    pc = [float(f1_score(Y[:, j].astype(int), preds[:, j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc


# ═══════════════ Run one fold ═══════════════

def run_fold(holdout):
    t_fold = time.time()
    print(f"\n{'='*70}", flush=True)
    print(f"  HOLD-OUT PARTITION {holdout}", flush=True)
    print(f"{'='*70}", flush=True)

    # Load data for this fold
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != holdout)
    test_mask = (parts == holdout)
    n_tr, n_te = train_mask.sum(), test_mask.sum()

    with h5py.File(ATTN_H5, "r") as f:
        prot5 = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)
    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    aux = np.load(AUX_FEATS)

    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    X_all = np.concatenate([prot5, net_filled, aux], axis=1).astype(np.float32)
    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]
    indim = X_tr.shape[1]

    # Baseline (no cleaning)
    print(f"  Baseline ({n_tr} train, {n_te} test)...", flush=True)
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)

    # ═══ Step 1: CL consensus on raw data ═══
    print(f"  Step 1: CleanLearning consensus ({n_tr} proteins)...", flush=True)
    keep_cl1 = cleanlearning_consensus(X_tr, Y_tr, indim)
    X_cl1, Y_cl1 = X_tr[keep_cl1], Y_tr[keep_cl1]
    n_cl1 = len(Y_cl1)

    # ═══ Step 2: MLP OOF + self-confidence ═══
    print(f"  Step 2: MLP self-confidence ({n_cl1} proteins)...", flush=True)
    oof_mlp = gen_oof(X_cl1, Y_cl1)
    keep_mlp = cleanlab_self_confidence(Y_cl1, oof_mlp, CL_CUTOFF)
    X_mlp, Y_mlp = X_cl1[keep_mlp], Y_cl1[keep_mlp]
    n_mlp = len(Y_mlp)

    # ═══ Step 3: CL consensus on MLP-cleaned data ═══
    print(f"  Step 3: CleanLearning consensus ({n_mlp} proteins)...", flush=True)
    keep_cl2 = cleanlearning_consensus(X_mlp, Y_mlp, indim)
    X_cl2, Y_cl2 = X_mlp[keep_cl2], Y_mlp[keep_cl2]
    n_cl2 = len(Y_cl2)

    # ═══ Step 4: Final MLP + threshold tuning ═══
    print(f"  Step 4: Final train ({n_cl2} proteins)...", flush=True)
    final_f1_05, final_pc_05, final_tp = train_mlp(X_cl2, Y_cl2, X_te, Y_te)

    # Threshold tuning from step 2 OOF (subset to final kept proteins)
    oof_tune = oof_mlp[keep_mlp][keep_cl2]
    thr = tune_thresholds(oof_tune, Y_cl2)
    tuned_f1, tuned_pc = eval_at_thresholds(final_tp, Y_te, thr)

    elapsed = time.time() - t_fold
    print(f"  {'─'*55}", flush=True)
    print(f"  P{holdout}:  Baseline={base_f1:.4f}  F1@0.5={final_f1_05:.4f}  "
          f"F1@tuned={tuned_f1:.4f}", flush=True)
    print(f"  Δ vs baseline:  {'+'+str(round(final_f1_05-base_f1,4)) if final_f1_05>=base_f1 else str(round(final_f1_05-base_f1,4))} (0.5)  "
          f"{'+'+str(round(tuned_f1-base_f1,4)) if tuned_f1>=base_f1 else str(round(tuned_f1-base_f1,4))} (tuned)", flush=True)
    print(f"  Protein flow: {n_tr} → {n_cl1} → {n_mlp} → {n_cl2}", flush=True)
    print(f"  Time: {elapsed:.0f}s", flush=True)

    return {
        "holdout": holdout,
        "n_train": int(n_tr), "n_test": int(n_te),
        "n_cl1": int(n_cl1), "n_mlp": int(n_mlp), "n_cl2": int(n_cl2),
        "baseline_f1": round(float(base_f1), 4),
        "baseline_per_class": [round(float(x), 4) for x in base_pc],
        "f1_05": round(float(final_f1_05), 4),
        "f1_tuned": round(float(tuned_f1), 4),
        "per_class_05": [round(float(x), 4) for x in final_pc_05],
        "per_class_tuned": [round(float(x), 4) for x in tuned_pc],
        "thresholds": [round(float(t), 3) for t in thr],
        "wall_s": round(elapsed, 1),
    }


# ═══════════════ Main ═══════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  ALTERNATING CL — 5-FOLD CROSS-VALIDATION", flush=True)
    print("  Pipeline: CL consensus → MLP self-confidence → CL consensus → final MLP", flush=True)
    print("=" * 80, flush=True)

    print(f"\n  Config: T5+SPACE+CL (attention-pooled L22, 1538-d)", flush=True)
    print(f"  CL cutoff: {CL_CUTOFF}, min_flags: {MIN_FLAGS}", flush=True)
    print(f"  MLP: {HIDDEN} hidden, dropout {DROP}, LR {LR}, early stop {PAT}", flush=True)

    results = []
    for holdout in range(5):
        r = run_fold(holdout)
        results.append(r)

    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════

    baselines = [r["baseline_f1"] for r in results]
    f1_05s = [r["f1_05"] for r in results]
    f1_tuneds = [r["f1_tuned"] for r in results]

    print(f"\n{'='*80}", flush=True)
    print(f"  5-FOLD CV SUMMARY — ALTERNATING CL", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Holdout':>8}  {'Baseline':>9}  {'Alt@0.5':>9}  {'Alt@tuned':>10}  "
          f"n_train → n_final", flush=True)
    print(f"  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*24}", flush=True)

    for r in results:
        print(f"  P{r['holdout']:>7}  {r['baseline_f1']:>9.4f}  {r['f1_05']:>9.4f}  "
              f"{r['f1_tuned']:>10.4f}  {r['n_train']} → {r['n_cl2']}", flush=True)

    print(f"  {'─'*8}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*24}", flush=True)
    print(f"  {'Mean':>8}  {np.mean(baselines):>9.4f}  {np.mean(f1_05s):>9.4f}  "
          f"{np.mean(f1_tuneds):>10.4f}", flush=True)
    print(f"  {'Std':>8}  {np.std(baselines):>9.4f}  {np.std(f1_05s):>9.4f}  "
          f"{np.std(f1_tuneds):>10.4f}", flush=True)

    # Per-class
    print(f"\n  Per-class F1@tuned (mean ± std across 5 folds):", flush=True)
    alt_per_class_means = []
    alt_per_class_stds = []
    for j, c in enumerate(COMPARTMENTS):
        vals = [r["per_class_tuned"][j] for r in results]
        alt_per_class_means.append(np.mean(vals))
        alt_per_class_stds.append(np.std(vals))
        print(f"    {c:>15s}:  {np.mean(vals):.4f} ± {np.std(vals):.4f}", flush=True)

    # ═══════════════════════════════════════════
    # COMPARISON WITH MANUAL AND HYBRID
    # ═══════════════════════════════════════════

    # Manual 5-fold attn-pooled (from champion_5fold_cv_attn.py output)
    manual_json = PROJ / "output_champion_5fold_cv_attn.json"
    manual_per_fold = {}
    manual_per_class = {}
    if manual_json.exists():
        md = json.loads(manual_json.read_text())
        for f in md["per_fold"]:
            manual_per_fold[f["holdout"]] = f["champion_f1"]
        manual_mean = md["champion_mean"]
        manual_std = md["champion_std"]
        for j, c in enumerate(COMPARTMENTS):
            vals = [f["champion_per_class"][j] for f in md["per_fold"]]
            manual_per_class[c] = (np.mean(vals), np.std(vals))
    else:
        manual_per_fold = {}
        manual_mean = 0; manual_std = 0

    # Hybrid 5-fold attn-pooled (from hybrid_5fold_cv.py output)
    hybrid_json = PROJ / "output_hybrid_5fold_cv.json"
    hybrid_per_fold = {}
    hybrid_per_class = {}
    if hybrid_json.exists():
        hd = json.loads(hybrid_json.read_text())
        for f in hd.get("per_fold", []):
            hybrid_per_fold[f["holdout"]] = f.get("hybrid_f1_tuned", f.get("hybrid_f1_05", 0))
        hybrid_mean = hd.get("hybrid_mean_tuned", hd.get("hybrid_mean_05", 0))
        hybrid_std = hd.get("hybrid_std_tuned", hd.get("hybrid_std_05", 0))
        for j, c in enumerate(COMPARTMENTS):
            vals = [f.get("hybrid_per_class_tuned", f.get("hybrid_per_class_05", [0]*M))[j] for f in hd.get("per_fold", [])]
            hybrid_per_class[c] = (np.mean(vals) if vals else 0, np.std(vals) if vals else 0)
    else:
        hybrid_per_fold = {}
        hybrid_mean = 0; hybrid_std = 0

    alt_mean = np.mean(f1_tuneds)
    alt_std = np.std(f1_tuneds)

    print(f"\n{'─'*75}", flush=True)
    print(f"  VS MANUAL & HYBRID 5-FOLD CV", flush=True)
    print(f"{'─'*75}", flush=True)
    print(f"  {'Fold':>6}  {'Manual':>9}  {'Hybrid':>9}  {'Alternating':>12}  "
          f"{'Δ vs Hyb':>9}  {'Δ vs Man':>9}", flush=True)
    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*12}  {'─'*9}  {'─'*9}", flush=True)

    for holdout in range(5):
        m = manual_per_fold.get(holdout, 0)
        h = hybrid_per_fold.get(holdout, 0)
        a = f1_tuneds[holdout]
        print(f"  P{holdout:>5}  {m:>9.4f}  {h:>9.4f}  {a:>12.4f}  "
              f"{a-h:>+9.4f}  {a-m:>+9.4f}", flush=True)

    print(f"  {'─'*6}  {'─'*9}  {'─'*9}  {'─'*12}  {'─'*9}  {'─'*9}", flush=True)
    print(f"  {'Mean':>6}  {manual_mean:>9.4f}  {hybrid_mean:>9.4f}  "
          f"{alt_mean:>12.4f}  {alt_mean-hybrid_mean:>+9.4f}  "
          f"{alt_mean-manual_mean:>+9.4f}", flush=True)
    print(f"  {'Std':>6}  {manual_std:>9.4f}  {hybrid_std:>9.4f}  "
          f"{alt_std:>12.4f}", flush=True)

    # Per-class comparison
    print(f"\n  Per-class F1@tuned comparison (mean across 5 folds):", flush=True)
    print(f"  {'Compartment':>15s}  {'Manual':>8}  {'Hybrid':>8}  {'Alternating':>12}  "
          f"{'Δ vs Hyb':>9}  {'Δ vs Man':>9}", flush=True)
    print(f"  {'─'*15}  {'─'*8}  {'─'*8}  {'─'*12}  {'─'*9}  {'─'*9}", flush=True)

    for j, c in enumerate(COMPARTMENTS):
        m_val = manual_per_class.get(c, (0,0))[0]
        h_val = hybrid_per_class.get(c, (0,0))[0]
        a_val = alt_per_class_means[j]
        print(f"  {c:>15s}  {m_val:>8.4f}  {h_val:>8.4f}  {a_val:>12.4f}  "
              f"{a_val-h_val:>+9.4f}  {a_val-m_val:>+9.4f}", flush=True)

    print(f"\n  Total wall time: {time.time()-t0:.1f}s ({((time.time()-t0)/60):.1f} min)", flush=True)

    # Save
    out = PROJ / "output_alternating_5fold_cv_attn.json"
    out_data = {
        "pipeline": "CL consensus → MLP self-confidence → CL consensus → final MLP",
        "config": "T5+SPACE+CL (1538-d), 5-fold CV",
        "params": {"min_flags": MIN_FLAGS, "cl_cutoff": CL_CUTOFF, "mlp_hidden": HIDDEN},
        "per_fold": results,
        "alt_mean": round(float(alt_mean), 4),
        "alt_std": round(float(alt_std), 4),
        "alt_per_class": {c: [round(float(alt_per_class_means[j]), 4),
                               round(float(alt_per_class_stds[j]), 4)]
                          for j, c in enumerate(COMPARTMENTS)},
        "comparison": {
            "manual_mean": manual_mean, "manual_std": manual_std,
            "hybrid_mean": hybrid_mean, "hybrid_std": hybrid_std,
            "alternating_mean": round(float(alt_mean), 4),
            "alternating_std": round(float(alt_std), 4),
            "delta_vs_manual": round(float(alt_mean - manual_mean), 4),
            "delta_vs_hybrid": round(float(alt_mean - hybrid_mean), 4),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(out_data, indent=2))
    print(f"  Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
