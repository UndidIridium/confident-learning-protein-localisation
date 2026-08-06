#!/usr/bin/env python3
"""alternating_cl_p4.py

NEW PIPELINE: CL consensus → MLP self-confidence cleanlab → CL consensus → final MLP

Compare against:
  - Manual:      2-round MLP self-confidence (0.7994)
  - Hybrid:      CL → manual R1 → manual R2 (0.7996)

Usage:
  python3 alternating_cl_p4.py 2>&1 | tee alternating_cl_p4.log
  tail -f alternating_cl_p4.log
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
        print(f"        [Fold {f+1}/{n_folds}] F1={f1_f:.4f}", flush=True)
    return oof


def cleanlab_self_confidence(Y, oof, cutoff):
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"        Self-confidence: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)", flush=True)
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
        print(f"      {COMPARTMENTS[j]:>12s}: {n_issues:>5d} flagged ({100*n_issues/n:.1f}%)  [{time.time()-t_j:.1f}s]", flush=True)

    keep = flag_counts < MIN_FLAGS
    n_drop = int((~keep).sum())
    print(f"    Flag distribution: {dict(sorted(zip(*np.unique(flag_counts, return_counts=True))))}", flush=True)
    print(f"    Consensus (min_flags={MIN_FLAGS}): {n_drop} dropped, {int(keep.sum())} kept ({100*int(keep.sum())/n:.1f}%)", flush=True)
    return keep, flag_counts


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


# ═══════════════ Main ═══════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  ALTERNATING CL PIPELINE — P4 — T5+SPACE+CL", flush=True)
    print("  CL consensus → MLP self-confidence → CL consensus → final MLP", flush=True)
    print("=" * 80, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != 4); test_mask = (parts == 4)
    Y_tr, Y_te = Y_all[train_mask], Y_all[test_mask]
    n_tr, n_te = train_mask.sum(), test_mask.sum()
    print(f"  Train: {n_tr}  Test: {n_te}", flush=True)

    with h5py.File(ATTN_H5, "r") as f:
        prot5 = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)
    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    aux = np.load(AUX_FEATS)

    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    X_tr = np.concatenate([prot5[train_mask], net_filled[train_mask], aux[train_mask]], axis=1).astype(np.float32)
    X_te = np.concatenate([prot5[test_mask], net_filled[test_mask], aux[test_mask]], axis=1).astype(np.float32)
    indim = X_tr.shape[1]
    print(f"  Features: {indim}-d (T5 attn L22 + SPACE zero-pad + aux)", flush=True)

    # ═══════════════════════════════════════════════════════
    # PIPELINE: CL → MLP → CL → final
    # ═══════════════════════════════════════════════════════

    # Step 1: CL consensus on raw data
    print(f"\n{'─'*60}", flush=True)
    print(f"  STEP 1: CleanLearning consensus on raw data ({n_tr} proteins)", flush=True)
    print(f"{'─'*60}", flush=True)
    keep_cl1, flags1 = cleanlearning_consensus(X_tr, Y_tr, indim)
    X_cl1, Y_cl1 = X_tr[keep_cl1], Y_tr[keep_cl1]
    n_cl1 = len(Y_cl1)

    # Step 2: MLP OOF + self-confidence cleanlab
    print(f"\n{'─'*60}", flush=True)
    print(f"  STEP 2: MLP self-confidence cleanlab ({n_cl1} proteins)", flush=True)
    print(f"{'─'*60}", flush=True)
    oof_mlp = gen_oof(X_cl1, Y_cl1)
    keep_mlp = cleanlab_self_confidence(Y_cl1, oof_mlp, CL_CUTOFF)
    X_mlp, Y_mlp = X_cl1[keep_mlp], Y_cl1[keep_mlp]
    n_mlp = len(Y_mlp)

    # Step 3: CL consensus AGAIN on MLP-cleaned data
    print(f"\n{'─'*60}", flush=True)
    print(f"  STEP 3: CleanLearning consensus on MLP-cleaned data ({n_mlp} proteins)", flush=True)
    print(f"{'─'*60}", flush=True)
    keep_cl2, flags2 = cleanlearning_consensus(X_mlp, Y_mlp, indim)
    X_cl2, Y_cl2 = X_mlp[keep_cl2], Y_mlp[keep_cl2]
    n_cl2 = len(Y_cl2)

    # Step 4: Final MLP train + threshold tuning
    print(f"\n{'─'*60}", flush=True)
    print(f"  STEP 4: Final MLP train ({n_cl2} proteins)", flush=True)
    print(f"{'─'*60}", flush=True)
    final_f1_05, final_pc_05, final_tp = train_mlp(X_cl2, Y_cl2, X_te, Y_te)

    # Threshold tuning from step 2 OOF (on kept proteins)
    oof_tune = oof_mlp[keep_mlp]
    # Need to further subset to CL2-kept proteins for threshold tuning
    oof_tune_final = oof_tune[keep_cl2]
    thr = tune_thresholds(oof_tune_final, Y_cl2)
    tuned_f1, tuned_pc = eval_at_thresholds(final_tp, Y_te, thr)

    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════

    print(f"\n{'='*80}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  Pipeline: CL → MLP self-conf → CL → final MLP", flush=True)
    print(f"", flush=True)
    print(f"  Protein flow:", flush=True)
    print(f"    Start:              {n_tr:>6d}", flush=True)
    print(f"    After CL #1:        {n_cl1:>6d}  ({100*n_cl1/n_tr:.1f}%)  dropped {n_tr-n_cl1}", flush=True)
    print(f"    After MLP cleanlab: {n_mlp:>6d}  ({100*n_mlp/n_cl1:.1f}%)  dropped {n_cl1-n_mlp}", flush=True)
    print(f"    After CL #2:        {n_cl2:>6d}  ({100*n_cl2/n_mlp:.1f}%)  dropped {n_mlp-n_cl2}", flush=True)
    print(f"", flush=True)
    print(f"  Scores:", flush=True)
    print(f"    F1@0.5:         {final_f1_05:.4f}", flush=True)
    print(f"    F1@tuned:        {tuned_f1:.4f}", flush=True)
    print(f"", flush=True)
    print(f"  Per-compartment (F1@tuned):", flush=True)
    for j, c in enumerate(COMPARTMENTS):
        print(f"    {c:>15s}:  {tuned_pc[j]:.4f}", flush=True)
    print(f"", flush=True)
    print(f"  Thresholds:", flush=True)
    for j, c in enumerate(COMPARTMENTS):
        print(f"    {c:>15s}:  {thr[j]:.3f}", flush=True)

    # Comparison with known baselines
    print(f"\n{'─'*60}", flush=True)
    print(f"  COMPARISON (T5+SPACE+CL, P4)", flush=True)
    print(f"{'─'*60}", flush=True)
    print(f"  {'Method':<40s}  {'F1@tuned':>8}", flush=True)
    print(f"  {'─'*40}  {'─'*8}", flush=True)
    print(f"  {'Manual 2-round MLP self-confidence':<40s}  {0.7994:>8.4f}", flush=True)
    print(f"  {'Hybrid CL → manual R1 → manual R2':<40s}  {0.7996:>8.4f}", flush=True)
    print(f"  {'Alternating CL → MLP → CL → final':<40s}  {tuned_f1:>8.4f}", flush=True)
    print(f"  {'─'*40}  {'─'*8}", flush=True)
    print(f"  {'Δ vs Manual':<40s}  {tuned_f1-0.7994:>+8.4f}", flush=True)
    print(f"  {'Δ vs Hybrid':<40s}  {tuned_f1-0.7996:>+8.4f}", flush=True)

    print(f"\n  Wall time: {time.time()-t0:.1f}s ({((time.time()-t0)/60):.1f} min)", flush=True)

    # Save
    out = PROJ / "output_alternating_cl_p4.json"
    out_data = {
        "pipeline": "CL consensus → MLP self-confidence → CL consensus → final MLP",
        "config": "T5+SPACE+CL (1538-d), P4",
        "protein_flow": {
            "start": int(n_tr),
            "after_cl1": int(n_cl1), "drop_cl1": int(n_tr - n_cl1),
            "after_mlp": int(n_mlp), "drop_mlp": int(n_cl1 - n_mlp),
            "after_cl2": int(n_cl2), "drop_cl2": int(n_mlp - n_cl2),
            "final_retention_pct": round(100*n_cl2/n_tr, 1),
        },
        "f1_05": round(float(final_f1_05), 4),
        "f1_tuned": round(float(tuned_f1), 4),
        "per_class_tuned": [round(float(x), 4) for x in tuned_pc],
        "thresholds": [round(float(t), 3) for t in thr],
        "comparison": {
            "manual": 0.7994,
            "hybrid": 0.7996,
            "alternating": round(float(tuned_f1), 4),
            "delta_vs_manual": round(float(tuned_f1) - 0.7994, 4),
            "delta_vs_hybrid": round(float(tuned_f1) - 0.7996, 4),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(out_data, indent=2))
    print(f"  Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
