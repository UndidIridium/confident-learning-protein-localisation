#!/usr/bin/env python3
"""hybrid_ablation_p4.py

Hybrid pipeline ablation on P4 - all 5 configs:

  T5 baseline          manual 2-round + tuned thresholds (no CL)
  T5 + CL              hybrid: CL consensus → manual 2-round + tuned
  SPACE baseline       manual 2-round + tuned thresholds (no CL)
  SPACE + CL           hybrid: CL consensus → manual 2-round + tuned
  T5 + SPACE + CL    hybrid: CL consensus → manual 2-round + tuned

Compare: Manual (from output_ablation_tuned.json) vs Hybrid.

Usage:
  python3 hybrid_ablation_p4.py 2>&1 | tee hybrid_ablation.log
  tail -f hybrid_ablation.log
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


# ═══════════════ Sklearn MLP (per-compartment binary - for CleanLearning) ═══════════════

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


# ═══════════════ Manual pipeline (multi-output MLP) ═══════════════

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
        print(f"        [Fold {f+1}/{n_folds}] F1={f1_f:.4f}", flush=True)
    return oof


def cleanlab_step_manual(Y, oof, cutoff):
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"        Manual cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)", flush=True)
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


# ═══════════════ Config runners ═══════════════

def run_manual(config_name, Xtr, Ytr, Xte, Yte):
    """Manual pipeline: gen_oof → cleanlab × 2 → train → tune thresholds."""
    n_tr = len(Ytr)
    print(f"\n  {'─'*50}", flush=True)
    print(f"  [{config_name}]  Manual pipeline", flush=True)

    # Round 1 OOF
    print(f"    Round 1 OOF ({n_tr} proteins)...", flush=True)
    oof_r1 = gen_oof_manual(Xtr, Ytr)
    keep_r1 = cleanlab_step_manual(Ytr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = Xtr[keep_r1], Ytr[keep_r1]
    n_r1 = len(Y_r1)

    # Round 2 OOF
    print(f"    Round 2 OOF ({n_r1} proteins)...", flush=True)
    oof_r2 = gen_oof_manual(X_r1, Y_r1)
    keep_r2 = cleanlab_step_manual(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    n_r2 = len(Y_r2)

    # Final train
    print(f"    Final train ({n_r2} proteins)...", flush=True)
    final_f1_05, final_pc_05, final_tp = train_mlp_manual(X_r2, Y_r2, Xte, Yte)

    # Threshold tuning from round 2 OOF
    oof_tune = oof_r2[keep_r2]
    thr = tune_thresholds(oof_tune, Y_r2)
    tuned_f1, tuned_pc = eval_at_thresholds(final_tp, Yte, thr)

    print(f"    Result: F1@0.5={final_f1_05:.4f}  F1@tuned={tuned_f1:.4f}", flush=True)

    return {
        "method": "manual",
        "n_start": n_tr, "n_r1": n_r1, "n_r2": n_r2,
        "f1_05": round(float(final_f1_05), 4),
        "f1_tuned": round(float(tuned_f1), 4),
        "per_class_05": [round(float(x), 4) for x in final_pc_05],
        "per_class_tuned": [round(float(x), 4) for x in tuned_pc],
        "thresholds": [round(float(t), 3) for t in thr],
    }


def run_hybrid(config_name, Xtr, Ytr, Xte, Yte, indim):
    """Hybrid pipeline: CL consensus → manual 2-round → tune thresholds."""
    n_tr = len(Ytr)
    print(f"\n  {'─'*50}", flush=True)
    print(f"  [{config_name}]  Hybrid pipeline (min_flags={MIN_FLAGS})", flush=True)

    # Step 1: CleanLearning noise detection
    print(f"    CleanLearning × 7 compartments...", flush=True)
    flag_counts = np.zeros(n_tr, dtype=int)
    for j in range(M):
        t_j = time.time()
        clf = SklearnMLP(indim=indim, hidden=HIDDEN, dropout=DROP, lr=LR, max_ep=MAX_EP, patience=PAT, bs=BS)
        cl = CleanLearning(clf=clf, cv_n_folds=4, seed=42, verbose=False)
        cl.fit(Xtr, Ytr[:, j].astype(int))
        flag_counts += cl.label_issues_mask.astype(int)
        n_issues = int(cl.label_issues_mask.sum())
        print(f"      {COMPARTMENTS[j]:>12s}: {n_issues:>5d} flagged ({100*n_issues/n_tr:.1f}%)  [{time.time()-t_j:.1f}s]", flush=True)

    # Step 2: Consensus drop
    keep_cl = flag_counts < MIN_FLAGS
    n_drop_cl = int((~keep_cl).sum())
    print(f"    Flag distribution: {dict(sorted(zip(*np.unique(flag_counts, return_counts=True))))}", flush=True)
    print(f"    Consensus: drop if >= {MIN_FLAGS} flags → {n_drop_cl} dropped, {int(keep_cl.sum())} kept", flush=True)

    X_kp, Y_kp = Xtr[keep_cl], Ytr[keep_cl]
    n_cl = len(Y_kp)

    # Step 3: Manual 2-round cleanlab
    print(f"    Manual Round 1 OOF ({n_cl} proteins)...", flush=True)
    oof_r1 = gen_oof_manual(X_kp, Y_kp)
    keep_r1 = cleanlab_step_manual(Y_kp, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_kp[keep_r1], Y_kp[keep_r1]
    n_r1 = len(Y_r1)

    print(f"    Manual Round 2 OOF ({n_r1} proteins)...", flush=True)
    oof_r2 = gen_oof_manual(X_r1, Y_r1)
    keep_r2 = cleanlab_step_manual(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    n_r2 = len(Y_r2)

    # Step 4: Final train + tune
    print(f"    Final train ({n_r2} proteins)...", flush=True)
    final_f1_05, final_pc_05, final_tp = train_mlp_manual(X_r2, Y_r2, Xte, Yte)

    oof_tune = oof_r2[keep_r2]
    thr = tune_thresholds(oof_tune, Y_r2)
    tuned_f1, tuned_pc = eval_at_thresholds(final_tp, Yte, thr)

    print(f"    Result: F1@0.5={final_f1_05:.4f}  F1@tuned={tuned_f1:.4f}", flush=True)

    return {
        "method": "hybrid",
        "n_start": n_tr, "n_drop_cl": n_drop_cl, "n_cl": n_cl,
        "n_r1": n_r1, "n_r2": n_r2,
        "f1_05": round(float(final_f1_05), 4),
        "f1_tuned": round(float(tuned_f1), 4),
        "per_class_05": [round(float(x), 4) for x in final_pc_05],
        "per_class_tuned": [round(float(x), 4) for x in tuned_pc],
        "thresholds": [round(float(t), 3) for t in thr],
        "flag_distribution": {str(k): int(v) for k, v in zip(*np.unique(flag_counts, return_counts=True))},
    }


# ═══════════════ Main ═══════════════

def main():
    t0 = time.time()
    print("=" * 80, flush=True)
    print("  HYBRID ABLATION - P4 - ALL 5 CONFIGS", flush=True)
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

    mean_vec = net_emb[net_mask].mean(axis=0)
    net_mean = net_emb.copy(); net_mean[~net_mask] = mean_vec
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0

    X_t5    = np.concatenate([prot5, aux], axis=1).astype(np.float32)
    X_space = np.concatenate([net_mean, aux], axis=1).astype(np.float32)
    X_t5s   = np.concatenate([prot5, net_filled, aux], axis=1).astype(np.float32)

    # Configs: (name, X_tr, X_te, indim, use_hybrid)
    configs = [
        ("T5 only",            X_t5[train_mask],    X_t5[test_mask],    X_t5.shape[1],    False),
        ("T5 + CL",            X_t5[train_mask],    X_t5[test_mask],    X_t5.shape[1],    True),
        ("SPACE only",         X_space[train_mask], X_space[test_mask], X_space.shape[1], False),
        ("SPACE + CL",         X_space[train_mask], X_space[test_mask], X_space.shape[1], True),
        ("T5 + SPACE + CL ", X_t5s[train_mask],   X_t5s[test_mask],   X_t5s.shape[1],   True),
    ]

    results = {}
    for cfg_name, Xtr, Xte, indim, use_cl in configs:
        t_cfg = time.time()
        print(f"\n{'='*70}", flush=True)
        print(f"  [{cfg_name}]  indim={indim}  hybrid={use_cl}", flush=True)
        print(f"{'='*70}", flush=True)

        if use_cl:
            r = run_hybrid(cfg_name, Xtr, Y_tr, Xte, Y_te, indim)
        else:
            r = run_manual(cfg_name, Xtr, Y_tr, Xte, Y_te)

        r["wall_s"] = round(time.time() - t_cfg, 1)
        results[cfg_name] = r

    # ═══ Summary ════════════════════════════════════════════════════════
    # Load manual results for comparison
    manual_json = PROJ / "output_ablation_tuned.json"
    manual_lookup = {}
    if manual_json.exists():
        raw = json.loads(manual_json.read_text())
        for cfg, v in raw.items():
            if cfg.startswith("_"): continue  # skip metadata keys like _description
            if isinstance(v, dict):
                manual_lookup[cfg] = v.get("baseline_tuned", v.get("best_f1", v.get("tuned", 0)))
            else:
                try:
                    manual_lookup[cfg] = float(v)
                except (ValueError, TypeError):
                    continue
    else:
        manual_lookup = {"T5 only": 0.7826, "T5 + CL": 0.7809, "SPACE only": 0.7292,
                         "SPACE + CL": 0.7054, "T5 + SPACE + CL ": 0.7994}

    print(f"\n{'='*95}", flush=True)
    print(f"  HYBRID ABLATION RESULTS - P4  (min_flags={MIN_FLAGS})", flush=True)
    print(f"{'='*95}", flush=True)
    print(f"  {'Config':<25s}  {'Method':>7}  {'Start':>7}  {'CLdrop':>7}  {'R1':>7}  {'R2':>7}  {'F1@0.5':>8}  {'F1@tuned':>8}  {'Manual':>8}  {'Δ':>8}", flush=True)
    print(f"  {'─'*25}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}", flush=True)

    for cfg_name, r in results.items():
        method = r["method"]
        n_s = r.get("n_start", n_tr)
        n_cl = r.get("n_drop_cl", 0)
        n_r1 = r.get("n_r1", 0)
        n_r2 = r["n_r2"]
        f1_05 = r["f1_05"]
        f1_tuned = r["f1_tuned"]
        manual_f1 = manual_lookup.get(cfg_name, 0)
        delta = f1_tuned - manual_f1 if manual_f1 else 0

        print(f"  {cfg_name:<25s}  {method:>7}  {n_s:>7}  {n_cl:>7}  {n_r1:>7}  {n_r2:>7}  "
              f"{f1_05:>8.4f}  {f1_tuned:>8.4f}  {manual_f1:>8.4f}  {delta:>+8.4f}", flush=True)

    # Per-class breakdown
    print(f"\n  Per-class breakdown (F1@tuned):", flush=True)
    print(f"  {'Compartment':>15s}", end="", flush=True)
    for cfg_name in results:
        print(f"  {cfg_name[:18]:>18s}", end="", flush=True)
    print(flush=True)
    print(f"  {'─'*15}", end="", flush=True)
    for _ in results:
        print(f"  {'─'*18}", end="", flush=True)
    print(flush=True)
    for j, c in enumerate(COMPARTMENTS):
        print(f"  {c:>15s}", end="", flush=True)
        for cfg_name, r in results.items():
            pc = r.get("per_class_tuned", [0]*M)
            print(f"  {pc[j]:>18.4f}", end="", flush=True)
        print(flush=True)

    print(f"\n  Wall time: {time.time()-t0:.1f}s ({((time.time()-t0)/60):.1f} min)", flush=True)

    # Save
    out = PROJ / "output_hybrid_ablation_p4.json"
    out_data = {
        "min_flags": MIN_FLAGS,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "flag_distribution"} for k, v in results.items()},
        "manual_baseline": manual_lookup,
        "wall_s": round(time.time() - t0, 1),
    }
    out.write_text(json.dumps(out_data, indent=2))
    print(f"  Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
