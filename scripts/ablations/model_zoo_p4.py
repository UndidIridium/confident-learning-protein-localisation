#!/usr/bin/env python3
"""model_zoo_p4.py

Systematic classifier comparison on df_adi partition 4.
ALL models use identical features (T5+SPACE+aux = 1538d) and
identical training data (P0-P3, 2-round cleanlab cleaned).

Models: MLP (baseline), RandomForest, XGBoost, LightGBM, AdaBoost,
        LogisticRegression, ExtraTrees, GradientBoosting, CatBoost.

For each: hyperparameter tuning → 4-fold OOF → per-class threshold
tune → final train on all cleaned data → predict P4 → F1@0.5 + F1@tuned.

Usage:
  python3 model_zoo_p4.py

Output: output_model_zoo_p4.json + stdout report
"""

import json, os, time, warnings
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import torch
torch.set_num_threads(1)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import clone

# sklearn classifiers
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Boosting libraries
import xgboost as xgb
import lightgbm as lgb

# CatBoost - optional
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING:  CatBoost not installed - skipping. Install: pip install catboost")

import torch, torch.nn as nn, torch.optim as optim
from cleanlab.multilabel_classification.rank import get_label_quality_scores

# ═══ Paths ═════════════════════════════════════════════════════════════

PROJ = Path(__file__).parent.resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
PROT5_H5 = str(PROJ / "data" / "prott5_all_layers_dfadi-3.h5")
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"

# ═══ Constants ════════════════════════════════════════════════════════

LAYER = 22
HOLDOUT = 4
CL_CUTOFF = 0.40
THR_GRID = np.arange(0.02, 0.96, 0.02)
N_ITER_SEARCH = 30   # RandomizedSearchCV iterations per model
CV_FOLDS = 4
SEED = 42

LABEL_COLS = ["membrane", "cytoplasm", "nucleus", "extracellular",
              "cell_surface", "mitochondrion", "endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane", "Cytoplasm", "Nucleus", "Extracell",
                "Cell_surf", "Mito", "Endom"]


# ═══ MLP (our champion baseline) ══════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, hdim), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(hdim, outdim),
        )

    def forward(self, x):
        return self.net(x)


def posw(Y):
    pw = np.ones(M, dtype=np.float32)
    for j in range(M):
        pos = float(Y[:, j].sum())
        neg = float(Y.shape[0]) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg / pos)
    return np.clip(pw, 1.0, 20.0)


def train_mlp(Xtr, Ytr, Xte, Yte, seed=42, hidden=512, dropout=0.5,
              lr=1e-4, max_ep=50, patience=5, bs=256):
    es_frac = 0.10
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr).astype(np.float32)
    Xtes = sc.transform(Xte).astype(np.float32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    ti, ei = train_test_split(np.arange(len(Xts)), test_size=es_frac,
                              random_state=seed)
    pw = posw(Ytr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw.astype(np.float32)))
    model = MLP(Xts.shape[1], hidden, M, dropout)
    opt = optim.Adam(model.parameters(), lr=lr)
    Xt = torch.from_numpy(Xts)
    Yt = torch.from_numpy(Ytr.astype(np.float32))
    Xe = torch.from_numpy(Xts[ei])
    Ye = Ytr[ei]
    best_f1, best_state, stall = -1.0, None, 0
    for ep in range(1, max_ep + 1):
        model.train()
        perm = torch.randperm(len(ti))
        for s in range(0, len(ti), bs):
            ix = perm[s:s + bs]
            criterion(model(Xt[ix]), Yt[ix]).backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        with torch.no_grad():
            ep_ = torch.sigmoid(model(Xe)).numpy()
        ef = float(np.mean([
            f1_score(Ye[:, j].astype(int), (ep_[:, j] >= 0.5).astype(int),
                     zero_division=0)
            for j in range(M)
        ]))
        if ef > best_f1 + 1e-6:
            best_f1 = ef
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    pr = (tp >= 0.5).astype(int)
    pc = [float(f1_score(Yte[:, j].astype(int), pr[:, j], zero_division=0))
          for j in range(M)]
    return float(np.mean(pc)), pc, tp


def gen_oof_mlp(X, Y, n_folds=4, seed=42, label=""):
    n = len(X)
    oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    fs = n // n_folds
    for f in range(n_folds):
        vs = f * fs
        ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]
        ti = np.concatenate([idx[:vs], idx[ve:]])
        t0f = time.time()
        _, _, tp = train_mlp(X[ti], Y[ti], X[vi], Y[vi], seed=seed + f)
        f1_f = float(np.mean([f1_score(Y[vi][:,j].astype(int), (tp[:,j]>=0.5).astype(int), zero_division=0) for j in range(M)]))
        print(f"    {label}[Fold {f+1}/{n_folds}] F1={f1_f:.4f}  [{time.time()-t0f:.0f}s]", flush=True)
        oof[vi] = tp
    return oof


# ═══ Helper: cleanlab step (shared) ═══════════════════════════════════

def cleanlab_step(Y, oof, cutoff):
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(
        labels=labs, pred_probs=oof.astype(np.float64),
        method="self_confidence", adjust_pred_probs=True,
    )
    keep = scores >= cutoff
    return keep


# ═══ Helper: per-class threshold tuning ═══════════════════════════════

def tune_thresholds(oof, Y):
    """Tune per-class decision thresholds on OOF predictions."""
    best = np.full(M, 0.5, dtype=np.float32)
    for j in range(M):
        cands = np.array([
            f1_score(Y[:, j].astype(int), (oof[:, j] >= t).astype(int),
                     zero_division=0)
            for t in THR_GRID
        ])
        best[j] = THR_GRID[int(cands.argmax())]
    return best


def eval_at_thresholds(probs, Y, thresholds):
    preds = (probs >= thresholds).astype(int)
    pc = [float(f1_score(Y[:, j].astype(int), preds[:, j], zero_division=0))
          for j in range(M)]
    return float(np.mean(pc)), pc


# ═══ Generic 7-binary-classifier OOF generator ════════════════════════

def gen_oof_binary(X, Y, train_fn, n_folds=4, seed=42):
    """Generate OOF by training 7 classifiers via k-fold CV.

    train_fn(Xtr, Ytr_col, Xte) -> test_probs (n_te,)
    """
    n = len(X)
    oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    fs = n // n_folds
    for f in range(n_folds):
        vs = f * fs
        ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]
        ti = np.concatenate([idx[:vs], idx[ve:]])
        for j in range(M):
            oof[vi, j] = train_fn(X[ti], Y[ti, j], X[vi])
    return oof


# ═══ Model-specific training functions ════════════════════════════════

# --- XGBoost ---
def _train_xgb_binary(Xtr, Ytr_col, Xte, params):
    pos = float(Ytr_col.sum())
    neg = float(len(Ytr_col) - pos)
    spw = max(1.0, min(20.0, neg / max(pos, 1.0)))
    tr_ix, va_ix = train_test_split(
        np.arange(len(Xtr)), test_size=0.10, random_state=SEED,
        stratify=Ytr_col if pos > 0 else None,
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=spw,
        verbosity=0,
        n_jobs=4,
        random_state=SEED,
        **params,
    )
    model.fit(
        Xtr[tr_ix], Ytr_col[tr_ix],
        eval_set=[(Xtr[va_ix], Ytr_col[va_ix])],
        verbose=False,
    )
    return model.predict_proba(Xte)[:, 1].astype(np.float32)


def gen_oof_xgb(X, Y, params, n_folds=4, seed=42):
    return gen_oof_binary(
        X, Y,
        lambda Xtr, yc, Xte: _train_xgb_binary(Xtr, yc, Xte, params),
        n_folds, seed,
    )


def train_xgb_final(Xtr, Ytr, Xte, Yte, params):
    """Train 7 XGBoost binary classifiers, return (mean_f1, per_class, probs)."""
    n_tr = len(Xtr)
    probs = np.zeros((len(Xte), M), dtype=np.float32)
    for j in range(M):
        pos = float(Ytr[:, j].sum())
        neg = float(n_tr - pos)
        spw = max(1.0, min(20.0, neg / max(pos, 1.0)))
        tr_ix, va_ix = train_test_split(
            np.arange(n_tr), test_size=0.10, random_state=SEED + j,
            stratify=Ytr[:, j] if pos > 0 else None,
        )
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            verbosity=0,
            n_jobs=4,
            random_state=SEED + j,
            **params,
        )
        model.fit(
            Xtr[tr_ix], Ytr[tr_ix, j],
            eval_set=[(Xtr[va_ix], Ytr[va_ix, j])],
            verbose=False,
        )
        probs[:, j] = model.predict_proba(Xte)[:, 1].astype(np.float32)
    preds = (probs >= 0.5).astype(int)
    pc = [float(f1_score(Yte[:, j].astype(int), preds[:, j], zero_division=0))
          for j in range(M)]
    return float(np.mean(pc)), pc, probs


# --- LightGBM ---
def _train_lgb_binary(Xtr, Ytr_col, Xte, params):
    pos = float(Ytr_col.sum())
    neg = float(len(Ytr_col) - pos)
    spw = max(1.0, min(20.0, neg / max(pos, 1.0)))
    tr_ix, va_ix = train_test_split(
        np.arange(len(Xtr)), test_size=0.10, random_state=SEED,
        stratify=Ytr_col if pos > 0 else None,
    )
    model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=spw,
        verbose=-1,
        n_jobs=4,
        random_state=SEED,
        **params,
    )
    model.fit(
        Xtr[tr_ix], Ytr_col[tr_ix],
        eval_set=[(Xtr[va_ix], Ytr_col[va_ix])],
    )
    return model.predict_proba(Xte)[:, 1].astype(np.float32)


def gen_oof_lgb(X, Y, params, n_folds=4, seed=42):
    return gen_oof_binary(
        X, Y,
        lambda Xtr, yc, Xte: _train_lgb_binary(Xtr, yc, Xte, params),
        n_folds, seed,
    )


def train_lgb_final(Xtr, Ytr, Xte, Yte, params):
    n_tr = len(Xtr)
    probs = np.zeros((len(Xte), M), dtype=np.float32)
    for j in range(M):
        pos = float(Ytr[:, j].sum())
        neg = float(n_tr - pos)
        spw = max(1.0, min(20.0, neg / max(pos, 1.0)))
        tr_ix, va_ix = train_test_split(
            np.arange(n_tr), test_size=0.10, random_state=SEED + j,
            stratify=Ytr[:, j] if pos > 0 else None,
        )
        model = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=spw,
            verbose=-1,
            n_jobs=4,
            random_state=SEED + j,
            **params,
        )
        model.fit(
            Xtr[tr_ix], Ytr[tr_ix, j],
            eval_set=[(Xtr[va_ix], Ytr[va_ix, j])],
        )
        probs[:, j] = model.predict_proba(Xte)[:, 1].astype(np.float32)
    preds = (probs >= 0.5).astype(int)
    pc = [float(f1_score(Yte[:, j].astype(int), preds[:, j], zero_division=0))
          for j in range(M)]
    return float(np.mean(pc)), pc, probs


# --- CatBoost ---
if HAS_CATBOOST:
    def _train_cb_binary(Xtr, Ytr_col, Xte, params):
        pos = float(Ytr_col.sum())
        neg = float(len(Ytr_col) - pos)
        sw = np.ones(len(Ytr_col), dtype=np.float32)
        if pos > 0:
            sw[Ytr_col == 1] = min(20.0, neg / pos)
        tr_ix, va_ix = train_test_split(
            np.arange(len(Xtr)), test_size=0.10, random_state=SEED,
            stratify=Ytr_col if pos > 0 else None,
        )
        model = cb.CatBoostClassifier(
            loss_function="Logloss",
            verbose=False,
            random_seed=SEED,
            thread_count=4,
            **params,
        )
        model.fit(
            Xtr[tr_ix], Ytr_col[tr_ix],
            sample_weight=sw[tr_ix],
            eval_set=(Xtr[va_ix], Ytr_col[va_ix]),
            verbose=False,
        )
        return model.predict_proba(Xte)[:, 1].astype(np.float32)

    def gen_oof_cb(X, Y, params, n_folds=4, seed=42):
        return gen_oof_binary(
            X, Y,
            lambda Xtr, yc, Xte: _train_cb_binary(Xtr, yc, Xte, params),
            n_folds, seed,
        )

    def train_cb_final(Xtr, Ytr, Xte, Yte, params):
        n_tr = len(Xtr)
        probs = np.zeros((len(Xte), M), dtype=np.float32)
        for j in range(M):
            pos = float(Ytr[:, j].sum())
            neg = float(n_tr - pos)
            sw = np.ones(n_tr, dtype=np.float32)
            if pos > 0:
                sw[Ytr[:, j] == 1] = min(20.0, neg / pos)
            tr_ix, va_ix = train_test_split(
                np.arange(n_tr), test_size=0.10, random_state=SEED + j,
                stratify=Ytr[:, j] if pos > 0 else None,
            )
            model = cb.CatBoostClassifier(
                loss_function="Logloss",
                verbose=False,
                random_seed=SEED + j,
                thread_count=4,
                **params,
            )
            model.fit(
                Xtr[tr_ix], Ytr[tr_ix, j],
                sample_weight=sw[tr_ix],
                eval_set=(Xtr[va_ix], Ytr[va_ix, j]),
                verbose=False,
            )
            probs[:, j] = model.predict_proba(Xte)[:, 1].astype(np.float32)
        preds = (probs >= 0.5).astype(int)
        pc = [float(f1_score(Yte[:, j].astype(int), preds[:, j], zero_division=0))
              for j in range(M)]
        return float(np.mean(pc)), pc, probs


# ═══ Hyperparameter grids ═════════════════════════════════════════════

PARAM_GRIDS = {
    "RandomForest": {
        "estimator__n_estimators": [100, 200, 400, 600, 800],
        "estimator__max_depth": [None, 10, 20, 30, 50],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 4],
        "estimator__max_features": ["sqrt", "log2", None],
        "estimator__class_weight": ["balanced", "balanced_subsample", None],
    },
    "ExtraTrees": {
        "estimator__n_estimators": [100, 200, 400, 600, 800],
        "estimator__max_depth": [None, 10, 20, 30, 50],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 4],
        "estimator__max_features": ["sqrt", "log2", None],
        "estimator__class_weight": ["balanced", "balanced_subsample", None],
    },
    "GradientBoosting": {
        "estimator__n_estimators": [100, 200, 400],
        "estimator__max_depth": [3, 5, 7, 10],
        "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "estimator__min_samples_split": [2, 5, 10],
        "estimator__min_samples_leaf": [1, 2, 4],
        "estimator__subsample": [0.6, 0.8, 1.0],
    },
    "AdaBoost": {
        "estimator__n_estimators": [50, 100, 200, 400],
        "estimator__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
        "estimator__estimator__max_depth": [1, 2, 3, 5],
    },
    "LogisticRegression": {
        "estimator__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "estimator__penalty": ["l2"],
        "estimator__solver": ["lbfgs", "saga"],
        "estimator__max_iter": [500, 1000, 2000],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.0, 0.1, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 400, 600],
        "max_depth": [3, 5, 7, 10, -1],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_samples": [5, 10, 20, 50],
        "reg_lambda": [0.0, 0.1, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "num_leaves": [15, 31, 63, 127],
    },
}
if HAS_CATBOOST:
    PARAM_GRIDS["CatBoost"] = {
        "iterations": [100, 200, 400, 600],
        "depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "l2_leaf_reg": [1, 3, 5, 10],
        "border_count": [32, 64, 128],
    }


# ═══ Sklearn OOF: manual 4-fold to avoid cross_val_predict shape ambiguity ═

def _gen_oof_sklearn(X, Y, fitted_clf, n_folds=4, seed=42):
    """Generate OOF for a MultiOutputClassifier via manual k-fold CV.

    Clones the base estimator from fitted_clf each fold (not the fitted model)
    to avoid data leakage. Returns (n_samples, M) float32 array of probas.
    """
    n = len(X)
    oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    fs = n // n_folds

    # Extract base estimator class and params from the fitted MultiOutputClassifier
    base_template = fitted_clf.estimator

    for f in range(n_folds):
        vs = f * fs
        ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]
        ti = np.concatenate([idx[:vs], idx[ve:]])

        # Clone fresh model for this fold
        fold_clf = MultiOutputClassifier(clone(base_template), n_jobs=4)
        fold_clf.fit(X[ti], Y[ti])
        tp_full = fold_clf.predict_proba(X[vi])

        if isinstance(tp_full, list):
            oof[vi] = np.column_stack([p[:, 1] for p in tp_full]).astype(np.float32)
        else:
            oof[vi] = np.array([tp_full[i][:, 1] for i in range(M)]).T.astype(np.float32)

    return oof


# ═══ Tune a sklearn MultiOutputClassifier ═════════════════════════════

def tune_sklearn_multi(X, Y, name, n_iter=N_ITER_SEARCH):
    """Tune a sklearn classifier wrapped in MultiOutputClassifier."""
    grid = PARAM_GRIDS[name]
    if name == "RandomForest":
        base = RandomForestClassifier(random_state=SEED, n_jobs=4)
    elif name == "ExtraTrees":
        base = ExtraTreesClassifier(random_state=SEED, n_jobs=4)
    elif name == "GradientBoosting":
        base = GradientBoostingClassifier(random_state=SEED)
    elif name == "AdaBoost":
        # Handle sklearn < 1.2 (base_estimator) vs >= 1.2 (estimator)
        import sklearn
        sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if sk_ver >= (1, 2):
            base = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(), random_state=SEED,
            )
        else:
            base = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(), random_state=SEED,
            )
    elif name == "LogisticRegression":
        base = LogisticRegression(
            class_weight="balanced", random_state=SEED, max_iter=2000,
        )
    else:
        raise ValueError(f"Unknown sklearn model: {name}")

    clf = MultiOutputClassifier(base, n_jobs=1)  # n_jobs=1 to avoid nested parallelism
    search = RandomizedSearchCV(
        clf, grid, n_iter=n_iter, cv=3, scoring="f1_macro",
        random_state=SEED, n_jobs=4, verbose=0,
    )
    search.fit(X, Y)
    print(f"    Best params: {search.best_params_}", flush=True)
    print(f"    Best CV F1:  {search.best_score_:.4f}", flush=True)
    return search.best_estimator_, search.best_params_


# ═══ Tune XGBoost (tune on one compartment for speed) ═════════════════

def tune_xgb(X, Y, n_iter=N_ITER_SEARCH):
    """Tune XGBoost on Cytoplasm (largest class) for speed."""
    grid = PARAM_GRIDS["XGBoost"]
    j_ref = 1  # Cytoplasm
    pos = float(Y[:, j_ref].sum())
    neg = float(len(Y) - pos)
    spw = max(1.0, min(20.0, neg / max(pos, 1.0)))
    base = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        scale_pos_weight=spw, verbosity=0, n_jobs=4, random_state=SEED,
    )
    search = RandomizedSearchCV(
        base, grid, n_iter=n_iter, cv=3, scoring="f1",
        random_state=SEED, n_jobs=4, verbose=0,
    )
    search.fit(X, Y[:, j_ref])
    best = search.best_params_
    # Remove CV-specific params that XGBoost stores
    best.pop("early_stopping_rounds", None)
    print(f"    Best params: {best}", flush=True)
    print(f"    Best CV F1:  {search.best_score_:.4f}  (on Cytoplasm)", flush=True)
    return best


def tune_lgb(X, Y, n_iter=N_ITER_SEARCH):
    """Tune LightGBM on Cytoplasm for speed."""
    grid = PARAM_GRIDS["LightGBM"]
    j_ref = 1
    pos = float(Y[:, j_ref].sum())
    neg = float(len(Y) - pos)
    spw = max(1.0, min(20.0, neg / max(pos, 1.0)))
    base = lgb.LGBMClassifier(
        objective="binary", scale_pos_weight=spw,
        verbose=-1, n_jobs=4, random_state=SEED,
    )
    search = RandomizedSearchCV(
        base, grid, n_iter=n_iter, cv=3, scoring="f1",
        random_state=SEED, n_jobs=4, verbose=0,
    )
    search.fit(X, Y[:, j_ref])
    best = search.best_params_
    print(f"    Best params: {best}", flush=True)
    print(f"    Best CV F1:  {search.best_score_:.4f}  (on Cytoplasm)", flush=True)
    return best


if HAS_CATBOOST:
    def tune_cb(X, Y, n_iter=N_ITER_SEARCH):
        grid = PARAM_GRIDS["CatBoost"]
        j_ref = 1
        pos = float(Y[:, j_ref].sum())
        neg = float(len(Y) - pos)
        sw = np.ones(len(Y), dtype=np.float32)
        if pos > 0:
            sw[Y[:, j_ref] == 1] = min(20.0, neg / pos)
        base = cb.CatBoostClassifier(
            loss_function="Logloss", verbose=False,
            random_seed=SEED, thread_count=4,
        )
        search = RandomizedSearchCV(
            base, grid, n_iter=n_iter, cv=3, scoring="f1",
            random_state=SEED, n_jobs=4, verbose=0,
        )
        search.fit(X, Y[:, j_ref], sample_weight=sw)
        best = search.best_params_
        print(f"    Best params: {best}", flush=True)
        print(f"    Best CV F1:  {search.best_score_:.4f}  (on Cytoplasm)", flush=True)
        return best


# ═══ Run a full pipeline for one model ════════════════════════════════

def run_model(name, X_tr_clean, Y_tr_clean, X_te, Y_te):
    """Run full pipeline: tune → OOF → thresholds → final train → evaluate.

    Returns dict with all results.
    """
    t0 = time.time()
    print(f"\n{'─'*60}", flush=True)
    print(f"  {name}", flush=True)
    print(f"{'─'*60}", flush=True)

    result = {"name": name, "n_train": len(Y_tr_clean), "n_test": len(Y_te)}

    # --- Tune ---
    print(f"  Tuning hyperparameters ({N_ITER_SEARCH} iters)...", flush=True)
    if name == "MLP":
        # MLP: use fixed champion params (already tuned via architecture sweep)
        best_params = {"hidden": 512, "dropout": 0.5, "lr": 1e-4,
                       "max_ep": 50, "patience": 5, "bs": 256}
        print(f"    Fixed champion params: {best_params}", flush=True)
    elif name in ("RandomForest", "ExtraTrees", "GradientBoosting",
                   "AdaBoost", "LogisticRegression"):
        best_est, best_params = tune_sklearn_multi(X_tr_clean, Y_tr_clean, name)
    elif name == "XGBoost":
        best_params = tune_xgb(X_tr_clean, Y_tr_clean)
    elif name == "LightGBM":
        best_params = tune_lgb(X_tr_clean, Y_tr_clean)
    elif name == "CatBoost" and HAS_CATBOOST:
        best_params = tune_cb(X_tr_clean, Y_tr_clean)
    else:
        raise ValueError(f"Unknown model: {name}")

    result["best_params"] = {k: str(v) for k, v in best_params.items()}

    # --- OOF for threshold tuning ---
    print(f"  Generating OOF (4-fold CV)...", flush=True)
    if name == "MLP":
        oof = gen_oof_mlp(X_tr_clean, Y_tr_clean, label="")
    elif name in ("RandomForest", "ExtraTrees", "GradientBoosting",
                   "AdaBoost", "LogisticRegression"):
        # Manual 4-fold OOF to avoid cross_val_predict shape ambiguity
        oof = _gen_oof_sklearn(X_tr_clean, Y_tr_clean, best_est)
    elif name == "XGBoost":
        oof = gen_oof_xgb(X_tr_clean, Y_tr_clean, best_params)
    elif name == "LightGBM":
        oof = gen_oof_lgb(X_tr_clean, Y_tr_clean, best_params)
    elif name == "CatBoost" and HAS_CATBOOST:
        oof = gen_oof_cb(X_tr_clean, Y_tr_clean, best_params)
    else:
        raise ValueError(f"Unknown model: {name}")

    # --- Tune thresholds ---
    thresholds = tune_thresholds(oof, Y_tr_clean)
    f1_oof_05, _ = eval_at_thresholds(oof, Y_tr_clean, np.full(M, 0.5))
    f1_oof_tuned, _ = eval_at_thresholds(oof, Y_tr_clean, thresholds)
    result["f1_oof_05"] = round(float(f1_oof_05), 4)
    result["f1_oof_tuned"] = round(float(f1_oof_tuned), 4)
    result["thresholds"] = [round(float(t), 3) for t in thresholds]
    print(f"    OOF F1@0.5={f1_oof_05:.4f}  F1@tuned={f1_oof_tuned:.4f}", flush=True)

    # --- Final train + predict ---
    print(f"  Training final model on all {len(Y_tr_clean)} proteins...", flush=True)
    if name == "MLP":
        f1_05, pc_05, tp = train_mlp(X_tr_clean, Y_tr_clean, X_te, Y_te)
    elif name in ("RandomForest", "ExtraTrees", "GradientBoosting",
                   "AdaBoost", "LogisticRegression"):
        best_est.fit(X_tr_clean, Y_tr_clean)
        tp_full = best_est.predict_proba(X_te)
        if isinstance(tp_full, list):
            tp = np.column_stack([p[:, 1] for p in tp_full]).astype(np.float32)
        else:
            tp = np.array([tp_full[i][:, 1] for i in range(M)]).T.astype(np.float32)
        preds = (tp >= 0.5).astype(int)
        pc_05 = [float(f1_score(Y_te[:, j].astype(int), preds[:, j], zero_division=0))
                 for j in range(M)]
        f1_05 = float(np.mean(pc_05))
    elif name == "XGBoost":
        f1_05, pc_05, tp = train_xgb_final(X_tr_clean, Y_tr_clean, X_te, Y_te, best_params)
    elif name == "LightGBM":
        f1_05, pc_05, tp = train_lgb_final(X_tr_clean, Y_tr_clean, X_te, Y_te, best_params)
    elif name == "CatBoost" and HAS_CATBOOST:
        f1_05, pc_05, tp = train_cb_final(X_tr_clean, Y_tr_clean, X_te, Y_te, best_params)
    else:
        raise ValueError(f"Unknown model: {name}")

    f1_tuned, pc_tuned = eval_at_thresholds(tp, Y_te, thresholds)

    dt = time.time() - t0
    result["f1_05"] = round(float(f1_05), 4)
    result["f1_tuned"] = round(float(f1_tuned), 4)
    result["per_class_05"] = [round(float(x), 4) for x in pc_05]
    result["per_class_tuned"] = [round(float(x), 4) for x in pc_tuned]
    result["wall_time_s"] = round(dt, 1)

    print(f"    Test F1@0.5={f1_05:.4f}  F1@tuned={f1_tuned:.4f}  "
          f"[{dt:.0f}s]", flush=True)
    return result


# ═══ Main ═════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("  MODEL ZOO - Classifier Comparison on df_adi P4", flush=True)
    print("  ALL models: same features (T5+SPACE+aux=1538d), same data", flush=True)
    print("=" * 70, flush=True)

    # ── Load data ──
    print("\nLoading data...", flush=True)
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != HOLDOUT)
    test_mask = (parts == HOLDOUT)
    n_tr = int(train_mask.sum())
    n_te = int(test_mask.sum())
    print(f"  Train: {n_tr} (P0-P3)  Test: {n_te} (P{HOLDOUT})", flush=True)

    # ── Build features ──
    with h5py.File(PROT5_H5, "r") as f:
        prot5 = f[f"df_adi_layer_{LAYER:02d}"][:].astype(np.float32)

    net_emb = np.load(SPACE_EMB)
    net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy()
    net_filled[~net_mask] = 0.0

    aux_feats = np.load(AUX_FEATS)
    assert aux_feats.shape[0] == len(src)

    X_all = np.concatenate([prot5, net_filled, aux_feats], axis=1).astype(np.float32)
    FEAT_DIM = X_all.shape[1]
    print(f"  Features: {FEAT_DIM}-d (ProtT5 {prot5.shape[1]} + SPACE "
          f"{net_filled.shape[1]} + aux {aux_feats.shape[1]})", flush=True)

    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]

    # ── 2-round cleanlab (MLP OOF) → fixed cleaned training set ──
    print(f"\n  Running 2-round cleanlab (MLP OOF)...", flush=True)
    oof_r1 = gen_oof_mlp(X_tr, Y_tr, label="R1 ")
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    n_r1 = len(Y_r1)
    print(f"    R1: {n_r1} kept ({100 * n_r1 / n_tr:.1f}%)", flush=True)

    oof_r2 = gen_oof_mlp(X_r1, Y_r1, label="R2 ")
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    n_r2 = len(Y_r2)
    print(f"    R2: {n_r2} kept ({100 * n_r2 / n_r1:.1f}%) - "
          f"total retention {100 * n_r2 / n_tr:.1f}%", flush=True)

    # Store OOF from R2 for MLP threshold baseline
    oof_mlp = oof_r2[keep_r2]

    # ── Model list ──
    MODELS = [
        "MLP",
        "LogisticRegression",
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "AdaBoost",
        "XGBoost",
        "LightGBM",
    ]
    if HAS_CATBOOST:
        MODELS.append("CatBoost")

    # ── Run all models ──
    results = []
    for name in MODELS:
        try:
            r = run_model(name, X_r2, Y_r2, X_te, Y_te)
            results.append(r)
        except Exception as e:
            print(f"     {name} FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results.append({"name": name, "error": str(e)})

    # ── Print summary table ──
    print("\n" + "=" * 75, flush=True)
    print("  MODEL ZOO - FINAL RESULTS (df_adi P4)", flush=True)
    print("=" * 75, flush=True)
    header = f"  {'Model':>22s}  {'F1@0.5':>7s}  {'F1@tuned':>8s}  {'Gain':>7s}  {'OOF@tuned':>9s}  {'Time':>6s}"
    print(header, flush=True)
    print(f"  {'─'*22}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*6}", flush=True)

    mlp_tuned = None
    for r in results:
        if "error" in r:
            print(f"  {r['name']:>22s}   FAILED: {r['error'][:40]}", flush=True)
        else:
            gain = r["f1_tuned"] - r["f1_05"]
            print(f"  {r['name']:>22s}  {r['f1_05']:>7.4f}  {r['f1_tuned']:>8.4f}  "
                  f"{gain:>+7.4f}  {r['f1_oof_tuned']:>9.4f}  {r['wall_time_s']:>5.0f}s",
                  flush=True)
            if r["name"] == "MLP":
                mlp_tuned = r["f1_tuned"]

    print(f"  {'─'*22}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*9}  {'─'*6}", flush=True)

    # ── Per-class breakdown ──
    if mlp_tuned is not None:
        print(f"\n  Reference: MLP champion = {mlp_tuned:.4f}, "
              f"DeepLoc Accurate = 0.7674, DeepLoc Fast = 0.7491", flush=True)

    print(f"\n  Per-compartment F1@tuned:", flush=True)
    comp_header = f"  {'Compartment':>15s}"
    for r in results:
        if "error" not in r:
            comp_header += f"  {r['name']:>7s}"
    print(comp_header, flush=True)
    print(f"  {'─'*15}" + f"  {'─'*7}" * len([r for r in results if "error" not in r]),
          flush=True)

    for j, comp in enumerate(COMPARTMENTS):
        row = f"  {comp:>15s}"
        for r in results:
            if "error" not in r:
                row += f"  {r['per_class_tuned'][j]:>7.4f}"
        print(row, flush=True)

    # ── Save ──
    report = {
        "features": f"ProtT5 L{LAYER} + SPACE + aux = {FEAT_DIM}d",
        "training_set": f"df_adi P0-P3, 2-round cleanlab (MLP OOF, cutoff={CL_CUTOFF})",
        "n_train_original": n_tr,
        "n_train_cleaned": n_r2,
        "n_test": n_te,
        "holdout_partition": HOLDOUT,
        "cleanlab_retention_pct": round(100 * n_r2 / n_tr, 1),
        "models": results,
        "reference_mlp_champion": 0.8011,
        "reference_deeploc_accurate": 0.7674,
        "reference_deeploc_fast": 0.7491,
        "total_wall_time_s": round(time.time() - t0, 1),
    }

    out_path = PROJ / "output_model_zoo_p4.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved: {out_path}", flush=True)
    print(f"  Total wall time: {(time.time() - t0) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
