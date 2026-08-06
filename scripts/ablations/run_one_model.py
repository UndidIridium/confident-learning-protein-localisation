#!/usr/bin/env python3
"""run_one_model.py <ModelName> — fast classifier on cleaned data.

No OOF — trains on all cleaned data, tunes thresholds via 80/20 split on training set.
7 independent binary classifiers, single-threaded, fast estimators.
"""

import json, os, sys, time, warnings
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings("ignore")
import torch; torch.set_num_threads(1)

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
try: import catboost as cb; HAS_CATBOOST = True
except ImportError: HAS_CATBOOST = False

import torch.nn as nn, torch.optim as optim

PROJ = Path(__file__).parent.resolve()
M = 7; SEED = 42
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]
THR_GRID = np.arange(0.02, 0.96, 0.02)

# ═══ MLP ══════════════════════════════════════════════════════════════
class MLP(nn.Module):
    def __init__(self,i,h,o,d): super().__init__(); self.net=nn.Sequential(nn.Linear(i,h),nn.ReLU(True),nn.Dropout(d),nn.Linear(h,o))
    def forward(self,x): return self.net(x)

def posw(Y):
    pw=np.ones(M,np.float32)
    for j in range(M):
        pos=float(Y[:,j].sum()); neg=float(Y.shape[0])-pos
        pw[j]=1.0 if pos<=0 else min(20.0,neg/pos)
    return np.clip(pw,1.0,20.0)

def train_mlp(Xtr,Ytr,Xte,Yte,seed=42):
    H=512; D=0.5; LR=1e-4; MX=50; P=5; BS=256; EF=0.10
    sc=StandardScaler(); Xts=sc.fit_transform(Xtr).astype(np.float32); Xtes=sc.transform(Xte).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    ti,ei=train_test_split(np.arange(len(Xts)),test_size=EF,random_state=seed)
    pw=posw(Ytr); model=MLP(Xts.shape[1],H,M,D)
    opt=optim.Adam(model.parameters(),lr=LR)
    crit=nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw.astype(np.float32)))
    Xt=torch.from_numpy(Xts); Yt=torch.from_numpy(Ytr.astype(np.float32))
    Xe=torch.from_numpy(Xts[ei]); Ye=Ytr[ei]
    bf1,bs,st=-1.0,None,0
    for ep in range(1,MX+1):
        model.train(); perm=torch.randperm(len(ti))
        for s in range(0,len(ti),BS):
            ix=perm[s:s+BS]; crit(model(Xt[ix]),Yt[ix]).backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad(): ep_=torch.sigmoid(model(Xe)).numpy()
        ef=float(np.mean([f1_score(Ye[:,j].astype(int),(ep_[:,j]>=0.5).astype(int),zero_division=0) for j in range(M)]))
        if ef>bf1+1e-6: bf1=ef; bs={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; st=0
        else: st+=1
        if st>=P: break
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad(): tp=torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    pc=[float(f1_score(Yte[:,j].astype(int),(tp[:,j]>=0.5).astype(int),zero_division=0)) for j in range(M)]
    return float(np.mean(pc)),pc,tp

# ═══ 7-Binary Classifier Factory ═════════════════════════════════════

def make_binary_classifier(name, j):
    if name == "LogisticRegression":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED+j)
    elif name == "AdaBoost":
        skv = tuple(int(x) for x in __import__("sklearn").__version__.split(".")[:2])
        kw = {"n_estimators": 50, "learning_rate": 0.1, "random_state": SEED+j}
        kw["estimator" if skv >= (1,2) else "base_estimator"] = DecisionTreeClassifier(max_depth=2)
        return AdaBoostClassifier(**kw)
    elif name == "XGBoost":
        return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                 n_estimators=50, max_depth=6, learning_rate=0.1,
                                 subsample=0.8, colsample_bytree=0.8, verbosity=0,
                                 n_jobs=1, nthread=1, random_state=SEED+j)
    elif name == "LightGBM":
        return lgb.LGBMClassifier(objective="binary", n_estimators=50, max_depth=7,
                                  learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                  min_child_samples=20, verbose=-1, n_jobs=1, num_threads=1, random_state=SEED+j)
    elif name == "CatBoost":
        return cb.CatBoostClassifier(loss_function="Logloss", iterations=50, depth=6,
                                     learning_rate=0.1, l2_leaf_reg=3, verbose=False,
                                     thread_count=1, random_seed=SEED+j)
    else:
        raise ValueError(f"Unknown: {name}")

# ═══ Train 7 binary + tune thresholds on validation split ════════════

def train_and_tune(X_tr, Y_tr, model_name):
    """Train 7 binary classifiers on 80% of training data, tune thresholds on 20%."""
    n = len(X_tr)
    # Split training data 80/20 for threshold tuning
    tr_idx, tune_idx = train_test_split(np.arange(n), test_size=0.20, random_state=SEED)
    Xtr, Ytr = X_tr[tr_idx], Y_tr[tr_idx]
    Xtu, Ytu = X_tr[tune_idx], Y_tr[tune_idx]

    # Train 7 binary classifiers on the 80% split
    probs_val = np.zeros((len(Xtu), M), np.float32)
    n_tr = len(Xtr)
    for j in range(M):
        pos = float(Ytr[:,j].sum()); neg = float(n_tr - pos)
        spw = max(1.0, min(20.0, neg / max(pos, 1.0)))
        clf = make_binary_classifier(model_name, j)

        # Set class weights
        if model_name == "XGBoost": clf.set_params(scale_pos_weight=spw)
        elif model_name == "LightGBM": clf.set_params(scale_pos_weight=spw)

        # Sub-split for early stopping (XGBoost/LGBM/CatBoost)
        if model_name in ("XGBoost", "LightGBM", "CatBoost"):
            ti2, vi2 = train_test_split(np.arange(n_tr), test_size=0.10, random_state=SEED+j,
                                        stratify=Ytr[:,j] if pos > 0 else None)
            if model_name == "CatBoost":
                sw = np.ones(n_tr, np.float32)
                if pos > 0: sw[Ytr[:,j]==1] = spw
                clf.fit(Xtr[ti2], Ytr[ti2,j], sample_weight=sw[ti2],
                        eval_set=(Xtr[vi2], Ytr[vi2,j]), verbose=False)
            else:
                clf.fit(Xtr[ti2], Ytr[ti2,j], eval_set=[(Xtr[vi2], Ytr[vi2,j])])
        else:
            clf.fit(Xtr, Ytr[:,j])

        probs_val[:,j] = clf.predict_proba(Xtu)[:,1].astype(np.float32)

    # Tune thresholds on validation split
    thr = np.full(M, 0.5, np.float32)
    for j in range(M):
        cands = np.array([f1_score(Ytu[:,j].astype(int), (probs_val[:,j] >= t).astype(int), zero_division=0) for t in THR_GRID])
        thr[j] = THR_GRID[int(cands.argmax())]

    f1_val_05 = float(np.mean([f1_score(Ytu[:,j].astype(int), (probs_val[:,j] >= 0.5).astype(int), zero_division=0) for j in range(M)]))
    f1_val_tuned = float(np.mean([f1_score(Ytu[:,j].astype(int), (probs_val[:,j] >= thr[j]).astype(int), zero_division=0) for j in range(M)]))

    return thr, f1_val_05, f1_val_tuned

# ═══ Final train on all data ══════════════════════════════════════════

def train_final(X_tr, Y_tr, X_te, Y_te, model_name, thresholds):
    """Train 7 binary classifiers on ALL cleaned training data, predict test."""
    if model_name == "MLP":
        f1_05, pc_05, tp = train_mlp(X_tr, Y_tr, X_te, Y_te)
    else:
        n_tr = len(X_tr)
        tp = np.zeros((len(X_te), M), np.float32)
        for j in range(M):
            pos = float(Y_tr[:,j].sum()); neg = float(n_tr - pos)
            spw = max(1.0, min(20.0, neg / max(pos, 1.0)))
            clf = make_binary_classifier(model_name, j)
            if model_name == "XGBoost": clf.set_params(scale_pos_weight=spw)
            elif model_name == "LightGBM": clf.set_params(scale_pos_weight=spw)

            if model_name == "CatBoost":
                sw = np.ones(n_tr, np.float32)
                if pos > 0: sw[Y_tr[:,j]==1] = spw
                clf.fit(X_tr, Y_tr[:,j], sample_weight=sw, verbose=False)
            else:
                clf.fit(X_tr, Y_tr[:,j])
            tp[:,j] = clf.predict_proba(X_te)[:,1].astype(np.float32)

        preds = (tp >= 0.5).astype(int)
        pc_05 = [float(f1_score(Y_te[:,j].astype(int), preds[:,j], zero_division=0)) for j in range(M)]
        f1_05 = float(np.mean(pc_05))

    # Apply tuned thresholds
    preds_tuned = (tp >= thresholds).astype(int)
    pc_tuned = [float(f1_score(Y_te[:,j].astype(int), preds_tuned[:,j], zero_division=0)) for j in range(M)]
    f1_tuned = float(np.mean(pc_tuned))

    return f1_05, pc_05, f1_tuned, pc_tuned

# ═══ Main ═════════════════════════════════════════════════════════════
if len(sys.argv) < 2: print("Usage: python3 run_one_model.py <ModelName>"); sys.exit(1)

model_name = sys.argv[1]
t0 = time.time()
print(f"\n{'='*60}", flush=True)
print(f"  {model_name}", flush=True)
print(f"{'='*60}", flush=True)

data_dir = PROJ / "cleaned_data"
X_tr = np.load(data_dir / "X_train_cleaned.npy").astype(np.float32)
Y_tr = np.load(data_dir / "Y_train_cleaned.npy").astype(np.int64)
X_te = np.load(data_dir / "X_test.npy").astype(np.float32)
Y_te = np.load(data_dir / "Y_test.npy").astype(np.int64)
print(f"  Train: {X_tr.shape}  Test: {X_te.shape}", flush=True)

if model_name == "CatBoost" and not HAS_CATBOOST:
    print("  CatBoost not installed"); sys.exit(1)

# Phase 1: Train on 80% + tune thresholds on 20%
print(f"  Training 7 binary classifiers + threshold tuning (80/20 split)...", flush=True)
thr, val_05, val_tuned = train_and_tune(X_tr, Y_tr, model_name)
print(f"  Val: F1@0.5={val_05:.4f}  F1@tuned={val_tuned:.4f}  thr={[round(float(t),2) for t in thr]}", flush=True)

# Phase 2: Final train on all data + predict test
print(f"  Training final model on all {len(Y_tr)} proteins...", flush=True)
f1_05, pc_05, f1_tuned, pc_tuned = train_final(X_tr, Y_tr, X_te, Y_te, model_name, thr)
dt = time.time() - t0

print(f"  Test: F1@0.5={f1_05:.4f}  F1@tuned={f1_tuned:.4f}  [{dt:.0f}s]", flush=True)
for j, c in enumerate(COMPARTMENTS):
    print(f"    {c:>15s}: {pc_tuned[j]:.4f}", flush=True)

result = {
    "model": model_name, "n_train": len(Y_tr), "n_test": len(Y_te),
    "f1_05": round(float(f1_05), 4), "f1_tuned": round(float(f1_tuned), 4),
    "f1_val_05": round(float(val_05), 4), "f1_val_tuned": round(float(val_tuned), 4),
    "thresholds": [round(float(t), 3) for t in thr],
    "per_class_05": [round(float(x), 4) for x in pc_05],
    "per_class_tuned": [round(float(x), 4) for x in pc_tuned],
    "wall_time_s": round(dt, 1),
}
(PROJ / f"output_one_model_{model_name}.json").write_text(json.dumps(result, indent=2))
print(f"  Saved: output_one_model_{model_name}.json", flush=True)
