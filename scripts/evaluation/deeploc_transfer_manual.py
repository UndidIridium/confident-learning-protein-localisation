#!/usr/bin/env python3
"""deeploc_transfer_manual.py

Manual 2-round MLP cleanlab pipeline trained on DeepLoc (11,562 SwissProt proteins),
tested on df_adi P4. True held-out transfer — no df_adi training data used.

Both sides use attention-pooled ProtT5 L22 (only option for DeepLoc).
SPACE is zero-filled for DeepLoc proteins (no SPACE coverage).

Compare: df_adi-trained manual on P4 (0.7994 attn-pooled) vs DeepLoc-trained.

Usage:
  python3 deeploc_transfer_manual.py 2>&1 | tee deeploc_transfer_manual.log
  tail -f deeploc_transfer_manual.log
"""

import json, os, time, warnings
from pathlib import Path
import h5py, numpy as np, pandas as pd

PROJ = Path(__file__).parent.resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
DL_LABELS = PROJ / "data" / "deeploc_new_11562_labels.csv"
DL_H5 = str(PROJ / "data" / "deeploc_new_attn_pool.h5")
ADI_H5 = str(PROJ / "data" / "prott5_attn_all_layers.h5")
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
DL_SPACE_EMB = PROJ / "data" / "deeploc_space_embeddings.npy"
DL_SPACE_MASK = PROJ / "data" / "deeploc_space_mask.npy"
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torch, torch.nn as nn, torch.optim as optim
from cleanlab.multilabel_classification.rank import get_label_quality_scores

M = 7; LAYER = 22
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]
OUR_COLS = ["membrane","cytoplasm","nucleus","extracellular","cell_surface","mitochondrion","endom"]
DL_TO_OURS = {"Membrane":"membrane","Cytoplasm":"cytoplasm","Nucleus":"nucleus",
              "Extracellular":"extracellular","Cell membrane":"cell_surface",
              "Mitochondrion":"mitochondrion","Endoplasmic reticulum":"endom"}
HIDDEN = 512; DROP = 0.5; LR = 1e-4; MAX_EP = 50; PAT = 5; BS = 256; ES_FRAC = 0.10
CL_CUTOFF = 0.40
THR_GRID = np.arange(0.02, 0.96, 0.02)

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
        print(f"        [Fold {f+1}/4] F1={f1_f:.4f}", flush=True)
    return oof


def cleanlab_step(Y, oof, cutoff):
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"        Cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)", flush=True)
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


# ═══════════════ Main ═══════════════

t0 = time.time()
print("=" * 75, flush=True)
print("  MANUAL PIPELINE — DEEPLOC TRANSFER TEST", flush=True)
print("  Train: DeepLoc new (11,562)  |  Test: df_adi P4 (3,276)", flush=True)
print("  Pipeline: Manual 2-round MLP self-confidence → tuned thresholds", flush=True)
print("  Embeddings: Attention-pooled ProtT5 L22 (both sides)", flush=True)
print("=" * 75, flush=True)

# ── Load df_adi (test only) ──
print("\nLoading df_adi...", flush=True)
adi = pd.read_csv(SRC_CSV)
Y_te = adi[adi["partition"] == 4][OUR_COLS].values.astype(np.int64)
print(f"  Test proteins: {len(Y_te)}", flush=True)

# ── Load DeepLoc training data ──
print("Loading DeepLoc training data...", flush=True)
dl = pd.read_csv(DL_LABELS)

# Map labels
Y_dl = np.zeros((len(dl), M), dtype=np.int64)
for dl_col, our_col in DL_TO_OURS.items():
    if dl_col in dl.columns:
        Y_dl[:, OUR_COLS.index(our_col)] = dl[dl_col].values.astype(np.int64)
print(f"  DeepLoc train: {len(Y_dl)} proteins", flush=True)
for j, c in enumerate(COMPARTMENTS):
    print(f"    {c:>15s}: {int(Y_dl[:, j].sum()):>6,d} positive ({100*Y_dl[:, j].sum()/len(Y_dl):.1f}%)", flush=True)

# ── Build features ──
print("Building features...", flush=True)

# ProtT5 for DeepLoc (attn-pooled L22)
with h5py.File(DL_H5, "r") as f:
    prot5_dl = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)

# SPACE for df_adi test
net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
space_test = net_filled[adi["partition"] == 4]

# SPACE for DeepLoc: use real embeddings extracted via STRING aliases (92.9% coverage)
dl_space_emb = np.load(DL_SPACE_EMB)
dl_space_mask = np.load(DL_SPACE_MASK)
space_dl = dl_space_emb.copy()
space_dl[~dl_space_mask] = 0.0  # zero-pad the ~7% missing
n_dl_space = int(dl_space_mask.sum())
print(f"  DeepLoc SPACE coverage: {n_dl_space}/{len(dl)} ({100*n_dl_space/len(dl):.1f}%)", flush=True)

# Aux for df_adi test
aux_all = np.load(AUX_FEATS)
aux_test = aux_all[adi["partition"] == 4]

# Aux for DeepLoc: compute from sequences
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
      'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
      'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
def sp_proxy(seq):
    n_region = str(seq)[:30]
    if len(n_region) < 15: return 0.0
    return float(np.mean([KD.get(aa, 0.0) for aa in n_region]) >= 0.5)
def tmd_count(seq, window=19, thr=1.6):
    seq = str(seq)
    if len(seq) < window: return 0.0
    h = np.array([KD.get(aa, 0.0) for aa in seq], dtype=np.float32)
    if len(h) < window: return 0.0
    cs = np.convolve(h, np.ones(window)/window, mode='valid')
    return float((cs >= thr).sum())

aux_dl = np.zeros((len(dl), 2), dtype=np.float32)
for i, seq in enumerate(dl["Sequence"].values):
    aux_dl[i, 0] = sp_proxy(seq)
    aux_dl[i, 1] = tmd_count(seq)

print(f"  DeepLoc aux: sp_mean={aux_dl[:,0].mean():.3f}, tmd_mean={aux_dl[:,1].mean():.3f}", flush=True)

# Concat DeepLoc features
X_tr = np.concatenate([prot5_dl, space_dl, aux_dl], axis=1).astype(np.float32)

# ProtT5 for df_adi test (attn-pooled L22 — same feature space)
with h5py.File(ADI_H5, "r") as f:
    prot5_adi = f[f"attn_layer_{LAYER:02d}"][:].astype(np.float32)
prot5_test = prot5_adi[adi["partition"] == 4]
X_te = np.concatenate([prot5_test, space_test, aux_test], axis=1).astype(np.float32)

print(f"  Train: X_tr={X_tr.shape}  Y_tr={Y_dl.shape}", flush=True)
print(f"  Test:  X_te={X_te.shape}  Y_te={Y_te.shape}", flush=True)

# ═══ Baseline (no cleaning) ═══
print(f"\n{'─'*50}", flush=True)
print("  BASELINE (no cleaning)", flush=True)
bf1, bpc, _ = train_mlp(X_tr, Y_dl, X_te, Y_te)
print(f"  Baseline F1@0.5: {bf1:.4f}", flush=True)

# ═══ Manual 2-round cleanlab ═══
print(f"\n{'─'*50}", flush=True)
print("  MANUAL 2-ROUND CLEANLAB", flush=True)

# Round 1
print(f"  Round 1 OOF ({len(Y_dl)} proteins)...", flush=True)
oof_r1 = gen_oof(X_tr, Y_dl)
keep_r1 = cleanlab_step(Y_dl, oof_r1, CL_CUTOFF)
X_r1, Y_r1 = X_tr[keep_r1], Y_dl[keep_r1]
n_r1 = len(Y_r1)

# Round 2
print(f"  Round 2 OOF ({n_r1} proteins)...", flush=True)
oof_r2 = gen_oof(X_r1, Y_r1)
keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
n_r2 = len(Y_r2)

# Final train
print(f"  Final train ({n_r2} proteins)...", flush=True)
final_f1_05, final_pc_05, final_tp = train_mlp(X_r2, Y_r2, X_te, Y_te)

# Threshold tuning
oof_tune = oof_r2[keep_r2]
thr = tune_thresholds(oof_tune, Y_r2)
tuned_f1, tuned_pc = eval_at_thresholds(final_tp, Y_te, thr)

# ═══ Results ═══
print(f"\n{'='*65}", flush=True)
print(f"  DEEPLOC TRANSFER RESULTS — MANUAL PIPELINE", flush=True)
print(f"{'='*65}", flush=True)
print(f"  Baseline (no cleaning):           {bf1:.4f}", flush=True)
print(f"  Manual F1@0.5:                    {final_f1_05:.4f}", flush=True)
print(f"  Manual F1@tuned:                  {tuned_f1:.4f}", flush=True)
print(f"  Gain vs baseline (tuned):         {tuned_f1-bf1:+.4f}", flush=True)
print(f"", flush=True)
print(f"  Protein counts: {len(Y_dl)} → {n_r1} (R1) → {n_r2} (R2)", flush=True)
print(f"  Retention: {100*n_r2/len(Y_dl):.1f}%", flush=True)
print(f"", flush=True)
print(f"  ─── Comparison ───", flush=True)
print(f"  df_adi-trained manual P4 (attn):  0.7994", flush=True)
print(f"  DeepLoc-trained manual P4:        {tuned_f1:.4f}", flush=True)
print(f"  Δ (transfer − df_adi-trained):    {tuned_f1-0.7994:+.4f}", flush=True)
print(f"", flush=True)
print(f"  df_adi-trained manual 5-fold:     0.7838 ± 0.0156", flush=True)
print(f"  Δ (transfer − 5-fold mean):       {tuned_f1-0.7838:+.4f}", flush=True)

print(f"\n  Per-compartment (F1@tuned):", flush=True)
for j, c in enumerate(COMPARTMENTS):
    print(f"    {c:>15s}:  {tuned_pc[j]:.4f}", flush=True)

print(f"\n  Thresholds: {[round(float(t), 3) for t in thr]}", flush=True)
print(f"  Wall time: {time.time()-t0:.1f}s ({((time.time()-t0)/60):.1f}m)", flush=True)

# Save
out = PROJ / "output_deeploc_transfer_manual.json"
out.write_text(json.dumps({
    "train_data": "DeepLoc new (11,562 SwissProt)",
    "test_data": "df_adi P4 (3,276)",
    "pipeline": "Manual 2-round MLP self-confidence @0.40 → tuned thresholds",
    "embeddings": "Attention-pooled ProtT5 L22 (both sides)",
    "features": "1538-d (T5 1024 + SPACE zero-filled 512 + aux 2)",
    "baseline_f1": round(float(bf1), 4),
    "manual_f1_05": round(float(final_f1_05), 4),
    "manual_f1_tuned": round(float(tuned_f1), 4),
    "gain_vs_baseline": round(float(tuned_f1 - bf1), 4),
    "delta_vs_df_adi_trained_p4": round(float(tuned_f1 - 0.7994), 4),
    "delta_vs_df_adi_5fold_mean": round(float(tuned_f1 - 0.7838), 4),
    "per_class_tuned": [round(float(x), 4) for x in tuned_pc],
    "thresholds": [round(float(t), 3) for t in thr],
    "protein_counts": {"start": len(Y_dl), "after_r1": n_r1, "after_r2": n_r2},
    "retention_pct": round(100*n_r2/len(Y_dl), 1),
    "wall_s": round(time.time() - t0, 1),
}, indent=2))
print(f"  Saved: {out}", flush=True)
