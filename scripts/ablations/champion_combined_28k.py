#!/usr/bin/env python3
"""champion_combined_28k.py

Full-dimension champion on combined 28,303 proteins.
Merges DeepLoc SwissProt 11,562 new proteins with df_adi 16,741.
Attention-pooled ProtT5 L22 + SPACE + aux for all proteins.
1538-d features with champion config (512 hidden).

Pipeline: baseline → R1 cleanlab → R2 cleanlab → final evaluation
Holdout: df_adi partition 4 (same test set as champion 0.8011)

Usage:
  python3 champion_combined_28k.py

Output:
  output_combined_28k/report.json
"""

import os, time, warnings, json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cleanlab.multilabel_classification.rank import get_label_quality_scores

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.resolve()

# ── Paths ──
NEW_H5    = str(PROJ / "data" / "deeploc_new_attn_pool.h5")       # (11562, 4, 1024)
OLD_H5    = str(PROJ / "data" / "prott5_attn_all_layers.h5")      # (16741, 24, 1024)
DL_CSV    = PROJ / "data" / "deeploc_new_11562_labels.csv"
ADI_CSV   = PROJ / "data" / "df_adi.csv"
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK= PROJ / "data" / "space_network_mask.npy"
DL_SPACE_EMB = PROJ / "data" / "deeploc_space_embeddings.npy"
DL_SPACE_MASK = PROJ / "data" / "deeploc_space_mask.npy"
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"

# ── Label mapping: DeepLoc 11 cols → our 7 cols ──
DL_TO_OURS = {
    'Membrane':      'membrane',
    'Cytoplasm':     'cytoplasm',
    'Nucleus':       'nucleus',
    'Extracellular': 'extracellular',
    'Cell membrane': 'cell_surface',
    'Mitochondrion': 'mitochondrion',
    'Endoplasmic reticulum': 'endom',
    # Our 7 cols have no mapping for: Plastid, Lysosome/Vacuole, Golgi apparatus, Peroxisome
}

OUR_COLS = ['membrane','cytoplasm','nucleus','extracellular','cell_surface','mitochondrion','endom']
M = len(OUR_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]

# ── Config (champion config) ──
HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5; CL_CUTOFF = 0.40
HOLDOUT = 4  # df_adi partition 4


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
            best_f1 = ef; stall = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stall += 1
        if stall >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad(): tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
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
    return oof


def cleanlab_step(Y, oof, cutoff):
    labs = [list(np.where(Y[i]==1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"        Cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)")
    return keep


def load_combined():
    """Load and merge all data into (X_all, Y_all, parts) for 28,303 proteins."""
    print("=" * 65)
    print("  LOADING COMBINED 28k DATASET")
    print("=" * 65)
    
    # ── 1. Load labels ──
    df_dl = pd.read_csv(DL_CSV)
    df_adi = pd.read_csv(ADI_CSV)
    print(f"  df_adi labels:        {len(df_adi):,}")
    print(f"  DeepLoc new labels:   {len(df_dl):,}")
    
    # Map DeepLoc labels to our 7 columns
    dl_mapped = pd.DataFrame(0, index=range(len(df_dl)), columns=OUR_COLS)
    for dl_col, our_col in DL_TO_OURS.items():
        if dl_col in df_dl.columns:
            dl_mapped[our_col] = df_dl[dl_col].values
    df_dl['acc'] = df_dl['ACC'].astype(str)
    df_dl['partition'] = -1  # new proteins: no df_adi partition
    
    # Concatenate labels
    adi_labels = df_adi[['acc'] + OUR_COLS + ['partition']].copy()
    dl_labels = pd.concat([df_dl[['acc']], dl_mapped, df_dl[['partition']]], axis=1)
    df_combined = pd.concat([adi_labels, dl_labels], ignore_index=True)
    print(f"  Combined labels:      {len(df_combined):,}")
    
    Y_all = df_combined[OUR_COLS].values.astype(np.int64)
    parts = df_combined['partition'].values  # -1 for new proteins
    accs_combined = df_combined['acc'].values
    
    # ── 2. Load ProtT5 attn features (L22 only — champion config) ──
    with h5py.File(NEW_H5, 'r') as h:
        prot5_new = h[f'attn_layer_22'][:].astype(np.float32)
    print(f"  New attn (L22):       {prot5_new.shape}")
    
    with h5py.File(OLD_H5, 'r') as h:
        prot5_old = h[f'attn_layer_22'][:].astype(np.float32)
    print(f"  df_adi attn (L22):    {prot5_old.shape}")
    
    prot5_all = np.concatenate([prot5_old, prot5_new], axis=0)
    print(f"  Combined attn:        {prot5_all.shape} ({prot5_all.shape[1]}d)")
    
    # ── 3. SPACE + aux features ──
    # For df_adi proteins: existing SPACE/aux
    # For new DeepLoc proteins: SPACE is mostly absent (zero-pad), aux computed from sequence
    net_emb = np.load(SPACE_EMB); net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    
    # For new proteins: SPACE is zeros (they're not in df_adi's STRING network)
    # Compute heuristic aux features (SP/TMD proxy) for new proteins
    print("  Computing aux features for new proteins...")
    KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
          'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
          'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    def sp_proxy(seq):
        n_region = seq[:30]
        if len(n_region) < 15: return 0.0
        return float(np.mean([KD.get(aa, 0.0) for aa in n_region]) >= 0.5)
    def tmd_count(seq, window=19, thr=1.6):
        if len(seq) < window: return 0.0
        h = np.array([KD.get(aa, 0.0) for aa in seq], dtype=np.float32)
        if len(h) < window: return 0.0
        cs = np.convolve(h, np.ones(window)/window, mode='valid')
        return float((cs >= thr).sum())
    
    new_aux = np.zeros((len(df_dl), 2), dtype=np.float32)
    for i, seq in enumerate(df_dl['Sequence'].values):
        new_aux[i, 0] = sp_proxy(seq)
        new_aux[i, 1] = tmd_count(seq)
    
    # df_adi aux (existing) + new aux
    old_aux = np.load(AUX_FEATS)  # (16741, 2)
    aux_all = np.concatenate([old_aux, new_aux], axis=0)
    print(f"  Aux features:         {aux_all.shape}")
    
    # SPACE: use real embeddings for DeepLoc (extracted via STRING aliases, 92.9% coverage)
    dl_space_emb = np.load(DL_SPACE_EMB)
    dl_space_mask = np.load(DL_SPACE_MASK)
    space_new = dl_space_emb.copy()
    space_new[~dl_space_mask] = 0.0  # zero-pad the ~7% missing
    n_dl_space = int(dl_space_mask.sum())
    print(f"  DeepLoc SPACE coverage: {n_dl_space}/{len(df_dl)} ({100*n_dl_space/len(df_dl):.1f}%)")
    space_all = np.concatenate([net_filled, space_new], axis=0)
    print(f"  SPACE features:       {space_all.shape}")
    
    # ── 4. Concat all features ──
    X_all = np.concatenate([prot5_all, space_all, aux_all], axis=1).astype(np.float32)
    print(f"  Total features:       {X_all.shape} ({X_all.shape[1]}d)")
    print(f"  Total proteins:       {len(X_all):,}")
    
    return X_all, Y_all, parts, accs_combined


def main():
    t0 = time.time()
    print("=" * 72)
    print("  COMBINED 28k CHAMPION — FULL DIM (no PCA)")
    print("  df_adi (16,741) + DeepLoc new (11,562) = 28,303 total")
    print("  ProtT5 attn L22 (1024d) + SPACE (512d) + aux (2d) = 1538d")
    print("  MLP: 1538 → 512 → 7 (champion config)")
    print("=" * 72)
    
    X_all, Y_all, parts, accs_all = load_combined()
    
    # Split: partition 4 as test, everything else (including -1 new proteins) as train
    train_mask = (parts != HOLDOUT)
    test_mask = (parts == HOLDOUT)
    new_mask = (parts == -1)
    train_mask = train_mask | new_mask  # new proteins always used for training
    
    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]
    
    print(f"\n  Train: {len(Y_tr):,}  Test: {len(Y_te):,}")
    print(f"  Features: {X_tr.shape[1]}-d")
    
    # Baseline
    print(f"\n  Baseline ({len(Y_tr):,} train)...")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)
    print(f"  Baseline F1: {base_f1:.4f}")
    
    # R1 OOF + cleanlab
    print(f"\n  Round 1 OOF (4-fold CV)...")
    oof_r1 = gen_oof(X_tr, Y_tr)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    print(f"  R1 kept: {len(Y_r1)}/{len(Y_tr)} ({100*len(Y_r1)/len(Y_tr):.1f}%)")
    
    # R2 OOF + cleanlab
    print(f"\n  Round 2 OOF (4-fold CV)...")
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    print(f"  R2 kept: {len(Y_r2)}/{len(Y_r1)} ({100*len(Y_r2)/len(Y_r1):.1f}%)")
    
    # Final
    print(f"\n  Final ({len(Y_r2):,} train, {len(Y_te):,} test)...")
    final_f1, final_pc, _ = train_mlp(X_r2, Y_r2, X_te, Y_te)
    gain = final_f1 - base_f1
    
    dt = time.time() - t0
    
    # Report
    print(f"\n  {'='*55}")
    print(f"  COMBINED 28K CHAMPION — RESULT (Partition {HOLDOUT})")
    print(f"  {'='*55}")
    print(f"  {'Metric':>25s}  {'Score':>8s}")
    print(f"  {'-'*25}  {'-'*8}")
    print(f"  {'Baseline (28k train)':>25s}  {base_f1:>8.4f}")
    print(f"  {'Champion (28k + CL)':>25s}  {final_f1:>8.4f}")
    print(f"  {'Cleanlab gain':>25s}  {gain:>+8.4f}")
    print(f"  {'Wall time':>25s}  {dt:.0f}s ({dt/60:.1f}m)")
    print(f"  {'Train proteins':>25s}  {len(Y_tr):>8,}")
    print(f"  {'After cleanlab':>25s}  {len(Y_r2):>8,}")
    
    print(f"\n  Per-compartment champion F1:")
    print(f"  {'Compartment':>15s}  {'28k':>8s}  {'MLP*':>8s}  {'Δ':>8s}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")
    mlp_ref = {
        "Membrane": 0.8076, "Cytoplasm": 0.7597, "Nucleus": 0.7948,
        "Extracell": 0.8917, "Cell_surf": 0.7293, "Mito": 0.7870, "Endom": 0.6928,
    }

    for j, c in enumerate(COMPARTMENTS):
        mlp_v = mlp_ref.get(c, 0.0)
        delta = final_pc[j] - mlp_v
        marker = " 🏆" if final_pc[j] > mlp_v + 0.005 else (" 📉" if mlp_v > final_pc[j] + 0.005 else "")
        print(f"  {c:>15s}  {final_pc[j]:>8.4f}  {mlp_v:>8.4f}  {delta:>+8.4f}{marker}")

    champ_p4 = 0.7994
    print(f"\n  {'Overall':>15s}  {final_f1:>8.4f}  {champ_p4:>8.4f}  {final_f1 - champ_p4:>+8.4f}")
    
    # HEAD-TO-HEAD
    print("\n" + "=" * 65)
    print("  HEAD-TO-HEAD (Partition 4)")
    print("=" * 65)
    print(f"  {'Model':>35s}  {'F1-macro':>9s}")
    print(f"  {'-'*35}  {'-'*9}")
    print(f"  {'🏆  28k champion (this run)':>35s}  {final_f1:>9.4f}")
    print(f"  {'    28k baseline (no cleanlab)':>35s}  {base_f1:>9.4f}")
    print(f"  {'🏆  df_adi MLP champion (16k)':>35s}  {0.7994:>9.4f}")
    print(f"  {'    df_adi baseline (13k train)':>35s}  {0.7815:>9.4f}  (≈)")
    print(f"  {'    DeepLoc Accurate (ProtT5-XL)':>35s}  {0.7674:>9.4f}")
    print(f"  {'    DeepLoc Fast (ESM-1b)':>35s}  {0.7491:>9.4f}")
    
    # Save
    report = {
        "model": "Combined 28k champion (ProtT5 attn L20-23 + SPACE + aux)",
        "features": f"1024 + 512 + 2 = {X_tr.shape[1]}d",
        "n_train_total": 28303,
        "n_train_used": int(len(Y_tr)),
        "n_test": int(len(Y_te)),
        "n_after_r1": int(len(Y_r1)),
        "n_after_r2": int(len(Y_r2)),
        "baseline_f1": round(base_f1, 4),
        "champion_f1": round(final_f1, 4),
        "gain": round(gain, 4),
        "baseline_per_class": [round(x, 4) for x in base_pc],
        "champion_per_class": [round(x, 4) for x in final_pc],
        "wall_time_s": round(dt, 1),
        "comparison": {
            "df_adi_mlp_champion": 0.7994,
            "deeploc_accurate": 0.7674,
            "deeploc_fast": 0.7491,
            "vs_df_adi_mlp": round(final_f1 - 0.7994, 4),
            "vs_deeploc_accurate": round(final_f1 - 0.7674, 4),
            "vs_deeploc_fast": round(final_f1 - 0.7491, 4),
        },
    }
    
    out_dir = PROJ / "output_combined_28k"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
