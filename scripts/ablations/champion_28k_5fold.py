#!/usr/bin/env python3
"""champion_28k_5fold.py

Full 5-fold CV on the combined 28k dataset (df_adi 16,741 + DeepLoc new 11,562).
Uses df_adi's partition column (0-4) for fold splits.
New DeepLoc proteins (partition=-1) always go into training.
PCA 100d for speed.
"""
import h5py, numpy as np, pandas as pd, warnings, json, time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.optim as optim
warnings.filterwarnings('ignore')
from pathlib import Path

PROJ = Path(__file__).parent.resolve()
OUR_COLS = ['membrane','cytoplasm','nucleus','extracellular','cell_surface','mitochondrion','endom']
M = len(OUR_COLS)
HIDDEN = 256; DROPOUT = 0.5; LR = 1e-4; MAX_EP = 50; PATIENCE = 5
BATCH_SIZE = 256; ES_FRAC = 0.10; THR = 0.5; CL_CUTOFF = 0.40; N_COMP = 100
FOLDS = [0, 1, 2, 3, 4]

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

from cleanlab.multilabel_classification.rank import get_label_quality_scores
def cleanlab_step(Y, oof, cutoff):
    labs = [list(np.where(Y[i]==1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                      method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    print(f"      Cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)")
    return keep

def load_combined():
    """Load combined 28k dataset, return (X_all, Y_all, parts_all)."""
    print("Loading combined 28k dataset...")
    
    # df_adi labels + features
    adi = pd.read_csv(PROJ / "data" / "df_adi.csv")
    with h5py.File(str(PROJ / "data" / "prott5_attn_all_layers.h5"), 'r') as h:
        old_layers = [h[f'attn_layer_{l:02d}'][:].astype(np.float32) for l in [20,21,22,23]]
    prot5_old = np.concatenate(old_layers, axis=1)
    
    net_emb = np.load(str(PROJ / "data" / "space_network_embeddings.npy"))
    net_mask = np.load(str(PROJ / "data" / "space_network_mask.npy"))
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    old_aux = np.load(str(PROJ / "data" / "df_adi_aux_features.npy"))
    
    # New DeepLoc labels + features
    df_dl = pd.read_csv(PROJ / "data" / "deeploc_new_11562_labels.csv")
    dl_mapped = pd.DataFrame(0, index=range(len(df_dl)), columns=OUR_COLS)
    DL_TO_OURS = {'Membrane':'membrane','Cytoplasm':'cytoplasm','Nucleus':'nucleus',
        'Extracellular':'extracellular','Cell membrane':'cell_surface',
        'Mitochondrion':'mitochondrion','Endoplasmic reticulum':'endom'}
    for dl_col, our_col in DL_TO_OURS.items():
        if dl_col in df_dl.columns:
            dl_mapped[our_col] = df_dl[dl_col].values
    
    with h5py.File(str(PROJ / "data" / "deeploc_new_attn_pool.h5"), 'r') as h:
        new_layers = [h[f'attn_layer_{l:02d}'][:].astype(np.float32) for l in [20,21,22,23]]
    prot5_new = np.concatenate(new_layers, axis=1)
    
    # Combined labels
    adi_labels = adi[['acc'] + OUR_COLS + ['partition']].copy()
    df_dl['acc'] = df_dl['ACC'].astype(str)
    dl_temp = df_dl[['acc']].copy()
    dl_temp['partition'] = -1
    dl_temp[OUR_COLS] = dl_mapped.values
    df_combined = pd.concat([adi_labels, dl_temp], ignore_index=True)
    Y_all = df_combined[OUR_COLS].values.astype(np.int64)
    parts_all = df_combined['partition'].values  # -1 for new proteins
    
    # Combined features
    prot5_all = np.concatenate([prot5_old, prot5_new], axis=0)
    space_new = np.zeros((len(df_dl), net_filled.shape[1]), dtype=np.float32)
    space_all = np.concatenate([net_filled, space_new], axis=0)
    
    # Aux features for new proteins
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
    aux_all = np.concatenate([old_aux, new_aux], axis=0)
    
    X_all = np.concatenate([prot5_all, space_all, aux_all], axis=1).astype(np.float32)
    print(f"  X_all: {X_all.shape}  Y_all: {Y_all.shape}")
    print(f"  Partitions: {sorted(parts_all[pd.notna(parts_all) & (parts_all >= 0)].astype(int).tolist())}")
    return X_all, Y_all, parts_all


print("=" * 65)
print("  COMBINED 28k - 5-FOLD CV (PCA 100d)")
print("=" * 65)

t0 = time.time()
X_all, Y_all, parts_all = load_combined()

fold_results = []
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]

for fold in FOLDS:
    print(f"\n{'─'*55}")
    print(f"  FOLD {fold}")
    print(f"{'─'*55}")
    
    # Train = all partitions except fold, + new proteins (partition=-1)
    tr_mask = (parts_all != fold) | (parts_all == -1)
    te_mask = (parts_all == fold)
    
    X_tr_f, Y_tr_f = X_all[parts_all != fold], Y_all[parts_all != fold]
    X_te_f, Y_te_f = X_all[parts_all == fold], Y_all[parts_all == fold]
    
    print(f"  Train: {len(Y_tr_f):,}  Test: {len(Y_te_f):,}")
    
    # PCA on this fold
    pca = PCA(n_components=N_COMP)
    X_tr_pca = pca.fit_transform(X_tr_f).astype(np.float32)
    X_te_pca = pca.transform(X_te_f).astype(np.float32)
    vr = pca.explained_variance_ratio_.sum()
    
    # Baseline
    bf, bpc, _ = train_mlp(X_tr_pca, Y_tr_f, X_te_pca, Y_te_f)
    print(f"  Baseline F1: {bf:.4f} (var: {vr:.4f})")
    
    # Round 1 OOF + cleanlab
    print(f"  Round 1 OOF...")
    oof_r1 = gen_oof(X_tr_pca, Y_tr_f)
    keep_r1 = cleanlab_step(Y_tr_f, oof_r1, CL_CUTOFF)
    X_r1, Y_r1 = X_tr_pca[keep_r1], Y_tr_f[keep_r1]
    
    # Round 2 OOF + cleanlab
    print(f"  Round 2 OOF...")
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    
    # Final champion
    cf, cpc, _ = train_mlp(X_r2, Y_r2, X_te_pca, Y_te_f)
    print(f"  Champion F1: {cf:.4f}  (Gain: {cf-bf:+.4f})")
    
    fold_results.append({
        "fold": fold,
        "n_train": int(len(Y_tr_f)),
        "n_test": int(len(Y_te_f)),
        "n_after_cleanlab": int(len(Y_r2)),
        "baseline_f1": round(bf, 4),
        "champion_f1": round(cf, 4),
        "gain": round(cf - bf, 4),
        "baseline_per_class": [round(x,4) for x in bpc],
        "champion_per_class": [round(x,4) for x in cpc],
    })

# ── Aggregate ──
base_f1s = [r['baseline_f1'] for r in fold_results]
champ_f1s = [r['champion_f1'] for r in fold_results]
base_mean = float(np.mean(base_f1s))
base_std = float(np.std(base_f1s))
champ_mean = float(np.mean(champ_f1s))
champ_std = float(np.std(champ_f1s))

# Per-compartment
class_champs = {}
for j, c in enumerate(COMPARTMENTS):
    vals = [r['champion_per_class'][j] for r in fold_results]
    class_champs[c] = {
        "mean": round(float(np.mean(vals)), 4),
        "std": round(float(np.std(vals)), 4),
        "per_fold": [round(v, 4) for v in vals],
    }

print(f"\n{'='*55}")
print(f"  COMBINED 28k - 5-FOLD CV RESULT")
print(f"{'='*55}")
print(f"  {'':15s}  {'Mean':>8s}  {'Std':>8s}")
print(f"  {'─'*15}  {'─'*8}  {'─'*8}")
print(f"  {'Baseline':15s}  {base_mean:8.4f}  {base_std:8.4f}")
print(f"  {'Champion':15s}  {champ_mean:8.4f}  {champ_std:8.4f}")
print(f"  {'Gain':15s}  {champ_mean-base_mean:8.4f}")
print(f"\n  Per-fold breakdown:")
for r in fold_results:
    print(f"    Fold {r['fold']}: baseline={r['baseline_f1']:.4f}  champion={r['champion_f1']:.4f}  gain={r['gain']:+.4f}")

print(f"\n  Per-compartment champion F1:")
for c in COMPARTMENTS:
    cc = class_champs[c]
    print(f"    {c:>15s}: {cc['mean']:.4f} ± {cc['std']:.4f}  {cc['per_fold']}")

# Comparison with df_adi-only
print(f"\n{'='*55}")
print(f"  COMPARISON: df_adi-only (16,741) vs combined 28k")
print(f"{'='*55}")
print(f"  df_adi-only champion (5-fold CV, PCA 100d): 0.7414 ± 0.0147")
print(f"  Combined 28k champion (5-fold CV, PCA 100d): {champ_mean:.4f} ± {champ_std:.4f}")
print(f"  Gain from more data: {champ_mean - 0.7414:+.4f}")

# Save
report = {
    "method": "combined_28k_5fold_pca100",
    "n_total": int(len(X_all)),
    "df_adi_only_5fold_pca100": {"mean": 0.7414, "std": 0.0147},
    "combined_28k_5fold_pca100": {
        "mean": round(champ_mean, 4),
        "std": round(champ_std, 4),
        "baseline_mean": round(base_mean, 4),
        "baseline_std": round(base_std, 4),
    },
    "gain": round(champ_mean - 0.7414, 4),
    "per_class": class_champs,
    "per_fold": fold_results,
    "wall_time_min": round((time.time()-t0)/60, 1),
}

out_dir = PROJ / "output_cleaning_sweep"
out_dir.mkdir(exist_ok=True)
(out_dir / "champion_28k_5fold_result.json").write_text(json.dumps(report, indent=2))
print(f"\n  Saved: output_cleaning_sweep/champion_28k_5fold_result.json")
print(f"  Wall time: {time.time()-t0:.0f}s ({((time.time()-t0)/60):.1f}m)")
