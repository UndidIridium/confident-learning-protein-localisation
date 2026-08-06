#!/usr/bin/env python3
"""compare_deeploc_p4.py

Uses DeepLoc 2.1's OWN partition column (0-4) for train/test split.
Train on DeepLoc folds 0-3 (~22,642 proteins), test on fold 4 (~5,660).
Includes both df_adi and DeepLoc embeddings, mapped to our 7 compartments.

P4 = DeepLoc partition 4, independently held out from DeepLoc training.
This is a fresh test set - none of these proteins were trained on.
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
    print(f"    Cleanlab: {int(keep.sum())} kept, {int((~keep).sum())} dropped ({100*(~keep).sum()/len(Y):.1f}%)")
    return keep

print("="*62)
print("  DEEPLOC P4 COMPARISON - PCA 100d")
print("  Train: DeepLoc folds 0-3  |  Test: DeepLoc fold 4")
print("="*62)

# ── 1. Load the full DeepLoc dataset (28,303 proteins with Partition column) ──
dl_all = pd.read_csv(PROJ / 'data' / 'deeploc_all_28303.csv')
print(f"\nDeepLoc total: {len(dl_all)}")
print(f"Partition distribution: {dl_all['Partition'].value_counts().sort_index().to_dict()}")

# Map DeepLoc 11 cols to our 7
DL_TO_OURS = {'Membrane':'membrane','Cytoplasm':'cytoplasm','Nucleus':'nucleus',
    'Extracellular':'extracellular','Cell membrane':'cell_surface',
    'Mitochondrion':'mitochondrion','Endoplasmic reticulum':'endom'}
dl_mapped = pd.DataFrame(0, index=range(len(dl_all)), columns=OUR_COLS)
for dl_col, our_col in DL_TO_OURS.items():
    if dl_col in dl_all.columns:
        dl_mapped[our_col] = dl_all[dl_col].values

# Identify which rows are in DeepLoc partition 4
dl_partition = dl_all['Partition'].values  # 0-4
dl_accs = dl_all['ACC'].astype(str).values
Y_map = dl_mapped.values.astype(np.int64)

# ── 2. Load features for all 28,303 proteins ──
# df_adi attn (16,741)
with h5py.File(str(PROJ / "data" / "prott5_attn_all_layers.h5"), 'r') as h:
    old_layers = [h[f'attn_layer_{l:02d}'][:].astype(np.float32) for l in [20,21,22,23]]
prot5_old = np.concatenate(old_layers, axis=1)

# df_adi SPACE
net_emb = np.load(str(PROJ / "data" / "space_network_embeddings.npy"))
net_mask = np.load(str(PROJ / "data" / "space_network_mask.npy"))
net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0

# df_adi aux
old_aux = np.load(str(PROJ / "data" / "df_adi_aux_features.npy"))

# df_adi accessions (for aligning labels)
adi = pd.read_csv(PROJ / "data" / "df_adi.csv")
adi_accs = set(adi['acc'].astype(str).values)


# Load new DeepLoc attn (11,562 new proteins)
with h5py.File(str(PROJ / "data" / "deeploc_new_attn_pool.h5"), 'r') as h:
    new_layers = [h[f'attn_layer_{l:02d}'][:].astype(np.float32) for l in [20,21,22,23]]
prot5_new = np.concatenate(new_layers, axis=1)

# Aux for new proteins (compute from DeepLoc sequences)
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

# We need to align: for df_adi proteins (the first 16,741), use existing features.
# For new proteins (the remaining 11,562), compute new features.

# Build aligned feature matrix for ALL 28,303 DeepLoc proteins
# We need to map dl_accs -> feature index
# The df_adi H5 order matches df_adi CSV order
# The new H5 order matches the new DeepLoc CSV order

# Strategy: build all features by indexing dl_all in order
# For each DeepLoc accession, determine if it's in df_adi or new

# df_adi: look up by accession to get index in adi
adi_idx_map = {a: i for i, a in enumerate(adi['acc'].astype(str).values)}

# New DeepLoc: the new H5 is ordered by deeploc_new_11562_labels.csv
dl_new = pd.read_csv(PROJ / "data" / "deeploc_new_11562_labels.csv")
new_idx_map = {a: i for i, a in enumerate(dl_new['ACC'].astype(str).values)}

print(f"\nBuilding aligned features for all {len(dl_all)} proteins...")
X_full = np.zeros((len(dl_all), 4610), dtype=np.float32)  # 4096 prot + 512 space + 2 aux

for i, acc in enumerate(dl_accs):
    if i % 5000 == 0:
        print(f"  Aligning row {i}/{len(dl_all)}...")
    
    if acc in adi_idx_map:
        # df_adi protein: use existing features
        j = adi_idx_map[acc]
        X_full[i, :4096] = prot5_old[j]
        X_full[i, 4096:4608] = net_filled[j]
        X_full[i, 4608:] = old_aux[j]
    elif acc in new_idx_map:
        # New DeepLoc protein: use new attn + zero SPACE
        j = new_idx_map[acc]
        X_full[i, :4096] = prot5_new[j]
        X_full[i, 4096:4608] = 0.0  # no SPACE
        # Compute aux on the fly
        seq = dl_all.iloc[i]['Sequence']
        X_full[i, 4608] = sp_proxy(seq)
        X_full[i, 4609] = tmd_count(seq)
    else:
        print(f"  WARNING: {acc} not found in either dataset!")

print(f"  X_full: {X_full.shape}")

# ── 3. Split by DeepLoc partition ──
tr_mask = dl_partition != 4
te_mask = dl_partition == 4
X_tr, Y_tr = X_full[tr_mask], Y_map[tr_mask]
X_te, Y_te = X_full[te_mask], Y_map[te_mask]
print(f"\nTrain: {len(Y_tr):,}  Test: {len(Y_te):,}")

# Check: how many df_adi proteins end up in the test set?
test_accs = dl_accs[te_mask]
n_adi_in_test = sum(1 for a in test_accs if a in adi_accs)
print(f"  Test set: {len(test_accs)} proteins ({n_adi_in_test} from df_adi, {len(test_accs)-n_adi_in_test} new)")

# PCA 100d
pca = PCA(n_components=N_COMP)
X_tr_pca = pca.fit_transform(X_tr).astype(np.float32)
X_te_pca = pca.transform(X_te).astype(np.float32)
print(f"  PCA {N_COMP}d (var: {pca.explained_variance_ratio_.sum():.4f})")

# ── 4. Baseline ──
print(f"\n── Baseline ──")
bf, bpc, _ = train_mlp(X_tr_pca, Y_tr, X_te_pca, Y_te)
print(f"  Baseline F1: {bf:.4f}")
for j, c in enumerate(['Membrane','Cytoplasm','Nucleus','Extracell','Cell_surf','Mito','Endom']):
    print(f"    {c:>15s}: {bpc[j]:.4f}")

# ── 5. Champion (cleanlab 2-pass) ──
print(f"\n── Champion (cleanlab 2-pass) ──")
oof_r1 = gen_oof(X_tr_pca, Y_tr)
keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF)
print(f"  R1: {int(keep_r1.sum())}/{len(Y_tr)} kept")
X_r1, Y_r1 = X_tr_pca[keep_r1], Y_tr[keep_r1]

oof_r2 = gen_oof(X_r1, Y_r1)
keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF)
print(f"  R2: {int(keep_r2.sum())}/{len(Y_r1)} kept")
X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]

cf, cpc, _ = train_mlp(X_r2, Y_r2, X_te_pca, Y_te)
print(f"\n  Champion F1: {cf:.4f}")
print(f"  Gain:        {cf - bf:+.4f}")
print()
for j, c in enumerate(['Membrane','Cytoplasm','Nucleus','Extracell','Cell_surf','Mito','Endom']):
    print(f"    {c:>15s}: {cpc[j]:.4f}")

# ── Summary ──
print(f"\n{'='*55}")
print(f"  DEEPLOC PARTITION 4 - SUMMARY (PCA {N_COMP}d)")
print(f"{'='*55}")
print(f"  Train: {len(Y_tr):,} (DeepLoc folds 0-3)")
print(f"  Test:  {len(Y_te):,} (DeepLoc fold 4, {n_adi_in_test} from df_adi)")
print(f"  Baseline F1:  {bf:.4f}")
print(f"  Champion F1:  {cf:.4f}")
print(f"  Gain:         {cf - bf:+.4f}")
print(f"  Cleanlab:     {len(Y_tr):,} → {len(Y_r2):,} ({100*len(Y_r2)/len(Y_tr):.0f}% kept)")

result = {
    "method": "deeploc_p4_comparison_pca100",
    "n_train": int(len(Y_tr)),
    "n_test": int(len(Y_te)),
    "n_adi_in_test": int(n_adi_in_test),
    "n_after_cleanlab": int(len(Y_r2)),
    "baseline_f1": round(bf, 4),
    "champion_f1": round(cf, 4),
    "gain": round(cf - bf, 4),
    "baseline_per_class": [round(x,4) for x in bpc],
    "champion_per_class": [round(x,4) for x in cpc],
}
out_dir = PROJ / "output_cleaning_sweep"
out_dir.mkdir(exist_ok=True)
(out_dir / "compare_deeploc_p4.json").write_text(json.dumps(result, indent=2))
print(f"\n  Saved: output_cleaning_sweep/compare_deeploc_p4.json")
