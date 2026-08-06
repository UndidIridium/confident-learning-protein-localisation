#!/usr/bin/env python3
"""champion_labelprop.py

Label Propagation champion on partition 4.
Uses SPACE embeddings to find PPI nearest neighbors, then propagates labels
through the SPACE network as additional features for the MLP.

Pipeline:
  1. Precompute kNN graph on SPACE embeddings (cosine similarity)
  2. For each training protein: neighbor_label_propensity = mean label of SPACE neighbors (train-only)
  3. For each test protein:   neighbor_label_propensity = mean label of SPACE neighbors (from train)
  4. Append these 7 propensity features to X: [ProtT5 + SPACE + aux + neighbor_labels] = 1545-d
  5. Champion MLP + 2-round cleanlab

No leakage: neighbor propensities are always computed from the OTHER set.
Within OOF folds, neighbors are restricted to the other training folds.

Usage:
  python3 champion_labelprop.py

Output: output_champion_labelprop/report.json + stdout
"""

import json, os, time, warnings
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from cleanlab.multilabel_classification.rank import get_label_quality_scores

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

PROJ = Path(__file__).parent.resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
PROT5_H5 = str(PROJ / "data" / "prott5_all_layers_dfadi-3.h5")
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
AUX_FEATS = PROJ / "data" / "df_adi_aux_features.npy"

LAYER = 22
LP_K = 50          # number of SPACE nearest neighbors for label propagation
LP_FEATURES = 7    # one per compartment

# MLP hyperparams (same as champion)
HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5; CL_CUTOFF = 0.40
HOLDOUT = 4

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]


def build_knn_graph(space_emb):
    """Build kNN graph on SPACE embeddings using cosine similarity.
    
    Returns: NearestNeighbors fit on non-zero SPACE rows.
    """
    # Use only proteins with SPACE edges for the graph
    mask = (np.abs(space_emb).max(axis=1) > 1e-6)
    X_graph = space_emb[mask]
    print(f"  SPACE graph: {X_graph.shape[0]}/{space_emb.shape[0]} proteins with edges")
    
    # Normalize embeddings for cosine similarity
    X_norm = X_graph / (np.linalg.norm(X_graph, axis=1, keepdims=True) + 1e-8)
    
    nn_model = NearestNeighbors(n_neighbors=LP_K + 1, metric="cosine", n_jobs=4)
    nn_model.fit(X_norm)
    
    return nn_model, mask


def get_neighbor_labels(space_emb, nn_model, graph_mask, Y_all, train_ix, target_ix):
    """Compute neighbor-label propensities for target proteins.
    
    For each protein in target_ix:
      - Find its LP_K nearest SPACE neighbors
      - Restrict to neighbors that are in train_ix (training set)
      - Compute mean label from those neighbors for each compartment
    
    Args:
        space_emb: (n, 512) SPACE embeddings
        nn_model: fitted NearestNeighbors on SPACE graph
        graph_mask: (n,) bool — which proteins have SPACE edges
        Y_all: (n, 7) full labels
        train_ix: indices of training proteins
        target_ix: indices of target proteins (could be train or test)
    
    Returns: (len(target_ix), 7) neighbor-label propensity scores
    """
    n_target = len(target_ix)
    prop = np.zeros((n_target, M), dtype=np.float32)
    
    # Get SPACE embeddings for target proteins
    X_target = space_emb[target_ix]
    
    # For targets without SPACE edges, return zeros
    has_space = graph_mask[target_ix]
    
    # Normalize for cosine similarity
    X_norm = X_target[has_space].copy()
    norms = np.linalg.norm(X_norm, axis=1, keepdims=True) + 1e-8
    X_norm = X_norm / norms
    
    if len(X_norm) == 0:
        return prop
    
    # Query kNN
    distances, indices = nn_model.kneighbors(X_norm)
    
    # Map graph indices back to full dataset indices
    graph_idx_to_full = np.where(graph_mask)[0]
    
    for i, t_idx in enumerate(np.where(has_space)[0]):
        full_t_idx = target_ix[t_idx]
        
        # Get neighbor indices in full dataset
        neighbor_full_idx = graph_idx_to_full[indices[i]]
        
        # Remove self if present
        neighbor_full_idx = neighbor_full_idx[neighbor_full_idx != full_t_idx]
        
        # Restrict to training set neighbors
        neighbor_in_train = neighbor_full_idx[np.isin(neighbor_full_idx, train_ix)]
        
        if len(neighbor_in_train) > 0:
            # Mean label across training neighbors
            prop[t_idx] = Y_all[neighbor_in_train].mean(axis=0).astype(np.float32)
    
    return prop


# =========================================================================
# MLP + Champion pipeline (same as champion_5fold_cv)
# =========================================================================

class MLP(nn.Module):
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, outdim))
    def forward(self, x):
        return self.net(x)


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
    for ep in range(1, MAX_EP + 1):
        model.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), BATCH_SIZE):
            ix = perm[s:s + BATCH_SIZE]
            criterion(model(Xt[ix]), Yt[ix]).backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            ep_ = torch.sigmoid(model(Xe)).numpy()
        ef = float(np.mean([f1_score(Ye[:, j].astype(int), (ep_[:, j] >= THR).astype(int), zero_division=0) for j in range(M)]))
        if ef > best_f1 + 1e-6:
            best_f1 = ef; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stall = 0
        else:
            stall += 1
            if stall >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    preds = (tp >= THR).astype(int)
    pc = [float(f1_score(Yte[:, j].astype(int), preds[:, j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc, tp


def gen_oof(X, Y, nn_model, graph_mask, space_emb, Y_all, all_ix, seed=42):
    """Generate OOF predictions. For each fold, recompute neighbor labels
    using only the other training folds (no leakage)."""
    n = len(X)
    oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_folds = 4; fs = n // n_folds
    
    for f in range(n_folds):
        vs = f * fs; ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
        
        # Map fold indices back to full dataset indices
        ti_full = all_ix[ti]
        vi_full = all_ix[vi]
        
        # Compute neighbor labels for this fold (no leakage)
        nei_tr = get_neighbor_labels(space_emb, nn_model, graph_mask, Y_all, ti_full, ti_full)
        nei_va = get_neighbor_labels(space_emb, nn_model, graph_mask, Y_all, ti_full, vi_full)
        
        X_tr_lp = np.concatenate([X[ti], nei_tr], axis=1)
        X_va_lp = np.concatenate([X[vi], nei_va], axis=1)
        
        _, _, tp = train_mlp(X_tr_lp, Y[ti], X_va_lp, Y[vi], seed=seed + f)
        oof[vi] = tp
        
        f1_f = np.mean([f1_score(Y[vi][:, j].astype(int), (tp[:, j] >= THR).astype(int), zero_division=0) for j in range(M)])
        print(f"        [Fold {f + 1}] F1={f1_f:.4f}", flush=True)
    
    return oof


def cleanlab_step(Y, oof, cutoff, label=""):
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(
        labels=labs, pred_probs=oof.astype(np.float64),
        method="self_confidence", adjust_pred_probs=True
    )
    keep = scores >= cutoff
    print(f"        Cleanlab {label}: {int(keep.sum())} kept, "
          f"{int((~keep).sum())} dropped ({100 * (~keep).sum() / len(Y):.1f}%)")
    return keep


def main():
    print("=" * 70)
    print("  LABEL PROPAGATION CHAMPION — PARTITION 4")
    print(f"  SPACE kNN (k={LP_K}) → neighbor label propensities → +{LP_FEATURES}d features")
    print("=" * 70)
    
    t0 = time.time()
    
    # Load data
    src = pd.read_csv(SRC_CSV)
    Y_all_full = src[LABEL_COLS].values.astype(np.float32)  # used for neighbor computation
    parts = src["partition"].to_numpy()
    train_mask = (parts != HOLDOUT); test_mask = (parts == HOLDOUT)
    n_tr = train_mask.sum(); n_te = test_mask.sum()
    
    # Load ProtT5 + SPACE + aux
    with h5py.File(PROT5_H5, "r") as f:
        prot5 = f[f"df_adi_layer_{LAYER:02d}"][:].astype(np.float32)
    net_emb = np.load(SPACE_EMB)
    net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy()
    net_filled[~net_mask] = 0.0
    aux_feats = np.load(AUX_FEATS)
    
    # Base features (without neighbor labels)
    X_base = np.concatenate([prot5, net_filled, aux_feats], axis=1).astype(np.float32)
    BASE_DIM = X_base.shape[1]
    print(f"\n  Data: {n_tr} train, {n_te} test")
    print(f"  Base features: {BASE_DIM}-d (ProtT5 {prot5.shape[1]} + SPACE {net_filled.shape[1]} + aux {aux_feats.shape[1]})")
    
    # ===================================================================
    # BUILD kNN GRAPH ON SPACE EMBEDDINGS
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  BUILDING SPACE kNN GRAPH (k={LP_K})")
    print(f"  {'='*60}")
    nn_model, graph_mask = build_knn_graph(net_emb)
    
    # ===================================================================
    # COMPUTE NEIGHBOR LABELS FOR TRAIN AND TEST
    # ===================================================================
    train_ix = np.where(train_mask)[0]
    test_ix = np.where(test_mask)[0]
    
    print(f"\n  Computing neighbor label propensities...")
    t_lp = time.time()
    
    # For train: neighbors from other train proteins only
    nei_tr = get_neighbor_labels(net_emb, nn_model, graph_mask, Y_all_full, train_ix, train_ix)
    # For test: neighbors from all training proteins
    nei_te = get_neighbor_labels(net_emb, nn_model, graph_mask, Y_all_full, train_ix, test_ix)
    
    dt_lp = time.time() - t_lp
    print(f"  Done ({dt_lp:.1f}s)")
    print(f"  Neighbor propensity stats:")
    for j, c in enumerate(COMPARTMENTS):
        print(f"    {c:>15s}: train mean={nei_tr[:, j].mean():.4f} test mean={nei_te[:, j].mean():.4f}")
    
    # Concatenate features
    X_tr = np.concatenate([X_base[train_ix], nei_tr], axis=1).astype(np.float32)
    X_te = np.concatenate([X_base[test_ix], nei_te], axis=1).astype(np.float32)
    Y_tr, Y_te = Y_all_full[train_ix].astype(np.int64), Y_all_full[test_ix].astype(np.int64)
    
    FEAT_DIM = X_tr.shape[1]
    print(f"  Final features: {FEAT_DIM}-d (base {BASE_DIM} + neighbor propensities {LP_FEATURES})")
    
    # ===================================================================
    # BASELINE (no cleanlab)
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  BASELINE MLP (with neighbor propensities)")
    print(f"  {'='*60}")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)
    print(f"  Baseline F1 = {base_f1:.4f}")
    
    # ===================================================================
    # ROUND 1: OOF + cleanlab (with per-fold neighbor recomputation)
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  ROUND 1 — OOF + CLEANLAB")
    print(f"  {'='*60}")
    all_ix = train_ix
    oof_r1 = gen_oof(X_base[train_ix], Y_tr, nn_model, graph_mask, net_emb, Y_all_full, all_ix)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF, "R1")
    
    # R1 kept indices and their corresponding neighbor labels
    r1_ix = train_ix[keep_r1]  # full dataset indices for R1-kept
    Y_r1 = Y_tr[keep_r1]
    
    # Recompute neighbor labels for R1-kept rows (neighbors from R1-kept only)
    nei_r1 = get_neighbor_labels(net_emb, nn_model, graph_mask, Y_all_full, r1_ix, r1_ix)
    X_r1 = np.concatenate([X_base[r1_ix], nei_r1], axis=1).astype(np.float32)
    print(f"  R1 kept: {len(Y_r1)}/{n_tr} ({100 * len(Y_r1) / n_tr:.1f}%)")
    
    # ===================================================================
    # ROUND 2: OOF + cleanlab
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  ROUND 2 — OOF + CLEANLAB")
    print(f"  {'='*60}")
    all_r1 = r1_ix
    oof_r2 = gen_oof(X_base[r1_ix], Y_r1, nn_model, graph_mask, net_emb, Y_all_full, all_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF, "R2")
    
    r2_ix = r1_ix[keep_r2]
    Y_r2 = Y_r1[keep_r2]
    
    # Recompute neighbor labels for R2-kept rows
    nei_r2 = get_neighbor_labels(net_emb, nn_model, graph_mask, Y_all_full, r2_ix, r2_ix)
    X_r2 = np.concatenate([X_base[r2_ix], nei_r2], axis=1).astype(np.float32)
    print(f"  R2 kept: {len(Y_r2)}/{len(Y_r1)} ({100 * len(Y_r2) / len(Y_r1):.1f}%)")
    
    # Recompute test neighbor labels from R2-kept training
    nei_te_r2 = get_neighbor_labels(net_emb, nn_model, graph_mask, Y_all_full, r2_ix, test_ix)
    X_te_r2 = np.concatenate([X_base[test_ix], nei_te_r2], axis=1).astype(np.float32)
    
    # ===================================================================
    # FINAL
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  FINAL MLP (with R2 neighbor propensities)")
    print(f"  {'='*60}")
    final_f1, final_pc, _ = train_mlp(X_r2, Y_r2, X_te_r2, Y_te)
    gain = final_f1 - base_f1
    
    dt = time.time() - t0
    
    # ===================================================================
    # REPORT
    # ===================================================================
    print("\n" + "=" * 65)
    print("  LABEL PROPAGATION — PARTITION 4 RESULTS")
    print("=" * 65)
    print(f"  {'Metric':>30s}  {'Score':>8s}")
    print(f"  {'-'*30}  {'-'*8}")
    print(f"  {'Baseline (with LP features)':>30s}  {base_f1:>8.4f}")
    print(f"  {'Champion (LP + cleanlab)':>30s}  {final_f1:>8.4f}")
    print(f"  {'Cleanlab gain':>30s}  {gain:>+8.4f}")
    print(f"  {'Wall time':>30s}  {dt:.0f}s ({dt/60:.1f}m)")
    
    print(f"\n  Per-compartment champion F1:")
    print(f"  {'Compartment':>15s}  {'LP':>8s}  {'MLP*':>8s}  {'Δ':>8s}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")
    
    mlp_ref = {
        "Membrane": 0.8344, "Cytoplasm": 0.7656, "Nucleus": 0.8260,
        "Extracell": 0.8886, "Cell_surf": 0.7570, "Mito": 0.8432, "Endom": 0.6926,
    }
    
    for j, c in enumerate(COMPARTMENTS):
        mlp_v = mlp_ref.get(c, 0.0)
        delta = final_pc[j] - mlp_v
        marker = " 🏆" if final_pc[j] > mlp_v + 0.005 else (" 📉" if mlp_v > final_pc[j] + 0.005 else "")
        print(f"  {c:>15s}  {final_pc[j]:>8.4f}  {mlp_v:>8.4f}  {delta:>+8.4f}{marker}")
    
    print(f"\n  {'Overall':>15s}  {final_f1:>8.4f}  {0.8011:>8.4f}  {final_f1 - 0.8011:>+8.4f}")
    
    print("\n" + "=" * 65)
    print("  HEAD-TO-HEAD (Partition 4)")
    print("=" * 65)
    print(f"  {'Model':>35s}  {'F1-macro':>9s}")
    print(f"  {'-'*35}  {'-'*9}")
    print(f"  {'🏆  Label Prop champion (this run)':>35s}  {final_f1:>9.4f}")
    print(f"  {'    Label Prop baseline':>35s}  {base_f1:>9.4f}")
    print(f"  {'🏆  Raw ProtT5 MLP champion':>35s}  {0.8011:>9.4f}")
    print(f"  {'    DeepLoc Accurate (ProtT5-XL)':>35s}  {0.7674:>9.4f}")
    print(f"  {'    DeepLoc Fast (ESM-1b)':>35s}  {0.7491:>9.4f}")
    
    # Save
    report = {
        "model": "Label Propagation + MLP",
        "features": f"ProtT5 L22 ({prot5.shape[1]}) + SPACE ({net_filled.shape[1]}) + aux ({aux_feats.shape[1]}) + neighbor_propensities ({LP_FEATURES}) = {FEAT_DIM}d",
        "label_prop": {"k": LP_K, "metric": "cosine", "source": "SPACE embeddings"},
        "holdout": HOLDOUT,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "n_after_r1": int(len(Y_r1)),
        "n_after_r2": int(len(Y_r2)),
        "baseline_f1": round(base_f1, 4),
        "champion_f1": round(final_f1, 4),
        "gain": round(gain, 4),
        "baseline_per_class": [round(x, 4) for x in base_pc],
        "champion_per_class": [round(x, 4) for x in final_pc],
        "wall_time_s": round(dt, 1),
        "comparison": {
            "raw_prott5_mlp_champion": 0.8011,
            "deeploc_accurate": 0.7674,
            "deeploc_fast": 0.7491,
            "vs_raw_prott5": round(final_f1 - 0.8011, 4),
            "vs_deeploc_accurate": round(final_f1 - 0.7674, 4),
            "vs_deeploc_fast": round(final_f1 - 0.7491, 4),
        },
    }
    
    out_dir = PROJ / "output_champion_labelprop"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
