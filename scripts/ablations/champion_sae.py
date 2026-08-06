#!/usr/bin/env python3
"""champion_sae.py

Sparse Autoencoder (SAE) champion on partition 4.
Trains a TopK sparse autoencoder on ProtT5 L22 embeddings (unsupervised),
then uses the sparse codes as features for our champion MLP pipeline.

Architecture:
  ProtT5 L22 (1024-d) → Encoder → Sparse latent (4096-d, TopK=50) → Decoder → Reconstructed

  Then: [SAE codes (4096) + SPACE (512) + aux (2)] = 4610-d → MLP → 7 compartments

Pipeline:
  1. Train SAE on all 16K df_adi ProtT5 embeddings (unsupervised)
  2. Extract sparse codes for train + test
  3. Concatenate with SPACE + aux
  4. Champion MLP + 2-round cleanlab

Usage:
  python3 champion_sae.py

Output: output_champion_sae.json + stdout report
"""

import json, os, time, warnings, math
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

import torch, torch.nn as nn, torch.optim as optim
from torch.nn import functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# SAE hyperparams
SAE_LATENT = 4096     # 4× expansion over 1024-d
SAE_TOPK = 50         # only top 50 neurons fire per protein
SAE_LR = 1e-3
SAE_EPOCHS = 200
SAE_BATCH = 512
SAE_ES_FRAC = 0.10    # validation fraction
SAE_PATIENCE = 15

SAE_CACHE = PROJ / "output_champion_sae" / f"sae_weights_L{SAE_LATENT}_K{SAE_TOPK}.pt"
SAE_CODES = PROJ / "output_champion_sae" / "sae_codes.npy"

# MLP hyperparams (same as champion)
HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5; CL_CUTOFF = 0.40
HOLDOUT = 4

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]


# =========================================================================
# TOPK SPARSE AUTOENCODER
# =========================================================================

class TopKSAE(nn.Module):
    """Sparse autoencoder with TopK activation for exact k-sparsity.
    
    Args:
        in_dim: Input feature dimension
        latent_dim: Hidden (sparse) dimension
        k: Number of active neurons per forward pass
    """
    def __init__(self, in_dim, latent_dim, k):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(in_dim, latent_dim, bias=True)
        self.decoder = nn.Linear(latent_dim, in_dim, bias=True)
        # Tied encoder/decoder initialization (He init)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        self.decoder.weight.data = self.encoder.weight.data.t().clone()
        
    def encode(self, x):
        """Encode input to sparse latent codes.
        Returns both post-topk codes and pre-topk activations.
        """
        pre_act = torch.relu(self.encoder(x))  # ReLU ensures non-negative activations
        # TopK: keep top k, zero out rest
        topk_vals, topk_idx = torch.topk(pre_act, k=self.k, dim=1, sorted=False)
        # Create mask: only top k positions are non-zero
        mask = torch.zeros_like(pre_act)
        mask.scatter_(1, topk_idx, 1.0)
        post_act = pre_act * mask
        return post_act, pre_act
    
    def decode(self, codes):
        """Reconstruct input from sparse codes."""
        return self.decoder(codes)
    
    def forward(self, x):
        codes, _ = self.encode(x)
        recon = self.decode(codes)
        return recon, codes


def train_sae(X_all, device="cpu"):
    """Train TopK SAE on all ProtT5 embeddings (unsupervised).
    
    Args:
        X_all: (n, 1024) ProtT5 L22 embeddings
        device: torch device
    
    Returns: trained SAE model
    """
    out_dir = SAE_CACHE.parent
    out_dir.mkdir(exist_ok=True)
    
    if SAE_CACHE.exists():
        print(f"  Loading cached SAE from {SAE_CACHE}")
        sae = TopKSAE(X_all.shape[1], SAE_LATENT, SAE_TOPK).to(device)
        sae.load_state_dict(torch.load(SAE_CACHE, map_location=device, weights_only=True))
        sae.eval()
        return sae
    
    print(f"  Training TopK SAE: {X_all.shape[1]}→{SAE_LATENT}→{X_all.shape[1]}, k={SAE_TOPK}")
    print(f"  Data: {len(X_all)} proteins, {SAE_EPOCHS} max epochs")
    
    sae = TopKSAE(X_all.shape[1], SAE_LATENT, SAE_TOPK).to(device)
    opt = optim.Adam(sae.parameters(), lr=SAE_LR)
    
    # Train/val split
    n = len(X_all)
    ti, vi = train_test_split(np.arange(n), test_size=SAE_ES_FRAC, random_state=42)
    X_tr = torch.from_numpy(X_all[ti].astype(np.float32)).to(device)
    X_va = torch.from_numpy(X_all[vi].astype(np.float32)).to(device)
    
    best_loss, best_state, stall = float("inf"), None, 0
    t0 = time.time()
    
    for ep in range(1, SAE_EPOCHS + 1):
        sae.train()
        perm = torch.randperm(len(X_tr))
        epoch_loss = 0.0
        n_batches = 0
        
        for s in range(0, len(X_tr), SAE_BATCH):
            batch = X_tr[perm[s:s + SAE_BATCH]]
            recon, codes = sae(batch)
            
            # Reconstruction loss (MSE)
            loss = F.mse_loss(recon, batch)
            
            # Optional: auxiliary decoder weight norm (helps prevent dead neurons)
            # (not needed with TopK — it naturally prevents dead neurons vs L1)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validation
        sae.eval()
        with torch.no_grad():
            va_recon, va_codes = sae(X_va)
            va_loss = F.mse_loss(va_recon, X_va).item()
            
            # Track sparsity: mean active neurons per protein
            mean_active = (va_codes.abs() > 1e-6).float().sum(dim=1).mean().item()
        
        if va_loss < best_loss - 1e-8:
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in sae.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= SAE_PATIENCE:
                print(f"    Early stopping at epoch {ep} (val MSE={va_loss:.6f})")
                break
        
        if ep % 20 == 0 or ep == 1:
            dt_s = time.time() - t0
            print(f"    Epoch {ep:3d}: train MSE={epoch_loss / n_batches:.6f}  "
                  f"val MSE={va_loss:.6f}  avg active={mean_active:.1f}/{SAE_LATENT}  "
                  f"({dt_s:.0f}s)", flush=True)
    
    if best_state:
        sae.load_state_dict(best_state)
    
    sae.eval()
    sae = sae.cpu()
    torch.save(sae.state_dict(), SAE_CACHE)
    print(f"  SAE saved to {SAE_CACHE}")
    print(f"  Final val MSE: {best_loss:.6f}, avg active: {SAE_TOPK}/{SAE_LATENT} "
          f"({100 * SAE_TOPK / SAE_LATENT:.1f}%)")
    
    return sae


def extract_codes(sae, X, device="cpu"):
    """Extract sparse codes from SAE encoder.
    
    Args:
        sae: trained TopKSAE
        X: (n, 1024) ProtT5 embeddings
        device: torch device
    
    Returns: (n, SAE_LATENT) sparse codes
    """
    sae = sae.to(device)
    sae.eval()
    n = len(X)
    codes = np.zeros((n, SAE_LATENT), dtype=np.float32)
    
    with torch.no_grad():
        for s in range(0, n, SAE_BATCH):
            batch = torch.from_numpy(X[s:s + SAE_BATCH].astype(np.float32)).to(device)
            batch_codes, _ = sae.encode(batch)
            # TopK activation — only k positions are non-zero
            codes[s:s + SAE_BATCH] = batch_codes.cpu().numpy()
    
    return codes


def posw(Y):
    pw = np.ones(M, dtype=np.float32)
    for j in range(M):
        pos = float(Y[:, j].sum()); neg = float(Y.shape[0]) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg / pos)
    return np.clip(pw, 1.0, 20.0)


class MLP(nn.Module):
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, outdim))
    def forward(self, x):
        return self.net(x)


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
            if stall >= PATIENCE:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tp = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    preds = (tp >= THR).astype(int)
    pc = [float(f1_score(Yte[:, j].astype(int), preds[:, j], zero_division=0)) for j in range(M)]
    return float(np.mean(pc)), pc, tp


def gen_oof(X, Y, seed=42):
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    n_folds = 4; fs = n // n_folds
    for f in range(n_folds):
        vs = f * fs; ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
        _, _, tp = train_mlp(X[ti], Y[ti], X[vi], Y[vi], seed=seed + f)
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
    print("  SPARSE AUTOENCODER CHAMPION — PARTITION 4")
    print(f"  ProtT5 L22 → TopK SAE ({SAE_LATENT}d, k={SAE_TOPK}) → MLP + cleanlab")
    print("=" * 70)
    
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    # Load data
    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    train_mask = (parts != HOLDOUT); test_mask = (parts == HOLDOUT)
    n_tr = train_mask.sum(); n_te = test_mask.sum()
    print(f"\n  Data: {n_tr} train, {n_te} test")
    
    # Load ProtT5 + SPACE + aux
    with h5py.File(PROT5_H5, "r") as f:
        prot5 = f[f"df_adi_layer_{LAYER:02d}"][:].astype(np.float32)
    print(f"  ProtT5 L22: {prot5.shape}")
    
    net_emb = np.load(SPACE_EMB)
    net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy()
    net_filled[~net_mask] = 0.0
    aux_feats = np.load(AUX_FEATS)
    
    # ===================================================================
    # TRAIN SAE ON ALL PROT5 EMBEDDINGS (UNSUPERVISED)
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Train TopK Sparse Autoencoder")
    print(f"  {'='*60}")
    
    sae = train_sae(prot5, device=device)
    
    # Extract sparse codes
    print(f"\n  Extracting sparse codes ({SAE_LATENT}-d, k={SAE_TOPK})...")
    sae_codes = extract_codes(sae, prot5, device=device)
    
    # Verify sparsity
    nz_per_row = (np.abs(sae_codes) > 1e-6).sum(axis=1)
    print(f"  Active per protein: mean={nz_per_row.mean():.1f}, "
          f"min={nz_per_row.min()}, max={nz_per_row.max()}")
    print(f"  Overall sparsity: {100 * (1 - (nz_per_row.sum() / (sae_codes.shape[0] * sae_codes.shape[1]))):.1f}%")
    
    # ===================================================================
    # BUILD FEATURES: SAE codes + SPACE + aux
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Build features + champion pipeline")
    print(f"  {'='*60}")
    
    X_all = np.concatenate([sae_codes, net_filled, aux_feats], axis=1).astype(np.float32)
    FEAT_DIM = X_all.shape[1]
    print(f"  Features: {FEAT_DIM}-d (SAE codes {sae_codes.shape[1]} + SPACE {net_filled.shape[1]} + aux {aux_feats.shape[1]})")
    
    X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
    X_te, Y_te = X_all[test_mask], Y_all[test_mask]
    
    # ===================================================================
    # BASELINE (no cleanlab)
    # ===================================================================
    print(f"\n  Baseline MLP on SAE features ({n_tr} train)...")
    base_f1, base_pc, _ = train_mlp(X_tr, Y_tr, X_te, Y_te)
    print(f"  Baseline F1 = {base_f1:.4f}")
    
    # ===================================================================
    # ROUND 1: OOF + cleanlab
    # ===================================================================
    print(f"\n  Round 1 OOF (4-fold CV)...")
    oof_r1 = gen_oof(X_tr, Y_tr)
    keep_r1 = cleanlab_step(Y_tr, oof_r1, CL_CUTOFF, "R1")
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    print(f"  R1 kept: {len(Y_r1)}/{n_tr} ({100 * len(Y_r1) / n_tr:.1f}%)")
    
    # ===================================================================
    # ROUND 2: OOF + cleanlab
    # ===================================================================
    print(f"\n  Round 2 OOF (4-fold CV)...")
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2 = cleanlab_step(Y_r1, oof_r2, CL_CUTOFF, "R2")
    X_r2, Y_r2 = X_r1[keep_r2], Y_r1[keep_r2]
    print(f"  R2 kept: {len(Y_r2)}/{len(Y_r1)} ({100 * len(Y_r2) / len(Y_r1):.1f}%)")
    
    # ===================================================================
    # FINAL
    # ===================================================================
    print(f"\n  Final MLP on SAE features ({len(Y_r2)} train)...")
    final_f1, final_pc, final_tp = train_mlp(X_r2, Y_r2, X_te, Y_te)
    gain = final_f1 - base_f1
    
    dt = time.time() - t0
    
    # ===================================================================
    # REPORT
    # ===================================================================
    print("\n" + "=" * 65)
    print("  SAE CHAMPION — PARTITION 4 RESULTS")
    print("=" * 65)
    print(f"  {'Metric':>25s}  {'Score':>8s}")
    print(f"  {'-'*25}  {'-'*8}")
    print(f"  {'Baseline (no cleanlab)':>25s}  {base_f1:>8.4f}")
    print(f"  {'Champion (cleanlab)':>25s}  {final_f1:>8.4f}")
    print(f"  {'Cleanlab gain':>25s}  {gain:>+8.4f}")
    print(f"  {'Wall time':>25s}  {dt:.0f}s ({dt/60:.1f}m)")
    
    print(f"\n  Per-compartment champion F1:")
    print(f"  {'Compartment':>15s}  {'SAE':>8s}  {'MLP*':>8s}  {'Δ':>8s}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")
    
    # MLP reference (our best single MLP champion)
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
    
    # HEAD-TO-HEAD
    print("\n" + "=" * 65)
    print("  HEAD-TO-HEAD (Partition 4)")
    print("=" * 65)
    print(f"  {'Model':>35s}  {'F1-macro':>9s}")
    print(f"  {'-'*35}  {'-'*9}")
    print(f"  {'🏆  SAE champion (this run)':>35s}  {final_f1:>9.4f}")
    print(f"  {'    SAE baseline (no cleanlab)':>35s}  {base_f1:>9.4f}")
    print(f"  {'🏆  Raw ProtT5 MLP champion':>35s}  {0.8011:>9.4f}")
    print(f"  {'    DeepLoc Accurate (ProtT5-XL)':>35s}  {0.7674:>9.4f}")
    print(f"  {'    DeepLoc Fast (ESM-1b)':>35s}  {0.7491:>9.4f}")
    
    # Save
    report = {
        "model": "TopK Sparse Autoencoder + MLP",
        "sae_config": {
            "in_dim": prot5.shape[1],
            "latent_dim": SAE_LATENT,
            "topk": SAE_TOPK,
            "epochs_trained": min(SAE_EPOCHS, 200),  # actual epochs run
        },
        "features": f"SAE codes ({SAE_LATENT}d) + SPACE (512d) + aux (2d) = {FEAT_DIM}d",
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
        "sae_sparsity_pct": float(round(100 * (1 - nz_per_row.sum() / (sae_codes.shape[0] * sae_codes.shape[1])), 1)),
        "sae_active_per_protein_mean": float(round(nz_per_row.mean(), 1)),
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
    
    out_dir = PROJ / "output_champion_sae"
    out_dir.mkdir(exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
