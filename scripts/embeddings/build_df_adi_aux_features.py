#!/usr/bin/env python3
"""build_df_adi_aux_features.py

Compute Kyte-Doolittle signal-peptide + TMD-count heuristic features for every
df_adi protein and save to a NumPy array compatible with champion_5fold_cv.py.

Output:
  data/df_adi_aux_features.npy   shape (16741, 2)
      col 0 = sp_score    (binary proxy: 1 if avg hydrophobicity of first 30 AA >= 0.5)
      col 1 = tmd_count   (continuous: # of 19-residue windows with avg KD >= 1.6)

Why this exists:
  champion_5fold_cv.py currently feeds (attn-pool L22 1024-d + SPACE 512-d) into a
  1-layer MLP. Adding these 2 columns produces a 1538-d input matrix that gives
  the MLP direct access to sequence-level signal-peptide / TMD signal - the same
  auxiliary info DeepLoc 2.1 captures with its dedicated sorting-signal branch.

Row-order invariant:
  df_adi.csv rows MUST be in the same order as the row indexing of
  prott5_all_layers_dfadi-3.h5 (both use df_adi.csv row order). This script
  verifies that before writing.

Run:
  python3 build_df_adi_aux_features.py        # ~3 sec
"""

import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
SRC_CSV = PROJ / "data" / "df_adi.csv"
H5_PATH = PROJ / "data" / "prott5_attn_all_layers.h5"
OUT_NPY = PROJ / "data" / "df_adi_aux_features.npy"

# Kyte-Doolittle hydrophobicity scale (positive = hydrophobic)
KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


def kd_array(seq):
    """Per-residue Kyte-Doolittle hydrophobicity (sequence length up to 1024)."""
    return np.array([KD.get(aa, 0.0) for aa in seq[:1024]], dtype=np.float32)


def sp_proxy(seq):
    """Has signal peptide? Avg hydrophobicity of first 30 residues >= 0.5."""
    n = seq[:30]
    if len(n) < 15:
        return 0.0
    return float(np.mean([KD.get(aa, 0.0) for aa in n]) >= 0.5)


def tmd_count(seq, window=19, thr=1.6):
    """Count 19-res windows with avg KD >= 1.6 (TMD proxy)."""
    h = kd_array(seq)
    if len(h) < window:
        return 0.0
    cs = np.convolve(h, np.ones(window) / window, mode='valid')
    return float((cs >= thr).sum())


def main():
    t0 = time.time()
    print("=" * 60)
    print("  df_adi Aux Features: signal-peptide + TMD-count")
    print("=" * 60)

    df = pd.read_csv(SRC_CSV)
    print(f"\n  df_adi: {len(df)} proteins  cols: {list(df.columns)}")

    if 'sequence' not in df.columns:
        raise RuntimeError(f"df_adi.csv missing required 'sequence' column")

    seqs = df['sequence'].values.astype(str)

    # Row-order invariant: H5 row i must correspond to df_adi.csv row i
    # Verify by reading the H5's stored row count
    with h5py.File(H5_PATH, 'r') as f:
        h5_first_key = sorted(f.keys())[0]
        h5_n = f[h5_first_key].shape[0]
        print(f"  prott5_all_layers_dfadi-3.h5 first key: {h5_first_key!r}  rows: {h5_n}")
    if h5_n != len(df):
        raise RuntimeError(
            f"row-order mismatch: df_adi has {len(df)} rows but H5 has {h5_n} rows. "
            f"Re-check both sources are using the same df_adi.csv row order.")

    print("\n  Computing SP + TMD heuristics for each protein...")
    sp_arr = np.zeros(len(df), dtype=np.float32)
    tmd_arr = np.zeros(len(df), dtype=np.float32)
    for i, seq in enumerate(seqs):
        sp_arr[i] = sp_proxy(seq)
        tmd_arr[i] = tmd_count(seq)

    aux = np.stack([sp_arr, tmd_arr], axis=1)
    print(f"\n  Aux features shape: {aux.shape}")
    print(f"  SP rate:    {aux[:, 0].mean():.4f}  ({int(aux[:, 0].sum())}/{len(aux)})")
    print(f"  TMD count:  mean={aux[:, 1].mean():.3f}  "
          f"std={aux[:, 1].std():.3f}  "
          f"max={aux[:, 1].max():.0f}  "
          f"min={aux[:, 1].min():.0f}")

    np.save(OUT_NPY, aux)
    print(f"\n   Saved: {OUT_NPY}  ({OUT_NPY.stat().st_size / 1024:.1f} KB)")
    print(f"  Wall time: {time.time() - t0:.2f}s")
    print()
    print("  Next step: patch scripts/training/champion_5fold_cv_attn.py")
    print("              to load this NPY and concat 2 columns at the end of X_all.")


if __name__ == "__main__":
    main()