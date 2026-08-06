#!/usr/bin/env python3
"""extract_deeploc_space.py

Extract SPACE network embeddings for DeepLoc proteins using STRING aliases.
Maps UniProt accessions → STRING IDs → network embeddings.

Saves: deeploc_space_embeddings.npy + deeploc_space_mask.npy

Usage:
  python3 extract_deeploc_space.py 2>&1 | tee extract_deeploc_space.log
"""

import time, numpy as np, pandas as pd
from pathlib import Path
import h5py

PROJ = Path(__file__).parent.resolve()
DL_LABELS = PROJ / "data" / "deeploc_new_11562_labels.csv"
ALIASES = PROJ / "data" / "protein.aliases.v12.0.txt"
NETWORK_H5 = PROJ / "data" / "protein.network.embeddings.v12.0.h5"
OUT_EMB = PROJ / "data" / "deeploc_space_embeddings.npy"
OUT_MASK = PROJ / "data" / "deeploc_space_mask.npy"

t0 = time.time()
print("=" * 60)
print("  EXTRACT SPACE EMBEDDINGS FOR DEEPLOC PROTEINS")
print("=" * 60)

# ── Load DeepLoc accessions ──
dl = pd.read_csv(DL_LABELS)
n_dl = len(dl)
accs = dl["ACC"].values  # e.g. "Q9SE42"
base_accs = np.array([a.split("-")[0] for a in accs])  # strip isoform suffix

# Build lookup: base_acc → list of DeepLoc indices
acc_to_idxs = {}
for i, ba in enumerate(base_accs):
    acc_to_idxs.setdefault(ba, []).append(i)

unique_accs = set(base_accs)
print(f"DeepLoc proteins: {n_dl:,}")
print(f"Unique base accessions: {len(unique_accs):,}")

# ── Step 1: Scan aliases, map UniProt_AC → STRING_ID ──
print(f"\nScanning aliases ({ALIASES.name})...")
print(f"Looking for {len(unique_accs):,} unique accessions...")
t1 = time.time()

acc_to_string = {}  # base_acc → STRING_ID (first match only)
n_lines = 0; n_uni_lines = 0

with open(ALIASES, "rt") as f:
    for line in f:
        n_lines += 1
        parts = line.strip().split("\t")
        if len(parts) >= 3 and parts[2] == "UniProt_AC":
            n_uni_lines += 1
            uniprot = parts[1]
            if uniprot in acc_to_idxs and uniprot not in acc_to_string:
                acc_to_string[uniprot] = parts[0]  # STRING_ID
        if n_lines % 50_000_000 == 0:
            print(f"  {n_lines/1e6:.0f}M lines - {len(acc_to_string):,} matched...", flush=True)

elapsed = time.time() - t1
print(f"Scanned {n_lines:,} lines ({n_uni_lines:,} UniProt_AC) in {elapsed:.0f}s")
print(f"Matched {len(acc_to_string):,}/{len(unique_accs):,} DeepLoc accessions "
      f"({100*len(acc_to_string)/len(unique_accs):.1f}%)")

# ── Step 2: Extract embeddings from network H5 ──
print(f"\nExtracting embeddings from {NETWORK_H5.name}...")
t2 = time.time()

# Detect embedding dim
with h5py.File(NETWORK_H5, "r") as f:
    embed_dim = int(f["metadata"].attrs["embedding_dim"])

embeddings = np.zeros((n_dl, embed_dim), dtype=np.float32)
found_mask = np.zeros(n_dl, dtype=bool)

string_to_dl_idx = {}
for acc, sid in acc_to_string.items():
    string_to_dl_idx.setdefault(sid, []).extend(acc_to_idxs[acc])

n_found = 0
with h5py.File(NETWORK_H5, "r") as f:
    species_group = f["species"]
    n_species = len(species_group)
    for i, species_id in enumerate(species_group):
        if (i + 1) % 500 == 0:
            print(f"  Species {i+1}/{n_species} ({n_found} found)...", end="\r", flush=True)
        
        species_data = species_group[species_id]
        proteins = species_data["proteins"][:]
        emb = species_data["embeddings"][:]
        
        if isinstance(proteins[0], bytes):
            proteins_str = [p.decode("utf-8") for p in proteins]
        else:
            proteins_str = list(proteins)
        
        for j, sid in enumerate(proteins_str):
            if sid in string_to_dl_idx:
                for dl_idx in string_to_dl_idx[sid]:
                    if not found_mask[dl_idx]:
                        embeddings[dl_idx] = emb[j].astype(np.float32)
                        found_mask[dl_idx] = True
                        n_found += 1

elapsed2 = time.time() - t2
print(f"\nSpecies scanned: {n_species} in {elapsed2:.0f}s")
print(f"Found {n_found}/{n_dl} DeepLoc proteins with SPACE embeddings "
      f"({100*n_found/n_dl:.1f}%)")

# ── Save ──
np.save(OUT_EMB, embeddings)
np.save(OUT_MASK, found_mask)
print(f"\nSaved: {OUT_EMB}")
print(f"Saved: {OUT_MASK}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
