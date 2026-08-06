#!/usr/bin/env python3
"""
build_prott5_attn_extract_colab.py
==================================
Builder for the Colab notebook that extracts attention-pooled ProtT5-XL
all-layer embeddings for df_adi proteins.

Output: prott5_attn_extract_colab.ipynb (saved to project root)

Reuses patterns from v80_prott5_layer_extract.py and esm2_3B_extract_colab.ipynb:
- output_hidden_states=True (no hooks)
- Tokenizer diagnostic gate
- Memmap checkpointing + JSON status for crash recovery
- Mean-pooled + new attention-pooled per-layer outputs
"""

import json
from pathlib import Path

OUT = Path("/Users/aditya/Desktop/project_JL/prott5_attn_extract_colab.ipynb")

NB = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "cells": []
}


def md(src):
    NB["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [src]})


def code(src):
    NB["cells"].append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": [src]
    })


# ============================================================================
# CELL 1: Markdown intro
# ============================================================================
md(r"""# Attention-Pooled ProtT5-XL — per-layer df_adi embedding extraction

Extracts **mean-pooled** and **attention-pooled** 1024-d embeddings from all 24 encoder layers of `Rostlab/prot_t5_xl_uniref50` for every protein in `df_adi.csv`.

## What is "attention pooling" vs "fine-tuning"?

| Method | Updates transformer weights? | # trainable params | Cost | Where used |
|---|---|---|---|---|
| **Mean pooling** (current) | No | 0 | Free | Your `champion_pipeline.py` |
| **Attention pooling** (this notebook) | **No** | ~1.6M (24 small heads) | ~45 min on T4 | DeepLoc 2.1 |
| **Fine-tuning** (most expensive) | **Yes** (last N layers) | ~3B | Multi-day on full PLM | Some benchmark winners |

**Key fact:** Attention pooling is **NOT fine-tuning**. The transformer weights are frozen. We only train a small attention head (~66k params per layer) that learns to weight per-residue embeddings.

The trained attention head is the same paradigm as DeepLoc 2.1 (Ødum et al. 2024): they keep ProtT5 frozen and learn attention weights per position per layer.

## Outputs

- `prott5_attn_all_layers.h5` containing:
  - `mean_layer_00` … `mean_layer_23`: mean-pooled 1024-d per layer (matches your existing pipeline)
  - `attn_layer_00` … `attn_layer_23`: **attention-pooled** 1024-d per layer (NEW)
  - `attn_avg`: averaged attention-pooled across layers (1024-d)
- Each dataset: `(16741, 1024)` float32, gzip compressed

## Wall time

- T4 (15 GB): ~50 min
- A100 (40 GB): ~25 min

## Required: a 3-letter amino-acid FASTA-style CSV with `sequence` column
""")

# ============================================================================
# CELL 2: Setup
# ============================================================================
code(r"""# ── 0. Install deps & check GPU ──────────────────────────────────────
import sys, subprocess, os, math, json, time, warnings
from pathlib import Path

import torch
print(f'Python: {sys.version.split()[0]}')
print(f'Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: no GPU — extraction will be extremely slow (~hours).')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=4.40', 'sentencepiece>=0.2', 'protobuf>=4',
    'h5py>=3.10', 'scikit-learn>=1.4', 'pandas>=2.0', 'numpy>=1.26'], check=True)
print('Deps installed.')
""")

# ============================================================================
# CELL 3: Mount Drive
# ============================================================================
code(r"""# ── 1. Mount Drive (outputs persist here) ─────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/project_JL')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
(DRIVE_ROOT / 'data').mkdir(parents=True, exist_ok=True)
(DRIVE_ROOT / 'checkpoints' / 'prott5_attn_extract').mkdir(parents=True, exist_ok=True)
print(f'Drive root: {DRIVE_ROOT}')
""")

# ============================================================================
# CELL 4: Upload df_adi.csv
# ============================================================================
code(r"""# ── 2. Upload df_adi.csv ───────────────────────────────────────────────
# Run this cell MANUALLY (not via "Run All") — files.upload() needs your click.
from google.colab import files
import shutil

CSV_PATH = Path('/content/df_adi.csv')
if CSV_PATH.exists():
    print(f'df_adi.csv already uploaded: {CSV_PATH.stat().st_size / 1e6:.1f} MB')
else:
    print('Click "Choose Files" and select df_adi.csv')
    uploaded = files.upload()
    for fname in uploaded:
        shutil.move(fname, str(CSV_PATH))
        break
    print(f'df_adi.csv ready: {CSV_PATH.stat().st_size / 1e6:.1f} MB')
""")

# ============================================================================
# CELL 5: Load and inspect df_adi.csv
# ============================================================================
code(r"""# ── 3. Load & inspect df_adi.csv ──────────────────────────────────────────
import pandas as pd
import numpy as np

df = pd.read_csv(str(CSV_PATH))
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Partitions: {sorted(df["partition"].unique())}')

# We need both sequences and labels (for attention-head training)
seqs = df['sequence'].values
LABEL_COLS = ['membrane', 'cytoplasm', 'nucleus', 'extracellular',
              'cell_surface', 'mitochondrion', 'endom']
labels = df[LABEL_COLS].values.astype(np.int64)
print(f'Labels shape: {labels.shape}')

seq_lens = [len(s) for s in seqs]
print(f'Seq lengths: min={min(seq_lens)}  max={max(seq_lens)}  '
      f'median={int(np.median(seq_lens))}')

# Split: train (partitions 0-3) for attention-head training, full corpus for extraction
train_mask = (df['partition'].values != 4)
print(f'Train: {train_mask.sum()}  Extraction (full): {len(df)}')
""")

# ============================================================================
# CELL 6: Load ProtT5 + tokenizer
# ============================================================================
code(r"""# ── 4. Load ProtT5 model + tokenizer ──────────────────────────────────
from transformers import T5EncoderModel, T5Tokenizer

MODEL_NAME = 'Rostlab/prot_t5_xl_uniref50'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FP16 = True
BATCH_SIZE = 4
MAX_SEQ_LEN = 1024
N_LAYERS = 24
EMBED_DIM = 1024
M = 7  # number of compartments

print(f'Loading tokenizer: {MODEL_NAME}...')
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)

print(f'Loading model (≈5.9 GB download, then ~30s load)...')
model = T5EncoderModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()
if FP16:
    model = model.half()
print(f'Loaded. Encoder blocks: {len(model.encoder.block)} (expected {N_LAYERS})')
""")

# ============================================================================
# CELL 7: Tokenizer diagnostic gate
# ============================================================================
code(r"""# ── 5. TOKENIZER DIAGNOSTIC GATE ⛔ ──────────────────────────────────
# CRITICAL: ProtT5 tokenizer expects SPACE-SEPARATED amino acids.
# 'MILKRTV' -> 'M I L K R T V' (without spaces, only the first char is recognized).

import re

def format_seq(seq):
    cleaned = re.sub(r'[UZOB]', 'X', seq.upper())
    return ' '.join(list(cleaned))

print('=== TOKENIZER DIAGNOSTIC ===')
test_seqs = [
    'MILKRTV',
    'AAAAAAA',
    'WWWWWWW',
    df['sequence'].iloc[0],
    df['sequence'].iloc[100],
    df['sequence'].iloc[-1],
]
test_seqs_fmt = [format_seq(s) for s in test_seqs]
encoded = tokenizer(test_seqs_fmt, return_tensors='pt', padding='longest',
                    truncation=True, max_length=MAX_SEQ_LEN,
                    add_special_tokens=True)
ids = encoded['input_ids']

unique_ids = set()
for i in range(len(test_seqs)):
    unique_ids.add(tuple(ids[i].tolist()))

print(f'Input IDs shape: {ids.shape}')
print(f'Unique token sequences: {len(unique_ids)} / {len(test_seqs)}')

if len(unique_ids) < 2:
    raise RuntimeError('TOKENIZER BUG: All test sequences produced SAME token IDs!')
elif ids.shape[1] <= 3:
    print('WARNING: sequences tokenized to <=3 tokens — check space-formatting.')
else:
    print(f'✅ PASS: {len(unique_ids)} unique sequences, {ids.shape[1]} tokens.')
""")

# ============================================================================
# CELL 8: Define attention pool head
# ============================================================================
code(r"""# ── 6. Define ATTENTION POOL head (per-layer, learnable) ───────────────
# The attention head is a small MLP (2 layers) that scores each token:
#   score_i = MLP(hidden_i) ∈ R
#   weights = softmax(scores * mask)
#   pooled = sum(weights * hidden)
#
# This is the same paradigm as DeepLoc 2.1. Transformer is FROZEN — only
# the attention head is trained.

import torch.nn as nn

class AttentionPool(nn.Module):
    # Per-layer attention pooling head. ~66K params per layer.
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, mask):
        # x: (B, L, D), mask: (B, L) bool
        scores = self.score(x).squeeze(-1)              # (B, L)
        scores = scores.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(scores, dim=1)           # (B, L)
        return (x * weights.unsqueeze(-1)).sum(dim=1)    # (B, D)


# 24 attention heads (one per layer), tiny — ~1.6M params total
attn_heads = nn.ModuleList([AttentionPool(EMBED_DIM) for _ in range(N_LAYERS)])
attn_heads = attn_heads.to(device)
attn_heads.train()

# Final linear projection: concat all 24 layers' pooled embeddings -> 7 compartments
combined_proj = nn.Linear(N_LAYERS * EMBED_DIM, M).to(device)

# Optimizer — only train attention heads + final linear
optimizer = torch.optim.Adam(
    list(attn_heads.parameters()) + list(combined_proj.parameters()),
    lr=1e-3,
)

# Per-class pos_weight (matches your champion_pipeline.py)
pos = labels[train_mask].sum(axis=0)
neg = len(pos) - pos
pw = np.clip(neg / np.maximum(pos, 1), 1.0, 20.0).astype(np.float32)
pos_w = torch.from_numpy(pw).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

print(f'Attention heads: {N_LAYERS} × {sum(p.numel() for p in attn_heads[0].parameters()):,} params = '
      f'{sum(p.numel() for p in attn_heads.parameters()):,} total')
print(f'Final linear projection: {N_LAYERS * EMBED_DIM} -> {M}')
print(f'pos_weight: {pw.tolist()}')
""")

# ============================================================================
# CELL 9: Train attention heads on training labels
# ============================================================================
code(r"""# ── 7. TRAIN attention heads on the 7-compartment task ────────────────
# Train for 1 epoch on a 1000-protein sample (~250 batches).
# This is enough to teach the heads to weight signal-peptide / TMD-adjacent
# tokens more. After training, the heads are frozen for the full extraction.

train_idx = np.where(train_mask)[0]
np.random.seed(42)
sample_idx = np.random.choice(train_idx, size=min(1000, len(train_idx)), replace=False)
print(f'Training attention heads on {len(sample_idx)} proteins for 1 epoch...')

attn_heads.train()
combined_proj.train()
t_train = time.time()

for batch_start in range(0, len(sample_idx), BATCH_SIZE):
    ix = sample_idx[batch_start:batch_start + BATCH_SIZE]
    batch_seqs = [seqs[i] for i in ix]
    batch_labels = labels[ix]

    seqs_fmt = [format_seq(s) for s in batch_seqs]
    enc = tokenizer(seqs_fmt, return_tensors='pt', padding='longest',
                    truncation=True, max_length=MAX_SEQ_LEN,
                    add_special_tokens=True)
    input_ids = enc['input_ids'].to(device)
    attn_mask = enc['attention_mask'].to(device)            # (B, L) int

    with torch.no_grad():
        if FP16:
            with torch.autocast(device_type='cuda'):
                outputs = model.encoder(input_ids=input_ids,
                                          attention_mask=attn_mask,
                                          output_hidden_states=True)
        else:
            outputs = model.encoder(input_ids=input_ids,
                                      attention_mask=attn_mask,
                                      output_hidden_states=True)

    all_hidden = outputs.hidden_states                        # 25 entries
    mask = attn_mask.bool()

    pooled_per_layer = []
    for layer_i in range(N_LAYERS):
        hidden = all_hidden[layer_i + 1].float()              # (B, L, D)
        z = attn_heads[layer_i](hidden, mask)                  # (B, D)
        pooled_per_layer.append(z)

    pooled_cat = torch.cat(pooled_per_layer, dim=-1)         # (B, 24*1024)
    logits = combined_proj(pooled_cat)                        # (B, 7)
    loss = criterion(logits, torch.from_numpy(batch_labels.astype(np.float32)).to(device))

    optimizer.zero_grad()
    loss.backward()
    # Gradient clip + NaN guard (reviewer fix)
    torch.nn.utils.clip_grad_norm_(
        list(attn_heads.parameters()) + list(combined_proj.parameters()),
        max_norm=1.0)
    if torch.isnan(loss):
        print(f'  NaN at batch {batch_start // BATCH_SIZE}, skipping')
        optimizer.zero_grad()
        continue
    optimizer.step()

    if (batch_start // BATCH_SIZE) % 20 == 0:
        print(f'  batch {batch_start // BATCH_SIZE:3d}  loss={loss.item():.4f}')

# Switch to eval mode for extraction
attn_heads.eval()
combined_proj.eval()  # not used during extraction, just for cleanliness
print(f'\\nAttention heads trained in {time.time() - t_train:.0f}s')
""")


# ============================================================================
# CELL 10: Extract per-layer mean + attention-pooled for full corpus
# ============================================================================
code(r"""# ── 8. EXTRACT per-layer mean-pooled + attention-pooled for FULL corpus ─
# Uses the trained attention heads (now FROZEN) AND a uniform mean for comparison.
# Output: (N, 24, 1024) for mean-pooled and (N, 24, 1024) for attention-pooled.

CHECKPOINT_DIR = DRIVE_ROOT / 'checkpoints' / 'prott5_attn_extract'
MMAP_PATH = CHECKPOINT_DIR / 'prott5_attn_memmap.npy'
H5_PATH = DRIVE_ROOT / 'data' / 'prott5_attn_all_layers.h5'
STATUS_PATH = CHECKPOINT_DIR / 'extraction_status.json'

n_total = len(seqs)
extraction_shape = (n_total, 2 * N_LAYERS, EMBED_DIM)  # 2 = mean + attn

# Create/load memmap (fp16 to save disk)
if not MMAP_PATH.exists():
    mmap = np.lib.format.open_memmap(str(MMAP_PATH), mode='w+',
                                      dtype=np.float16, shape=extraction_shape)
    print(f'Created memmap: {extraction_shape} ({mmap.nbytes / 1e9:.1f} GB)')
else:
    mmap = np.lib.format.open_memmap(str(MMAP_PATH), mode='r+',
                                      dtype=np.float16, shape=extraction_shape)
    print(f'Loaded memmap: {extraction_shape}')

# Resume logic
extraction_status = {}
if STATUS_PATH.exists():
    extraction_status = json.loads(STATUS_PATH.read_text())
    done = sum(1 for v in extraction_status.values() if v.get('done'))
    print(f'Resume: {done}/{n_total} rows done')
else:
    print('Fresh start')

skip_to = sum(1 for v in extraction_status.values() if v.get('done'))
batch_start_idx = skip_to
print(f'Starting from row {batch_start_idx} / {n_total}')


def mean_pool(hidden, attn_mask):
    mask_expanded = attn_mask.unsqueeze(-1).float().to(hidden.device)
    sum_h = (hidden * mask_expanded).sum(dim=1)
    count = mask_expanded.sum(dim=1).clamp(min=1)
    return (sum_h / count).cpu().numpy().astype(np.float16)


# Main extraction loop
total_batches = int(math.ceil((n_total - batch_start_idx) / BATCH_SIZE))
batch_times = []
t_start = time.time()

attn_heads.eval()  # frozen for extraction

try:
    for b_idx in range(total_batches):
        t_b = time.time()

        i0 = batch_start_idx + b_idx * BATCH_SIZE
        i1 = min(i0 + BATCH_SIZE, n_total)
        batch_seqs = list(seqs[i0:i1])
        batch_ids = list(range(i0, i1))

        seqs_fmt = [format_seq(s) for s in batch_seqs]
        enc = tokenizer(seqs_fmt, return_tensors='pt', padding='longest',
                        truncation=True, max_length=MAX_SEQ_LEN,
                        add_special_tokens=True)
        input_ids = enc['input_ids'].to(device)
        attn_mask = enc['attention_mask'].to(device)
        mask = attn_mask.bool()

        with torch.no_grad():
            if FP16:
                with torch.autocast(device_type='cuda'):
                    outputs = model.encoder(input_ids=input_ids,
                                              attention_mask=attn_mask,
                                              output_hidden_states=True)
            else:
                outputs = model.encoder(input_ids=input_ids,
                                          attention_mask=attn_mask,
                                          output_hidden_states=True)

        all_hidden = outputs.hidden_states   # 25 entries (embed + 24 blocks)

        for layer_i in range(N_LAYERS):
            hidden = all_hidden[layer_i + 1].float()    # (B, L, D)
            # Mean pool (slot 0 of last axis)
            mean_p = mean_pool(hidden, attn_mask)        # (B, D) float16
            # Attention pool (slot 1 of last axis)
            with torch.no_grad():
                attn_p = attn_heads[layer_i](hidden, mask).cpu().numpy().astype(np.float16)
            for local_idx, global_idx in enumerate(batch_ids):
                mmap[global_idx, layer_i, :] = mean_p[local_idx]
                mmap[global_idx, N_LAYERS + layer_i, :] = attn_p[local_idx]

        mmap.flush()
        for g_idx in batch_ids:
            extraction_status[str(g_idx)] = {'done': True}

        elapsed = time.time() - t_b
        batch_times.append(elapsed)
        avg_bt = sum(batch_times) / len(batch_times)
        remaining = (total_batches - b_idx - 1) * avg_bt

        if (b_idx + 1) % 20 == 0 or b_idx == 0:
            STATUS_PATH.write_text(json.dumps(extraction_status))
            print(f'  batch {b_idx + 1}/{total_batches}  rows {i0}-{i1 - 1}  '
                  f'{elapsed:.1f}s  ETA={remaining:.0f}s  ({avg_bt:.1f}s/batch)')

except Exception as e:
    STATUS_PATH.write_text(json.dumps(extraction_status))
    print(f'CRASHED at batch {b_idx + 1}: {type(e).__name__}: {e}')
    print('Checkpoint saved. Re-run this cell to resume.')
    raise e

STATUS_PATH.write_text(json.dumps(extraction_status))
total_time = time.time() - t_start
print(f'\\nExtraction complete: {n_total} proteins × {N_LAYERS} layers × mean+attn')
print(f'Total wall time: {total_time:.0f}s ({total_time / 60:.1f} min)')
""")

# ============================================================================
# CELL 11: Save to HDF5
# ============================================================================
code(r"""# ── 9. Write HDF5 (mean-pooled + attention-pooled per layer + averaged) ─
import h5py

if H5_PATH.exists():
    H5_PATH.unlink()
    print('Removed old HDF5')

mmap = np.lib.format.open_memmap(str(MMAP_PATH), mode='r',
                                  dtype=np.float16, shape=extraction_shape)

t_h5 = time.time()
with h5py.File(str(H5_PATH), 'w') as h5:
    # Mean-pooled per layer, under EXISTING key convention
    # 'df_adi_layer_XX' (drop-in compatible with champion_pipeline.py,
    # which loads f"df_adi_layer_{LAYER:02d}" from prott5_all_layers_dfadi-3.h5)
    for layer_i in range(N_LAYERS):
        arr = mmap[:, layer_i, :].astype(np.float32)
        h5.create_dataset(f'df_adi_layer_{layer_i:02d}', data=arr,
                           compression='gzip', compression_opts=4)
    # Attention-pooled per layer (NEW — separate namespace)
    for layer_i in range(N_LAYERS):
        arr = mmap[:, N_LAYERS + layer_i, :].astype(np.float32)
        h5.create_dataset(f'attn_layer_{layer_i:02d}', data=arr,
                           compression='gzip', compression_opts=4)
    # Cross-layer averages
    mean_all = np.mean(mmap[:, :N_LAYERS, :].astype(np.float32), axis=1)
    h5.create_dataset('mean_avg', data=mean_all,
                       compression='gzip', compression_opts=4)
    attn_all = np.mean(mmap[:, N_LAYERS:, :].astype(np.float32), axis=1)
    h5.create_dataset('attn_avg', data=attn_all,
                       compression='gzip', compression_opts=4)

print(f'\\nHDF5 written in {time.time() - t_h5:.0f}s')
print(f'Size: {H5_PATH.stat().st_size / 1e9:.2f} GB')

# ── Verification ──
print('\\n=== HDF5 VERIFICATION ===')
with h5py.File(str(H5_PATH), 'r') as h5:
    keys = sorted(h5.keys())
    print(f'{len(keys)} datasets: {keys[0]} .. {keys[-1]}')
    for k in [keys[0], f'attn_layer_{N_LAYERS // 2:02d}', keys[-1]]:
        arr = h5[k][:]
        valid = (np.max(np.abs(arr.max() - arr.min())) > 0.1
                  and not np.isnan(arr).any())
        print(f'  {k}: {arr.shape}  mean={arr.mean():.2f}  std={arr.std():.2f}  '
              f'{"OK" if valid else "BROKEN"}')
    n_total_chk = h5['df_adi_layer_00'].shape[0]
    print(f'\\nTotal rows: {n_total_chk}')
    print(f'Compression: gzip-4')
    print(f'Output: {H5_PATH}')
""")

# ============================================================================
# CELL 12: Download
# ============================================================================
code(r"""# ── 10. Download HDF5 ────────────────────────────────────────────────────
from google.colab import files
print(f'Downloading: {H5_PATH}')
print(f'Size: {H5_PATH.stat().st_size / 1e9:.2f} GB')
print('Save to: /Users/aditya/Downloads/prott5_attn_all_layers.h5')
files.download(str(H5_PATH))
""")

# ============================================================================
# CELL 13: Cleanup
# ============================================================================
code(r"""# ── 11. (Optional) Cleanup memmap ────────────────────────────────────────
if MMAP_PATH.exists():
    size_gb = MMAP_PATH.stat().st_size / 1e9
    confirm = input(f'Remove memmap ({size_gb:.1f} GB)? (y/n): ')
    if confirm.lower().startswith('y'):
        MMAP_PATH.unlink()
        if STATUS_PATH.exists():
            STATUS_PATH.unlink()
        print('Memmap and status removed.')
    else:
        print(f'Kept. You can delete manually: {MMAP_PATH}')
else:
    print('No memmap to delete.')
""")

# ============================================================================
# WRITE
# ============================================================================
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(NB, indent=1, ensure_ascii=False))
print(f'Wrote {OUT}')
print(f'  {len(NB["cells"])} cells')
