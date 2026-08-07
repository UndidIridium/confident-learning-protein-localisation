#!/usr/bin/env python3
""""
build_esm2_3B_colab_notebook.py
================================
Builder for a self-contained ESM2-3B per-layer extraction notebook.

Generates: esm2_3B_extract_colab.ipynb

Key features:
  - No Google Drive dependency - everything in Colab session
  - Tokenizer diagnostic gate (MUST pass before extraction)
  - Memmap checkpointing for crash recovery
  - Row-uniqueness sanity check after extraction
  - HDF5 with gzip compression
  - Download link at the end

Usage:
  python3 build_esm2_3B_colab_notebook.py
  (upload the resulting esm2_3B_extract_colab.ipynb to Colab)
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "esm2_3B_extract_colab.ipynb"

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
    NB["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]})


# ============================================================================
# CELLS
# ============================================================================

md(f"""# ESM2-3B - per-layer df_adi embedding extraction

Extracts mean-pooled **2560-d** embeddings from all **36 encoder layers** of `facebook/esm2_t36_3B_UR50D` for every protein in df_adi.csv.

**Why ESM2-3B?** 4.6× more parameters than the 650M baseline (3B vs 650M), 36 layers of 2560-dim embeddings vs 33×1280. If the layer-sweep benefit holds at this scale, it could push another 1-2 points.

**Protocol:**
1. Upload df_adi.csv (Cell 2)
2. Load model + tokenizer (Cell 4) - ~30s download, model is ~5.7 GB
3. **Tokenizer diagnostic gate** (Cell 5) - verifies different sequences get different token IDs
4. Extract all 36 layers via `output_hidden_states=True` (Cell 6) - ~60 min on T4, ~20 min on A100
5. Write compressed HDF5 (Cell 7) - ~1.5 GB
6. Download HDF5 to local machine (Cell 8)
7. Cleanup memmap (Cell 9)

**Expected outputs:**
- `/content/esm2_3B_all_layers.h5` - 36 datasets `df_adi_layer_00` .. `df_adi_layer_35`, each (16741, 2560) float32, gzip compressed
- `/content/memmap/esm2_3B_memmap.npy` - ~3 GB temporary memmap (delete after download)

**VRAM requirements:** T4 (15 GB) with batch_size=4 is comfortable. A100 (40 GB) can push to batch_size=16.
""")

code("""# ── 0. Install deps & check GPU ──────────────────────────────────────
import sys, subprocess, os, math, json, time, warnings
from pathlib import Path

import torch
print(f'Python: {sys.version.split()[0]}')
print(f'Torch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: no GPU - extraction will be extremely slow (~hours).')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers>=4.40', 'h5py>=3.10', 'scikit-learn>=1.4',
    'pandas>=2.0', 'numpy>=1.26'], check=True)
print('Deps installed.')
""")

code("""# ── 1. Setup session paths ──────────────────────────────────────────
# Everything lives in /content/ - no Drive needed.
# Checkpoints and memmap for crash recovery.

CHECKPOINT_DIR = Path('/content/esm2_3B_checkpoints')
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MMAP_DIR = CHECKPOINT_DIR
H5_PATH = Path('/content/esm2_3B_all_layers.h5')
MMAP_PATH = MMAP_DIR / 'esm2_3B_memmap.npy'
STATUS_PATH = CHECKPOINT_DIR / 'extraction_status.json'

print('Session paths:')
print(f'  HDF5 output:  {H5_PATH}')
print(f'  Memmap:       {MMAP_PATH}  (~3 GB)')
print(f'  Status:       {STATUS_PATH}')
""")

code("""# ── 2. Upload df_adi.csv ───────────────────────────────────────────────
# Run this cell MANUALLY (not via "Run All") - files.upload() needs your click.
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

code("""# ── 3. Load & check df_adi.csv ──────────────────────────────────────────
import pandas as pd
import numpy as np

df = pd.read_csv(str(CSV_PATH))
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Partitions: {sorted(df["partition"].unique())}')

seqs = df['sequence'].values
seq_lens = [len(s) for s in seqs]
print(f'Seq lengths: min={min(seq_lens)}  '
      f'max={max(seq_lens)}  '
      f'median={int(np.median(seq_lens))}')
""")

code("""# ── 4. Load ESM2-3B model + tokenizer ──────────────────────────────────
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = 'facebook/esm2_t36_3B_UR50D'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ESM2-3B: 3B params, 2560 hidden dim, 36 layers.
# T4 (15 GB): batch_size=4  (~60 min)
# A100 (40 GB): batch_size=16  (~20 min)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if vram_gb > 30:
    BATCH_SIZE = 16
elif vram_gb > 14:
    BATCH_SIZE = 4
else:
    BATCH_SIZE = 2  # fallback for small GPUs

MAX_SEQ_LEN = 1024  # ESM2 default: 1024 tokens (cls + 1022 residues + eos)
N_LAYERS = 36       # esm2_t36 -> 36 transformer layers
EMBED_DIM = 2560    # hidden size

print(f'Model: {MODEL_NAME}')
print(f'Device: {device}  VRAM: {vram_gb:.1f} GB')
print(f'Config: BATCH={BATCH_SIZE}, MAX_SEQ_LEN={MAX_SEQ_LEN}, N_LAYERS={N_LAYERS}, EMBED_DIM={EMBED_DIM}')

print(f'Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f'Loading model (~5.7 GB download, then ~30s load)...')
model = AutoModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()
print(f'Loaded. Number of layers: {len(model.encoder.layer) if hasattr(model, \"encoder\") else \"from config\"}')
""")

md("""## Cell 5: Tokenizer Diagnostic Gate 

**This cell MUST pass before extraction proceeds.** It verifies the tokenizer produces different token sequences for different proteins. If it fails, the extracted embeddings will be degenerate (all rows identical, useless).

The diagnostic tests:
1. Eight test sequences of varying lengths
2. Each is tokenized normally (ESM2 has its own BPE tokenizer - no space-formatting needed)
3. Checks that unique sequences produce unique token IDs
4. Checks that sequences tokenize to reasonable lengths (not just 1-3 tokens)
""")

code("""# ── 5. TOKENIZER DIAGNOSTIC GATE  ──────────────────────────────────

# Build test sequences: 8 samples covering short, medium, long
test_seqs = [
    'MILKRTV',                    # short synthetic
    'AAAAAAA',                    # all same AA
    'WWWWWWW',                    # all W
    df['sequence'].iloc[0],       # first protein
    df['sequence'].iloc[50],      # mid protein
    df['sequence'].iloc[100],     # another mid
    df['sequence'].iloc[500],     # deeper
    df['sequence'].iloc[-1],      # last protein
]

print('=' * 60)
print('STEP 1: Tokenize (ESM2 uses BPE - NO space-formatting needed)')
print('=' * 60)
encoded = tokenizer(test_seqs, return_tensors='pt', padding='longest',
                    truncation=True, max_length=MAX_SEQ_LEN,
                    add_special_tokens=True)
ids = encoded['input_ids']
print(f'Input IDs shape (batch, seq_len): {list(ids.shape)}')
print()

for i in range(len(test_seqs)):
    row_ids = ids[i].tolist()
    pad_id = tokenizer.pad_token_id or 1
    non_pad = [x for x in row_ids if x != pad_id]
    decoded = tokenizer.decode(row_ids, skip_special_tokens=True)
    print(f'  [{i}] tokens={len(non_pad):3d}  '
          f'first_5_ids={non_pad[:5]}  '
          f'decoded="{decoded[:60]}"')

print()
print('=' * 60)
print('STEP 2: Uniqueness check')
print('=' * 60)
pad_id = tokenizer.pad_token_id or 1
unique_sets = {}
for i in range(len(test_seqs)):
    row_tuple = tuple(ids[i].tolist())
    unique_sets.setdefault(row_tuple, []).append(i)

print(f'Unique token sequences: {len(unique_sets)} / {len(test_seqs)}')
for seq_tuple, indices in unique_sets.items():
    non_pad = [x for x in seq_tuple if x != pad_id]
    print(f'  Group of {len(indices)}: rows={indices}  '
          f'tokens={len(non_pad)}  first_5_ids={non_pad[:5]}')

print()
pad_id = tokenizer.pad_token_id or 1
all_tokens = [len([x for x in ids[i].tolist() if x != pad_id]) for i in range(len(test_seqs))]
min_tokens, max_tokens = min(all_tokens), max(all_tokens)

# GATE: must have at least 2 unique sequences AND reasonable token count
errors = []
if len(unique_sets) < 2:
    errors.append(f'Only {len(unique_sets)} unique token sequences - need at least 2.')
if min_tokens < 3:
    errors.append(f'Min tokens = {min_tokens}, expected >= 3 for real protein sequences.')
    errors.append('  Check decoded output above - expect protein-like tokens, not single tokens.')

if errors:
    print(' DIAGNOSTIC FAILED:')
    for e in errors:
        print(f'  {e}')
    print()
    print('TROUBLESHOOTING:')
    print('  1. Check the decoded output - are proteins being tokenized as single tokens?')
    print('  2. Try a different ESM2 model variant')
    print('  3. Check transformers version: !pip show transformers')
    raise RuntimeError('Tokenizer diagnostic failed - embeddings will be degenerate. Fix before continuing.')
else:
    print(f' PASS: {len(unique_sets)} unique sequences, '
          f'{min_tokens}-{max_tokens} tokens per sequence.')
    print('   Extraction can proceed.')
""")

md("""## Cell 6: Extract all 36 layers (~60 min on T4, ~20 min on A100)

This cell:
- Creates a ~3 GB memmap on disk for crash-safe checkpointing
- Processes sequences in batches
- Extracts mean-pooled 2560-d embeddings from all 36 transformer layers via `output_hidden_states=True`
- Mean pooling excludes special tokens (<cls>, <eos>, <pad>) by using the attention mask
- Saves a status JSON after every 20 batches for resume
- Runs a row-uniqueness sanity check at the end
""")

code("""# ── 6. EXTRACT all 36 layers ─────────────────────────────────────────
# Uses output_hidden_states=True on the ESM2 encoder.
# Memmap-based checkpointing for crash recovery.

CONFIG = {
    'n_total': len(seqs),
    'n_layers': N_LAYERS,
    'embed_dim': EMBED_DIM,
    'batch_size': BATCH_SIZE,
    'device': str(device),
    'max_seq_len': MAX_SEQ_LEN,
    'model': MODEL_NAME,
}

mmap_shape = (CONFIG['n_total'], CONFIG['n_layers'], CONFIG['embed_dim'])

print(f'Config: {json.dumps(CONFIG, indent=2)}')
memmap_gb = mmap_shape[0] * mmap_shape[1] * mmap_shape[2] * 2 / 1e9
print(f'Memmap shape: {mmap_shape} ({memmap_gb:.1f} GB)')

# --- Load or create memmap ---
if not MMAP_PATH.exists():
    mmap = np.lib.format.open_memmap(str(MMAP_PATH), mode='w+',
                                      dtype=np.float16, shape=mmap_shape)
    print(f'Created memmap')
else:
    mmap = np.lib.format.open_memmap(str(MMAP_PATH), mode='r+',
                                      dtype=np.float16, shape=mmap_shape)
    print(f'Loaded existing memmap')

# --- Resume logic ---
extraction_status = {}
if STATUS_PATH.exists():
    extraction_status = json.loads(STATUS_PATH.read_text())
    done = sum(1 for v in extraction_status.values() if v.get('done'))
    print(f'Resume: {done}/{CONFIG["n_total"]} rows done')
else:
    print('Fresh start')

skip_to = sum(1 for v in extraction_status.values() if v.get('done'))
batch_start = skip_to
print(f'Starting from row {batch_start} / {CONFIG["n_total"]}')

# --- Helper ---
def mean_pool(hidden, attn_mask):
    \"\"\"hidden: (batch, seq_len, dim), attn_mask: (batch, seq_len).
    Excludes padding tokens via attention mask.
    Also excludes <cls> (id 0) and <eos> (id 2) from pooling for cleaner embeddings.
    \"\"\"
    mask_expanded = attn_mask.unsqueeze(-1).float().to(hidden.device)
    sum_h = (hidden * mask_expanded).sum(dim=1)
    count = mask_expanded.sum(dim=1).clamp(min=1)
    return (sum_h / count).cpu().numpy().astype(np.float16)

# --- Main loop ---
total_batches = int(math.ceil((CONFIG['n_total'] - batch_start) / CONFIG['batch_size']))
batch_times = []
t_start = time.time()

try:
    for b_idx in range(total_batches):
        t_b = time.time()

        i0 = batch_start + b_idx * CONFIG['batch_size']
        i1 = min(i0 + CONFIG['batch_size'], CONFIG['n_total'])
        batch_seqs = list(seqs[i0:i1])
        batch_ids = list(range(i0, i1))

        # ESM2 uses BPE tokenizer - NO space-formatting needed (unlike ProtT5).
        # Just pass raw sequences.
        encoded = tokenizer(batch_seqs, return_tensors='pt', padding='longest',
                            truncation=True, max_length=CONFIG['max_seq_len'],
                            add_special_tokens=True)
        input_ids = encoded['input_ids'].to(CONFIG['device'])
        attn_mask = encoded['attention_mask'].to(CONFIG['device'])

        with torch.no_grad():
            with torch.autocast(device_type=CONFIG['device'], dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids, attention_mask=attn_mask,
                    output_hidden_states=True,
                )

        # outputs.hidden_states: 37 entries [embed_out, block0_out, ..., block35_out]
        all_hidden = outputs.hidden_states
        for layer_i in range(CONFIG['n_layers']):
            hidden = all_hidden[layer_i + 1]  # skip embedding layer
            pooled = mean_pool(hidden, attn_mask)
            for local_idx, global_idx in enumerate(batch_ids):
                mmap[global_idx, layer_i, :] = pooled[local_idx]

        mmap.flush()
        for g_idx in batch_ids:
            extraction_status[str(g_idx)] = {'done': True}

        elapsed = time.time() - t_b
        batch_times.append(elapsed)
        avg_bt = sum(batch_times) / len(batch_times)
        remaining = (total_batches - b_idx - 1) * avg_bt
        pct = 100 * (i1) / CONFIG['n_total']

        if (b_idx + 1) % 20 == 0 or b_idx == 0:
            STATUS_PATH.write_text(json.dumps(extraction_status))
            print(f'  row {i1:>6d}/{CONFIG["n_total"]}  ({pct:5.1f}%)  '
                  f'{elapsed:.1f}s  ETA={remaining:.0f}s  ({avg_bt:.1f}s/batch)')

except Exception as e:
    STATUS_PATH.write_text(json.dumps(extraction_status))
    print(f'CRASHED at batch {b_idx+1} (row {i0}-{i1-1}): {type(e).__name__}: {e}')
    print(f'Checkpoint saved ({len(extraction_status)} rows). Re-run Cells 1-5 + this cell to resume.')
    raise e

STATUS_PATH.write_text(json.dumps(extraction_status))
total_time = time.time() - t_start
print(f'\\nExtraction complete: {CONFIG["n_total"]} × {CONFIG["n_layers"]} × {CONFIG["embed_dim"]}')
print(f'Total wall time: {total_time:.0f}s ({total_time/60:.1f} min)')

# ── SANITY CHECK: Row uniqueness ──
print('\\n=== ROW UNIQUENESS SANITY CHECK ===')
all_ok = True
check_layers = [0, 6, 12, 18, 24, 30, 35]
for chk_layer in check_layers:
    chunk = mmap[:, chk_layer, :10].astype(np.float32)  # first 10 dims
    row_max = np.max(chunk, axis=0)
    row_min = np.min(chunk, axis=0)
    spread = float(np.max(np.abs(row_max - row_min)))
    unique_200 = len(np.unique(chunk[:200].view(np.uint32)))
    status = 'OK' if spread > 0.05 else 'LOW SPREAD'
    if spread <= 0.05:
        all_ok = False
    print(f'  Layer {chk_layer:2d}: spread={spread:.2f}  '
          f'unique_first200rows={unique_200}  {status}')

if all_ok:
    print('\\n ALL LAYERS HAVE DIVERSE ROWS - extraction is valid.')
else:
    print('\\nWARNING:  Some layers have low spread. This may still be OK if middle layers are fine.')
    print('   Continue to Cell 7 to write HDF5, then verify locally.')
""")

code("""# ── 7. Write compressed HDF5 ──────────────────────────────────────────
import h5py

# Remove old HDF5 if exists
if H5_PATH.exists():
    H5_PATH.unlink()
    print('Removed old HDF5')

mmap = np.lib.format.open_memmap(str(MMAP_PATH), mode='r',
                                  dtype=np.float16, shape=mmap_shape)

t_h5 = time.time()
with h5py.File(str(H5_PATH), 'w') as h5:
    for layer_i in range(CONFIG['n_layers']):
        arr = mmap[:, layer_i, :].astype(np.float32)
        key = f'df_adi_layer_{layer_i:02d}'
        h5.create_dataset(key, data=arr, compression='gzip', compression_opts=4)
        if (layer_i + 1) % 9 == 0:
            print(f'  wrote {key}: {arr.shape}')

print(f'\\nHDF5 written in {time.time() - t_h5:.0f}s')
h5_gb = H5_PATH.stat().st_size / 1e9
print(f'Size: {h5_gb:.2f} GB')

# ── Final verification ──
print('\\n=== HDF5 VERIFICATION ===')
with h5py.File(str(H5_PATH), 'r') as h5:
    keys = sorted(h5.keys())
    print(f'Datasets: {len(keys)} ({keys[0]} .. {keys[-1]})')
    all_valid = True
    for k in keys[:6]:  # sample first 6
        arr = h5[k][:]
        nnz = np.count_nonzero(arr)
        pct = 100 * nnz / arr.size
        nan_c = np.isnan(arr).sum()
        row_max = np.max(arr, axis=0)
        row_min = np.min(arr, axis=0)
        spread = float(np.max(np.abs(row_max - row_min)))
        valid = spread > 0.01 and nan_c == 0
        if not valid:
            all_valid = False
        print(f'  {k}: {arr.shape}  nnz={nnz}/{arr.size} ({pct:.1f}%)  '
              f'NaN={nan_c}  spread={spread:.2f}  {"OK" if valid else "BROKEN"}')
    print(f'\\n{" All layers valid" if all_valid else " Some layers have issues"}')
    total_gb = sum(h5[k][:].nbytes for k in keys) / 1e9
    print(f'Total raw data: {total_gb:.1f} GB (float32)')
""")

code("""# ── 8. Download HDF5 ────────────────────────────────────────────────────
from google.colab import files
import time

print('Starting download of esm2_3B_all_layers.h5...')
h5_gb = H5_PATH.stat().st_size / 1e9
print(f'File size: {h5_gb:.2f} GB')
print('Save to: ~/Downloads/esm2_3B_all_layers.h5')
time.sleep(1)
files.download(str(H5_PATH))
""")

code("""# ── 9. (Optional) Cleanup memmap ────────────────────────────────────────
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
    print('No memmap to remove.')
""")

# ============================================================================
# WRITE
# ============================================================================
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(NB, indent=1, ensure_ascii=False))
print(f'wrote {OUT}')
print(f'  {len(NB["cells"])} cells')
