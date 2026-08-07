#!/usr/bin/env python3
"""build_df_adi_v2_extract_colab.py

Generates a Colab notebook that mirrors Kaggle v2's design for the **df_adi**
main benchmark:

  - ProtT5-XL backbone (frozen)
  - 24 attention-pool heads trained multi-task with localization
  - 2 auxiliary heads (signal-peptide binary + TMD-count regression)
  - Trained jointly with multi-task loss:
        L_total = L_localization + 0.3 * L_sp + 0.3 * L_tmd

Output H5:
  ~/Downloads/df_adi_attn_pool_v2.h5
    - attn_layer_22 : (16741, 1024) float32  - attention-pooled L22
    - sp_score      : (16741,)     float32  - model-predicted P(signal peptide)
    - tmd_count     : (16741,)     float32  - model-predicted # TMDs
    - accessions    : (16741,)     string

Difference from prior shortcut (build_df_adi_aux_features.py):
  - The shortcut precomputed heuristic SP/TMD values from sequence and
    concatenated them as 2 raw columns. ΔF1 was 0.0000.
  - This notebook TRAINS the aux heads jointly with the attention-pool
    localization head, exactly mirroring the Kaggle v2 approach that
    produced the +0.0249 F1 Kaggle lift.

Why df_adi (not Kaggle) is the main benchmark:
  - 16,741 proteins with 7 binary labels (membrane + 6 compartments)
  - 5-fold CV partition structure (partition ∈ {0,1,2,3,4})
  - Same training distribution as the partition_4 0.7904 champion score
  - Direct comparison vs DeepLoc 2.1's 0.647 partition_4 number

Run on Kaggle/Colab:
  1. Open Colab → A100 runtime
  2. Sidebar-upload df_adi.csv (16,741 proteins, 11 columns:
        acc, kingdom, partition, membrane, cytoplasm, nucleus,
        extracellular, cell_surface, mitochondrion, endom, sequence)
  3. Run all → ~35 min wall time on A100
  4. Download df_adi_attn_pool_v2.h5 (~370 MB compressed)

After download:
  Place at ~/Downloads/df_adi_attn_pool_v2.h5
  Then I patch champion_5fold_cv.py to load v2 H5 (attn + sp + tmd) and
  re-run the 5-fold CV.
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "notebooks" / "df_adi_v2_extract_colab.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)

# df_adi-specific labels (7 binary, in canonical order)
DF_ADI_LABELS = [
    'membrane', 'cytoplasm', 'nucleus', 'extracellular',
    'cell_surface', 'mitochondrion', 'endom',
]

CELLS = []

# ── Cell 1: Markdown title
CELLS.append({
    "cell_type": "markdown",
    "source": [
        "# Attention-Pooled ProtT5 L22 + Aux Heads - v2 for df_adi\n",
        "\n",
        "Mirror of the Kaggle v2 notebook but for the **main benchmark** (df_adi).\n",
        "\n",
        "Trains three things jointly on top of frozen ProtT5-XL:\n",
        "  - 24 attention-pool heads (one per layer) → 1024-d pooled feature per layer\n",
        "  - 1 localization head: 24×1024 → 7 binary labels (membrane + 6 compartments)\n",
        "  - 2 auxiliary heads on layer-22 attn-pool: signal-peptide (binary) + TMD count (regression)\n",
        "\n",
        "Multi-task loss: `L = L_loc + 0.3·L_sp + 0.3·L_tmd`\n",
        "\n",
        "Heuristic SP/TMD labels (same Kyte-Doolittle scale as Kaggle v2):\n",
        "  - SP proxy: avg hydrophobicity of first 30 residues ≥ 0.5 → positive\n",
        "  - TMD count: count of 19-residue windows with avg KD ≥ 1.6\n",
        "\n",
        "## Wall time\n",
        "\n",
        "- A100 (40 GB): ~35 min\n",
        "- T4 (15 GB): ~60 min\n",
        "\n",
        "## Output\n",
        "\n",
        "- `df_adi_attn_pool_v2.h5` (downloaded to your local Downloads folder)\n",
        "  - `attn_layer_22` : (16741, 1024) float32\n",
        "  - `sp_score`      : (16741,) float32 in [0,1]\n",
        "  - `tmd_count`     : (16741,) float32 ≥ 0\n",
        "  - `accessions`    : (16741,) string\n",
        "\n",
        "## Required inputs (sidebar upload)\n",
        "\n",
        "- `df_adi.csv` (16,741 proteins, 11 columns:\n",
        "  `acc, kingdom, partition, membrane, cytoplasm, nucleus, extracellular, cell_surface, mitochondrion, endom, sequence`)\n",
    ],
})

# ── Cell 2: Install deps + GPU check
CELLS.append({
    "cell_type": "code",
    "source": [
        "import sys, subprocess, math, json, time, warnings\n",
        "from pathlib import Path\n",
        "import torch\n",
        "print(f'Python: {sys.version.split()[0]}  Torch: {torch.__version__}')\n",
        "print(f'CUDA: {torch.cuda.is_available()}')\n",
        "if torch.cuda.is_available():\n",
        "    print(f'GPU: {torch.cuda.get_device_name(0)} '\n",
        "          f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n",
        "    'transformers>=4.40', 'sentencepiece>=0.2', 'protobuf>=4',\n",
        "    'h5py>=3.10', 'pandas>=2.0', 'numpy>=1.26'], check=True)\n",
        "print('Deps installed.')\n",
    ],
})

# ── Cell 3: Output paths
CELLS.append({
    "cell_type": "code",
    "source": [
        "OUT_H5 = TMP_H5  # downloaded to your local Downloads folder after the run\n",
        "TMP_H5 = Path('/content/df_adi_attn_pool_v2.h5')\n",
        "TMP_H5.parent.mkdir(parents=True, exist_ok=True)\n",
        "EMB_DIM = 1024; LAYER = 22; N_LAYERS = 24\n",
        "BATCH_SIZE = 4; MAX_SEQ_LEN = 1024\n",
        "M = 7  # df_adi compartments (membrane + 6)\n",
        "print(f'Output (will download): {OUT_H5}')\n",
        "print(f'Tmp (Colab-local):      {TMP_H5}')\n",
    ],
})

# ── Cell 4: Upload (sidebar)
CELLS.append({
    "cell_type": "code",
    "source": [
        "from google.colab import files  # noqa: F401\n",
        "ADJ_PATH = Path('/content/df_adi.csv')\n",
        "def status(p, name):\n",
        "    if p.exists():\n",
        "        print(f'  {name:>12s}: {p}  ({p.stat().st_size / 1e6:.1f} MB)')\n",
        "    else:\n",
        "        print(f'  {name:>12s}: MISSING  - upload to /content/')\n",
        "status(ADJ_PATH, 'df_adi.csv')\n",
        "assert ADJ_PATH.exists(), 'df_adi.csv must be uploaded before proceeding.'\n",
    ],
})

# ── Cell 5: Load + inspect
CELLS.append({
    "cell_type": "code",
    "source": [
        "import pandas as pd, numpy as np\n",
        "DF_ADI_LABELS = [\n",
        "    'membrane', 'cytoplasm', 'nucleus', 'extracellular',\n",
        "    'cell_surface', 'mitochondrion', 'endom',\n",
        "]\n",
        "df = pd.read_csv(ADJ_PATH)\n",
        "for col in ['acc', 'sequence'] + DF_ADI_LABELS:\n",
        "    if col not in df.columns:\n",
        "        raise RuntimeError(f'df_adi.csv missing required column: {col!r}')\n",
        "df['acc'] = df['acc'].astype(str)\n",
        "print(f'  df_adi: {len(df)} proteins, {len(DF_ADI_LABELS)} labels')\n",
        "print(f'  partitions: {sorted(df[\"partition\"].unique())}')\n",
        "print(f'  kingdom:    {df[\"kingdom\"].unique()}')\n",
        "seqs_all = df['sequence'].values\n",
        "accs_all = df['acc'].values\n",
        "labels_all = df[DF_ADI_LABELS].values.astype(np.int64)\n",
        "seq_lens = df['sequence'].str.len().values\n",
        "print(f'  seq lengths: min={seq_lens.min()}  max={seq_lens.max()}  '\n",
        "      f'median={int(np.median(seq_lens))}  p95={int(np.percentile(seq_lens, 95))}')\n",
        "print(f'  label prevalence:')\n",
        "for i, lab in enumerate(DF_ADI_LABELS):\n",
        "    print(f'    {lab:>15s}: {labels_all[:, i].mean():.4f}')\n",
    ],
})

# ── Cell 6: SP/TMD heuristic labels
CELLS.append({
    "cell_type": "code",
    "source": [
        "# Kyte-Doolittle hydrophobicity scale (positive = hydrophobic)\n",
        "KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,\n",
        "      'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,\n",
        "      'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,\n",
        "      'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}\n",
        "\n",
        "def sp_proxy(seq):\n",
        "    \"\"\"Has signal peptide? Avg hydrophobicity of first 30 residues >= 0.5.\"\"\"\n",
        "    n_region = seq[:30]\n",
        "    if len(n_region) < 15:\n",
        "        return 0.0\n",
        "    return float(np.mean([KD.get(aa, 0.0) for aa in n_region]) >= 0.5)\n",
        "\n",
        "def tmd_count(seq, window=19, thr=1.6):\n",
        "    \"\"\"Count 19-res windows with avg Kyte-Doolittle >= 1.6 (TMD proxy).\"\"\"\n",
        "    if len(seq) < window:\n",
        "        return 0.0\n",
        "    h = np.array([KD.get(aa, 0.0) for aa in seq[:MAX_SEQ_LEN]], dtype=np.float32)\n",
        "    if len(h) < window:\n",
        "        return 0.0\n",
        "    cs = np.convolve(h, np.ones(window) / window, mode='valid')\n",
        "    return float((cs >= thr).sum())\n",
        "\n",
        "print('Computing SP/TMD heuristic labels for all 16,741 df_adi proteins...')\n",
        "t0 = time.time()\n",
        "sp_labels = np.array([sp_proxy(s) for s in seqs_all], dtype=np.float32)\n",
        "tmd_labels = np.array([tmd_count(s) for s in seqs_all], dtype=np.float32)\n",
        "print(f'  SP rate:        {sp_labels.mean():.3f}')\n",
        "print(f'  TMD count:      mean={tmd_labels.mean():.2f}  max={tmd_labels.max():.0f}')\n",
        "print(f'  computed in {time.time()-t0:.1f}s')\n",
    ],
})

# ── Cell 7: Load ProtT5 + tokenizer
CELLS.append({
    "cell_type": "code",
    "source": [
        "from transformers import T5EncoderModel, T5Tokenizer\n",
        "MODEL_NAME = 'Rostlab/prot_t5_xl_uniref50'\n",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "FP16 = True\n",
        "tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)\n",
        "model = T5EncoderModel.from_pretrained(MODEL_NAME).to(device).eval()\n",
        "if FP16:\n",
        "    model = model.half()\n",
        "print(f'Loaded ProtT5-XL ({len(model.encoder.block)} blocks)')\n",
    ],
})

# ── Cell 8: Tokenizer diagnostic
CELLS.append({
    "cell_type": "code",
    "source": [
        "import re\n",
        "def format_seq(seq):\n",
        "    cleaned = re.sub(r'[UZOB]', 'X', seq.upper())\n",
        "    return ' '.join(list(cleaned))\n",
        "test_seqs = ['MILKRTV', df['sequence'].iloc[0],\n",
        "             df['sequence'].iloc[100], df['sequence'].iloc[-1]]\n",
        "encoded = tokenizer([format_seq(s) for s in test_seqs], return_tensors='pt',\n",
        "                    padding='longest', truncation=True, max_length=MAX_SEQ_LEN,\n",
        "                    add_special_tokens=True)\n",
        "ids = encoded['input_ids']\n",
        "unique = set(tuple(ids[i].tolist()) for i in range(len(test_seqs)))\n",
        "assert len(unique) >= 4, 'Tokenizer bug - distinct sequences produced same IDs'\n",
        "print(f'  tokenizer OK ({ids.shape[1]} tokens, {len(unique)} unique)')\n",
    ],
})

# ── Cell 9: Define AttentionPool + 24 heads + 7-class projection + aux heads
CELLS.append({
    "cell_type": "code",
    "source": [
        "import torch.nn as nn\n",
        "class AttentionPool(nn.Module):\n",
        "    def __init__(self, dim, hidden=64):\n",
        "        super().__init__()\n",
        "        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))\n",
        "    def forward(self, x, mask):\n",
        "        scores = self.score(x).squeeze(-1).masked_fill(~mask.bool(), -1e4)\n",
        "        weights = torch.softmax(scores, dim=1)\n",
        "        return (x * weights.unsqueeze(-1)).sum(dim=1)\n",
        "\n",
        "attn_heads = nn.ModuleList([AttentionPool(EMB_DIM) for _ in range(N_LAYERS)]).to(device)\n",
        "attn_heads.train()\n",
        "# Localization head (24*1024 -> 7 compartments)\n",
        "combined_proj = nn.Linear(N_LAYERS * EMB_DIM, M).to(device)\n",
        "# Aux heads (1024 -> 1 SP + 1 TMD, shared trunk on layer-22 attn)\n",
        "aux_trunk = nn.Sequential(nn.Linear(EMB_DIM, 128), nn.ReLU(True)).to(device)\n",
        "sp_head  = nn.Linear(128, 1).to(device)\n",
        "tmd_head = nn.Linear(128, 1).to(device)\n",
        "\n",
        "optimizer = torch.optim.Adam(\n",
        "    list(attn_heads.parameters())\n",
        "    + list(combined_proj.parameters())\n",
        "    + list(aux_trunk.parameters())\n",
        "    + list(sp_head.parameters())\n",
        "    + list(tmd_head.parameters()),\n",
        "    lr=1e-3,\n",
        ")\n",
        "# Per-class pos_weight on df_adi labels\n",
        "pos = labels_all.sum(axis=0)\n",
        "neg = len(pos) - pos\n",
        "pw = np.clip(neg / np.maximum(pos, 1), 1.0, 20.0).astype(np.float32)\n",
        "loc_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw).to(device))\n",
        "sp_criterion  = nn.BCEWithLogitsLoss()\n",
        "tmd_criterion = nn.PoissonNLLLoss(log_input=False)\n",
        "\n",
        "print(f'  attn heads:    {sum(p.numel() for p in attn_heads.parameters()):,}')\n",
        "print(f'  combined_proj: {sum(p.numel() for p in combined_proj.parameters()):,}')\n",
        "print(f'  aux trunk:     {sum(p.numel() for p in aux_trunk.parameters()):,}')\n",
        "print(f'  SP head:       {sum(p.numel() for p in sp_head.parameters()):,}')\n",
        "print(f'  TMD head:      {sum(p.numel() for p in tmd_head.parameters()):,}')\n",
        "print(f'  pos_weight:    {pw.round(2).tolist()}')\n",
    ],
})

# ── Cell 10: Multi-task training
CELLS.append({
    "cell_type": "code",
    "source": [
        "np.random.seed(42)\n",
        "sample_idx = np.random.choice(len(df), size=min(2000, len(df)), replace=False)\n",
        "print(f'Multi-task training on {len(sample_idx)} df_adi proteins for 1 epoch...')\n",
        "attn_heads.train(); combined_proj.train(); aux_trunk.train()\n",
        "sp_head.train(); tmd_head.train()\n",
        "t_train = time.time()\n",
        "for b_idx, batch_start in enumerate(range(0, len(sample_idx), BATCH_SIZE)):\n",
        "    ix = sample_idx[batch_start:batch_start + BATCH_SIZE]\n",
        "    batch_seqs = list(seqs_all[ix])\n",
        "    batch_labels = labels_all[ix]\n",
        "    batch_sp  = torch.from_numpy(sp_labels[ix]).to(device)\n",
        "    batch_tmd = torch.from_numpy(tmd_labels[ix]).to(device)\n",
        "    seqs_fmt = [format_seq(s) for s in batch_seqs]\n",
        "    enc = tokenizer(seqs_fmt, return_tensors='pt', padding='longest',\n",
        "                    truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)\n",
        "    input_ids = enc['input_ids'].to(device)\n",
        "    attn_mask = enc['attention_mask'].to(device)\n",
        "    mask = attn_mask.bool()\n",
        "    with torch.no_grad():\n",
        "        if FP16:\n",
        "            with torch.autocast(device_type='cuda'):\n",
        "                outputs = model.encoder(input_ids=input_ids,\n",
        "                                          attention_mask=attn_mask,\n",
        "                                          output_hidden_states=True)\n",
        "        else:\n",
        "            outputs = model.encoder(input_ids=input_ids,\n",
        "                                      attention_mask=attn_mask,\n",
        "                                      output_hidden_states=True)\n",
        "    all_hidden = outputs.hidden_states\n",
        "    pooled_per_layer = []\n",
        "    for layer_i in range(N_LAYERS):\n",
        "        z = attn_heads[layer_i](all_hidden[layer_i + 1].float(), mask)\n",
        "        pooled_per_layer.append(z)\n",
        "    pooled_cat = torch.cat(pooled_per_layer, dim=-1)\n",
        "    loc_logits = combined_proj(pooled_cat)\n",
        "    loc_loss = loc_criterion(loc_logits,\n",
        "                              torch.from_numpy(batch_labels.astype(np.float32)).to(device))\n",
        "    # Aux losses (use attn_layer_22 specifically)\n",
        "    l22_attn = pooled_per_layer[LAYER]\n",
        "    aux_feat = aux_trunk(l22_attn)\n",
        "    sp_logit = sp_head(aux_feat).squeeze(-1)\n",
        "    tmd_logit = tmd_head(aux_feat).squeeze(-1)\n",
        "    sp_loss = sp_criterion(sp_logit, batch_sp)\n",
        "    tmd_loss = tmd_criterion(tmd_logit, batch_tmd)\n",
        "    loss = loc_loss + 0.3 * sp_loss + 0.3 * tmd_loss\n",
        "    optimizer.zero_grad(); loss.backward()\n",
        "    torch.nn.utils.clip_grad_norm_(\n",
        "        list(attn_heads.parameters()) + list(combined_proj.parameters())\n",
        "        + list(aux_trunk.parameters()) + list(sp_head.parameters())\n",
        "        + list(tmd_head.parameters()), max_norm=1.0)\n",
        "    if torch.isnan(loss):\n",
        "        optimizer.zero_grad(); continue\n",
        "    optimizer.step()\n",
        "    if b_idx % 25 == 0:\n",
        "        print(f'  batch {b_idx:3d}  loc={loc_loss.item():.3f} '\n",
        "              f'sp={sp_loss.item():.3f}  tmd={tmd_loss.item():.3f}')\n",
        "attn_heads.eval(); combined_proj.eval(); aux_trunk.eval()\n",
        "sp_head.eval(); tmd_head.eval()\n",
        "print(f'\\nTraining done in {time.time() - t_train:.0f}s')\n",
    ],
})

# ── Cell 11: Extract attn + apply aux heads to all 16,741 proteins
CELLS.append({
    "cell_type": "code",
    "source": [
        "mmap_path = Path('/content/df_adi_attn_pool_v2_memmap.npy')\n",
        "STATUS_PATH = Path('/content/df_adi_attn_pool_v2_status.json')\n",
        "SP_PATH  = Path('/content/df_adi_attn_pool_v2_sp.npy')\n",
        "TMD_PATH = Path('/content/df_adi_attn_pool_v2_tmd.npy')\n",
        "n_total = len(accs_all)\n",
        "extraction_shape = (n_total, N_LAYERS, EMB_DIM)\n",
        "\n",
        "if not mmap_path.exists():\n",
        "    mmap = np.lib.format.open_memmap(str(mmap_path), mode='w+',\n",
        "                                      dtype=np.float16, shape=extraction_shape)\n",
        "else:\n",
        "    mmap = np.lib.format.open_memmap(str(mmap_path), mode='r+',\n",
        "                                      dtype=np.float16, shape=extraction_shape)\n",
        "sp_arr  = np.zeros(n_total, dtype=np.float32)\n",
        "tmd_arr = np.zeros(n_total, dtype=np.float32)\n",
        "if SP_PATH.exists():  sp_arr  = np.load(SP_PATH)\n",
        "if TMD_PATH.exists(): tmd_arr = np.load(TMD_PATH)\n",
        "status = {}\n",
        "if STATUS_PATH.exists():\n",
        "    status = json.loads(STATUS_PATH.read_text())\n",
        "skip_to = sum(1 for v in status.values() if v.get('done'))\n",
        "\n",
        "attn_heads.eval(); aux_trunk.eval(); sp_head.eval(); tmd_head.eval()\n",
        "total_batches = math.ceil((n_total - skip_to) / BATCH_SIZE)\n",
        "batch_times = []\n",
        "t_start = time.time()\n",
        "try:\n",
        "    for b_idx in range(total_batches):\n",
        "        t_b = time.time()\n",
        "        i0 = skip_to + b_idx * BATCH_SIZE\n",
        "        i1 = min(i0 + BATCH_SIZE, n_total)\n",
        "        batch_seqs = list(seqs_all[i0:i1])\n",
        "        batch_ids = list(range(i0, i1))\n",
        "        seqs_fmt = [format_seq(s) for s in batch_seqs]\n",
        "        enc = tokenizer(seqs_fmt, return_tensors='pt', padding='longest',\n",
        "                        truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)\n",
        "        input_ids = enc['input_ids'].to(device)\n",
        "        attn_mask = enc['attention_mask'].to(device)\n",
        "        mask = attn_mask.bool()\n",
        "        with torch.no_grad():\n",
        "            if FP16:\n",
        "                with torch.autocast(device_type='cuda'):\n",
        "                    outputs = model.encoder(input_ids=input_ids,\n",
        "                                              attention_mask=attn_mask,\n",
        "                                              output_hidden_states=True)\n",
        "            else:\n",
        "                outputs = model.encoder(input_ids=input_ids,\n",
        "                                          attention_mask=attn_mask,\n",
        "                                          output_hidden_states=True)\n",
        "        all_hidden = outputs.hidden_states\n",
        "        # Save attn for all 24 layers\n",
        "        for layer_i in range(N_LAYERS):\n",
        "            hidden = all_hidden[layer_i + 1].float()\n",
        "            with torch.no_grad():\n",
        "                attn_p = attn_heads[layer_i](hidden, mask).cpu().numpy().astype(np.float16)\n",
        "            for local_idx, g_idx in enumerate(batch_ids):\n",
        "                mmap[g_idx, layer_i, :] = attn_p[local_idx]\n",
        "        mmap.flush()\n",
        "        # Apply aux heads on layer-22 attn\n",
        "        with torch.no_grad():\n",
        "            l22_attn = attn_heads[LAYER](all_hidden[LAYER + 1].float(), mask)\n",
        "            aux_feat = aux_trunk(l22_attn)\n",
        "            sp_logits = sp_head(aux_feat).squeeze(-1)\n",
        "            tmd_logits = tmd_head(aux_feat).squeeze(-1)\n",
        "            sp_scores  = torch.sigmoid(sp_logits).cpu().numpy()\n",
        "            tmd_counts = torch.nn.functional.softplus(tmd_logits).cpu().numpy()\n",
        "        for local_idx, g_idx in enumerate(batch_ids):\n",
        "            sp_arr[g_idx]  = sp_scores[local_idx]\n",
        "            tmd_arr[g_idx] = tmd_counts[local_idx]\n",
        "        np.save(SP_PATH, sp_arr)\n",
        "        np.save(TMD_PATH, tmd_arr)\n",
        "        for g_idx in batch_ids:\n",
        "            status[str(g_idx)] = {'done': True}\n",
        "        elapsed = time.time() - t_b\n",
        "        batch_times.append(elapsed)\n",
        "        avg_bt = sum(batch_times) / len(batch_times)\n",
        "        remaining = (total_batches - b_idx - 1) * avg_bt\n",
        "        if (b_idx + 1) % 25 == 0 or b_idx == 0:\n",
        "            STATUS_PATH.write_text(json.dumps(status))\n",
        "            print(f'  batch {b_idx + 1}/{total_batches}  rows {i0}-{i1 - 1}  '\n",
        "                  f'{elapsed:.1f}s  ETA={remaining:.0f}s')\n",
        "\n",
        "except Exception as e:\n",
        "    STATUS_PATH.write_text(json.dumps(status))\n",
        "    print(f'CRASHED at batch {b_idx + 1}: {type(e).__name__}: {e}')\n",
        "    print('Checkpoint saved. Re-run to resume.')\n",
        "    raise e\n",
        "\n",
        "STATUS_PATH.write_text(json.dumps(status))\n",
        "print(f'\\nExtraction done: {n_total} proteins × {N_LAYERS} attn layers + aux preds')\n",
        "print(f'Wall time: {time.time() - t_start:.0f}s')\n",
    ],
})

# ── Cell 12: Write H5 + verify + download
CELLS.append({
    "cell_type": "code",
    "source": [
        "import h5py\n",
        "if TMP_H5.exists():\n",
        "    TMP_H5.unlink()\n",
        "mmap = np.lib.format.open_memmap(str(mmap_path), mode='r',\n",
        "                                  dtype=np.float16, shape=extraction_shape)\n",
        "accs_arr = np.array(accs_all, dtype=h5py.string_dtype(encoding='utf-8'))\n",
        "with h5py.File(str(TMP_H5), 'w') as h5:\n",
        "    h5.create_dataset('attn_layer_22', data=mmap[:, LAYER, :].astype(np.float32),\n",
        "                       compression='gzip', compression_opts=4)\n",
        "    h5.create_dataset('attn_avg', data=np.mean(mmap[:, :, :].astype(np.float32), axis=1),\n",
        "                       compression='gzip', compression_opts=4)\n",
        "    h5.create_dataset('sp_score', data=sp_arr.astype(np.float32))\n",
        "    h5.create_dataset('tmd_count', data=tmd_arr.astype(np.float32))\n",
        "    h5.create_dataset('accessions', data=accs_arr)\n",
        "print(f'\\n  H5 size: {TMP_H5.stat().st_size / 1e6:.1f} MB')\n",
        "\n",
        "print('\\n=== H5 VERIFICATION ===')\n",
        "with h5py.File(str(TMP_H5), 'r') as h5:\n",
        "    keys = sorted(h5.keys())\n",
        "    print(f'  keys: {keys}')\n",
        "    for k in keys:\n",
        "        ds = h5[k]\n",
        "        if ds.dtype.kind in ('S','U','O'):\n",
        "            sample = ds[0]\n",
        "            if isinstance(sample, bytes): sample = sample.decode('utf-8')\n",
        "            print(f'    {k}: shape={ds.shape}  dtype={ds.dtype}  sample={sample!r}')\n",
        "        else:\n",
        "            arr = ds[:]\n",
        "            print(f'    {k}: shape={arr.shape}  mean={arr.mean():.3f}  std={arr.std():.3f}')\n",
        "    n_chk = h5['attn_layer_22'].shape[0]\n",
        "    assert n_chk == n_total\n",
        "    assert h5['attn_layer_22'].shape == (n_total, EMB_DIM)\n",
        "    assert h5['sp_score'].shape == (n_total,)\n",
        "    assert h5['tmd_count'].shape == (n_total,)\n",
        "    print('  format-verified ')\n",
        "\n",
        "from google.colab import files as colab_files\n",
        "print(f'\\nDownloading: {TMP_H5} ({TMP_H5.stat().st_size / 1e6:.1f} MB)')\n",
        "colab_files.download(str(TMP_H5))\n",
    ],
})

# ── Cell 13: Cleanup
CELLS.append({
    "cell_type": "code",
    "source": [
        "if mmap_path.exists():\n",
        "    size_gb = mmap_path.stat().st_size / 1e9\n",
        "    confirm = input(f'Remove memmap ({size_gb:.1f} GB)? (y/n): ')\n",
        "    if confirm.lower().startswith('y'):\n",
        "        for p in [mmap_path, STATUS_PATH, SP_PATH, TMD_PATH]:\n",
        "            if p.exists(): p.unlink()\n",
        "        print('Cleaned.')\n",
        "    else:\n",
        "        print(f'Kept. Delete manually: {mmap_path}')\n",
    ],
})


nb = {
    "cells": [
        {"cell_type": c["cell_type"], "metadata": {}, "execution_count": None,
         "outputs": [], "source": c["source"]}
        for c in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"  Wrote: {OUT}  ({OUT.stat().st_size / 1024:.1f} KB, {len(CELLS)} cells)")
