#!/usr/bin/env python3
"""eval_against_deeploc_2_1.py

Apples-to-apples comparison evaluator. Takes DeepLoc 2.1's 14-column output
CSV (10 subcellular + 4 membrane probabilities per protein), maps it to
our 7-compartment scheme, and computes metrics matching
tinker11_diag_p4_metrics.py exactly.

INPUT (DeepLoc 2.1 typical output CSV):
    columns = [protein_id, sequence,
               Nucleus, Cytoplasm, Extracellular, Cell_membrane,
               Mitochondrion, Endoplasmic_reticulum, Lysosome/Vacuole,
               Golgi_apparatus, Peroxisome, Plastid,
               Transmembrane, Peripheral, Lipid-anchored, Soluble]

OUTPUT (in /Volumes/BOMBOCLAT/project_JL/output_eval_against_deeploc/):
    per_compartment.csv    : per-compartment accuracy/recall/precision/F1
    headline_table.md      : head-to-head vs tinker11
    deeploc_metrics.json   : full per-compartment + global metrics + raw preds

To run on real DeepLoc predictions:
    1. Acquire DTU's DeepLoc 2.1 package (academic download form).
    2. Run on df_adi partition-4 sequences.
    3. Output the CSV in the schema above.
    4. Run: python3 eval_against_deeploc_2_1.py /path/to/deeploc_p4.csv

To run a smoke-test (mocks DeepLoc's output):
    The script auto-detects the partition-4 ground truth and reads df_adi;
    if DL output is missing, --mock flag generates random probs to verify
    the eval pipeline structure works.
"""
import csv, json, os, argparse, sys, warnings
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, hamming_loss,
    log_loss,
)

# ---- paths ----
PROJ = Path('/Volumes/BOMBOCLAT/project_JL').resolve()
SRC_CSV = PROJ / 'data' / 'df_adi.csv'
OUT_DIR = PROJ / 'output_eval_against_deeploc'
OUT_DIR.mkdir(exist_ok=True)

# ---- label mapping ----
OUR_7 = ['Membrane','Cytoplasm','Nucleus','Extracellular','Cell_surf','Mito','Endom']
MAPPING = {
    'Membrane':      'derived',   # max(Transmembrane, Peripheral, Lipid-anchored) > t
    'Cytoplasm':     'direct',    # 1-to-1 from DL's Cytoplasm
    'Nucleus':       'direct',
    'Extracellular': 'direct',
    'Cell_surf':     'direct',    # 1-to-1 from DL's Cell.membrane
    'Mito':          'direct',    # 1-to-1 from DL's Mitochondrion
    'Endom':         'direct',    # 1-to-1 from DL's Endoplasmic.reticulum
}
OUR_LABEL_COLS = ['membrane','cytoplasm','nucleus','extracellular',
                  'cell_surface','mitochondrion','endom']

# DeepLoc 2.1 column names - accept whatever spelling DTU's CSV uses (dot, underscore, space, hyphen).
# Map_dl_to_ours uses the alias resolver below to handle all of them.
DL_SUB = ['Nucleus','Cytoplasm','Extracellular','Cell.membrane','Mitochondrion',
          'Endoplasmic.reticulum','Lysosome/Vacuole','Golgi.apparatus',
          'Peroxisome','Plastid']
DL_MEM = ['Transmembrane','Peripheral','Lipid.anchored','Soluble']

# --- alias candidates per column ---
# Tries dot / underscore / space / hyphen variants. First match wins.
DL_KEY_ALIASES = {
    'Nucleus':                ['Nucleus', 'nucleus'],
    'Cytoplasm':              ['Cytoplasm', 'cytoplasm'],
    'Extracellular':          ['Extracellular', 'extracellular'],
    'Cell.membrane':          ['Cell.membrane', 'Cell_membrane', 'Cell membrane', 'Cell-membrane', 'cell.membrane', 'cell_membrane'],
    'Mitochondrion':          ['Mitochondrion', 'mitochondrion'],
    'Endoplasmic.reticulum':  ['Endoplasmic.reticulum', 'Endoplasmic_reticulum', 'Endoplasmic reticulum', 'Endoplasmic-reticulum', 'endoplasmic_reticulum'],
    'Lysosome/Vacuole':       ['Lysosome/Vacuole', 'Lysosome_Vacuole', 'Lysosome Vacuole', 'lysosome/vacuole'],
    'Golgi.apparatus':        ['Golgi.apparatus', 'Golgi_apparatus', 'Golgi apparatus', 'golgi_apparatus'],
    'Peroxisome':             ['Peroxisome', 'peroxisome'],
    'Plastid':                ['Plastid', 'plastid'],
    'Transmembrane':          ['Transmembrane', 'transmembrane'],
    'Peripheral':             ['Peripheral', 'peripheral'],
    'Lipid.anchored':         ['Lipid.anchored', 'Lipid-anchored', 'Lipid_anchored', 'Lipid anchored', 'lipid_anchored'],
    'Soluble':                ['Soluble', 'soluble'],
}


def _resolve(row, canonical_key):
    """Look up row[canonical_key] trying every alias. Falls back to 0.0."""
    for alias in DL_KEY_ALIASES.get(canonical_key, [canonical_key]):
        if alias in row:
            return row[alias]
    return 0.0


def map_dl_to_ours(row, threshold: float = 0.5, dl_thresholds: Dict[str,float] = None) -> List[int]:
    """Map one DeepLoc output row to our 7-compartment prediction vector.

    Returns list of 7 binary integers in OUR_7 order.

    Robust to DTU's CSV format choice: dot / underscore / space / hyphen
    all flow through the same probability lookup via _resolve().
    """
    p = []
    # 1) Membrane = max(Transmembrane, Peripheral, Lipid-anchored)
    thr = (dl_thresholds or {}).get('Membrane', threshold)
    memb_prob = max(_resolve(row, 'Transmembrane'),
                    _resolve(row, 'Peripheral'),
                    _resolve(row, 'Lipid.anchored'))
    p.append(1 if memb_prob > thr else 0)
    # 2) Cytoplasm
    thr = (dl_thresholds or {}).get('Cytoplasm', threshold)
    p.append(1 if _resolve(row, 'Cytoplasm') > thr else 0)
    # 3) Nucleus
    thr = (dl_thresholds or {}).get('Nucleus', threshold)
    p.append(1 if _resolve(row, 'Nucleus') > thr else 0)
    # 4) Extracellular
    thr = (dl_thresholds or {}).get('Extracellular', threshold)
    p.append(1 if _resolve(row, 'Extracellular') > thr else 0)
    # 5) Cell surface ← DL's Cell.membrane
    thr = (dl_thresholds or {}).get('Cell_surf', threshold)
    p.append(1 if _resolve(row, 'Cell.membrane') > thr else 0)
    # 6) Mitochondrion
    thr = (dl_thresholds or {}).get('Mito', threshold)
    p.append(1 if _resolve(row, 'Mitochondrion') > thr else 0)
    # 7) Endom ← DL's Endoplasmic.reticulum
    thr = (dl_thresholds or {}).get('Endom', threshold)
    p.append(1 if _resolve(row, 'Endoplasmic.reticulum') > thr else 0)
    return p


def evaluate_full(probs_per_row, Y_true, threshold=0.5, dl_thresholds=None):
    """Compute metrics matching tinker11's evaluate_full signature."""
    preds_arr = np.array([map_dl_to_ours(r, threshold, dl_thresholds)
                          for r in probs_per_row], dtype=int)
    Y_true = np.asarray(Y_true, dtype=int)
    per = {'accuracy':[],'recall':[],'precision':[],'f1':[]}
    for j in range(7):
        yt = Y_true[:,j]; yp = preds_arr[:,j]
        per['accuracy'].append(float(accuracy_score(yt,yp)))
        per['recall'].append(float(recall_score(yt,yp,zero_division=0)))
        per['precision'].append(float(precision_score(yt,yp,zero_division=0)))
        per['f1'].append(float(f1_score(yt,yp,zero_division=0)))
    flat_t = Y_true.flatten()
    flat_p = preds_arr.flatten()
    overall = {
        'accuracy': float(accuracy_score(flat_t, flat_p)),
        'recall':   float(recall_score(flat_t, flat_p, zero_division=0)),
        'precision':float(precision_score(flat_t, flat_p, zero_division=0)),
        'f1_micro': float(f1_score(flat_t, flat_p, average='micro', zero_division=0)),
        'f1_macro': float(np.mean(per['f1'])),
        'accuracy_macro':  float(np.mean(per['accuracy'])),
        'recall_macro':    float(np.mean(per['recall'])),
        'precision_macro': float(np.mean(per['precision'])),
        'hamming_loss':    float(hamming_loss(flat_t, flat_p)),
    }
    return {'per_compartment': per, 'overall': overall, 'preds': preds_arr}


def load_deeploc_csv(path):
    """Loads the DeepLoc 2.1 output CSV. Returns (rows, header)."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        row = r.to_dict()
        rows.append(row)
    return rows, list(df.columns)


def write_outputs(metrics, out_csv, out_json, out_md):
    """Write the result in tinker11-compatible format."""
    per = metrics['per_compartment']; ov = metrics['overall']
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['compartment','mapping','accuracy','recall','precision','f1'])
        for j, c in enumerate(OUR_7):
            w.writerow([c,
                        MAPPING[c],
                        f"{per['accuracy'][j]:.4f}",
                        f"{per['recall'][j]:.4f}",
                        f"{per['precision'][j]:.4f}",
                        f"{per['f1'][j]:.4f}"])
    with open(out_json, 'w') as f:
        json.dump({'per_compartment': per, 'overall': ov, 'mapping': MAPPING}, f, indent=2)
    md = []
    md.append(f'## DeepLoc 2.1 metrics on partition-4 holdout (3,276 proteins)\n')
    md.append(f'### Overall (micro over 7 compartments × 3,276 = 22,932 binary samples)')
    md.append(f'- accuracy:   **{ov["accuracy"]:.4f}**')
    md.append(f'- recall:     **{ov["recall"]:.4f}**')
    md.append(f'- precision:  **{ov["precision"]:.4f}**')
    md.append(f'- F1_micro:   **{ov["f1_micro"]:.4f}**')
    md.append(f'- F1_macro:   **{ov["f1_macro"]:.4f}**')
    md.append(f'- Hamming loss: **{ov["hamming_loss"]:.4f}**\n')
    md.append('### Macro (mean over 7 compartments)')
    md.append(f'- accuracy_macro:   **{ov["accuracy_macro"]:.4f}**')
    md.append(f'- recall_macro:     **{ov["recall_macro"]:.4f}**')
    md.append(f'- precision_macro:  **{ov["precision_macro"]:.4f}**')
    md.append(f'- F1_macro:         **{ov["f1_macro"]:.4f}**\n')
    md.append('### Per-compartment')
    md.append('| compartment | mapping | accuracy | recall | precision | F1 |')
    md.append('|---|---:|---:|---:|---:|---:|')
    for j, c in enumerate(OUR_7):
        md.append(f'| {c} | {MAPPING[c]} | {per["accuracy"][j]:.4f} | {per["recall"][j]:.4f} | '
                  f'{per["precision"][j]:.4f} | {per["f1"][j]:.4f} |')
    md.append('')
    md.append('**Mapping legend:**')
    md.append('- **direct**: 1-to-1 mapping from a single DeepLoc probability column')
    md.append('- **derived**: Aggregated from multiple DeepLoc columns (Membrane = max(Transmembrane, Peripheral, Lipid-anchored) > t)')
    with open(out_md, 'w') as f:
        f.write('\n'.join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dl_csv', nargs='?', default=None,
                    help='Path to DeepLoc 2.1 output CSV. If not provided, uses --mock.')
    ap.add_argument('--mock', action='store_true',
                    help='Generate mock DeepLoc predictions (random uniform) for smoke-testing.')
    ap.add_argument('--threshold-strategy', choices=['fixed_0_5','mcc_default'],
                    default='fixed_0_5',
                    help='fixed_0_5 (apples-to-apples vs our pipeline) or mcc_default (DeepLoc-recommended per-class).')
    ap.add_argument('--partition', type=int, default=4,
                    help='Partition number to evaluate against (0-4). Default 4.')
    args = ap.parse_args()

    # --- Load ground truth ---
    src = pd.read_csv(SRC_CSV)
    gt = src[src['partition'] == args.partition].reset_index(drop=True)
    Y_true = gt[OUR_LABEL_COLS].values.astype(int)
    print(f'Ground truth: {Y_true.shape} (proteins × 7 compartments) - partition {args.partition} ({len(gt)} rows)')

    # --- Load DeepLoc predictions or mock them ---
    if args.mock or args.dl_csv is None:
        print('No DeepLoc CSV provided - generating MOCK uniform random probs for smoke-test.')
        rng = np.random.RandomState(42)
        rows = []
        for i in range(len(gt)):
            dict_row = dict(zip(DL_SUB + DL_MEM,
                                rng.uniform(0, 1, len(DL_SUB) + len(DL_MEM))))
            rows.append(dict_row)
    elif not Path(args.dl_csv).exists():
        sys.exit(f'ERROR: {args.dl_csv} does not exist. Pass --mock for smoke-test.')
    else:
        rows, _hdr = load_deeploc_csv(args.dl_csv)
        print(f'Loaded DeepLoc CSV: {args.dl_csv} ({len(rows)} rows)')

    # --- Threshold strategy ---
    # Currently only fixed_0_5 is implemented because DL's per-class MCC
    # thresholds aren't published and would need extracting from DTU.
    threshold = 0.5
    dl_thr = None

    # --- Eval ---
    metrics = evaluate_full(rows, Y_true, threshold=threshold, dl_thresholds=dl_thr)

    # --- Outputs ---
    tag = 't05' if args.threshold_strategy == 'fixed_0_5' else 'mcc'
    out_csv = OUT_DIR / f'per_compartment_{tag}.csv'
    out_json = OUT_DIR / f'deeploc_metrics_{tag}.json'
    out_md = OUT_DIR / f'deeploc_metrics_{tag}.md'
    write_outputs(metrics, out_csv, out_json, out_md)
    print(f'Saved: {out_csv}\n        {out_json}\n        {out_md}')

    # --- Print + headline ---
    ov = metrics['overall']
    print(f'\n=== DeepLoc 2.1 OVERALL (partition {args.partition}) ===')
    print(f'accuracy    : {ov["accuracy"]:.4f}')
    print(f'recall      : {ov["recall"]:.4f}')
    print(f'precision   : {ov["precision"]:.4f}')
    print(f'F1_micro    : {ov["f1_micro"]:.4f}')
    print(f'F1_macro    : {ov["f1_macro"]:.4f}     ← primary headline metric')
    print(f'Hamming loss: {ov["hamming_loss"]:.4f}')
    if args.mock:
        print('\nWARNING:  These numbers are from MOCK random probs. They are structurally correct (verify pipeline) but not a real comparison.')

    # --- Build head-to-head table ---
    # Map: metric label -> (ours_value, overall_key_for_dl)
    ours = {
        'accuracy':     0.9091,
        'recall':       0.8168,
        'precision':    0.7813,
        'F1_macro':     0.8011,
        'F1_micro':     0.9091,
    }
    ov_key_for = {
        'accuracy':  'accuracy',
        'recall':    'recall',
        'precision': 'precision',
        'F1_macro':  'f1_macro',
        'F1_micro':  'f1_micro',
    }
    md = []
    md.append(f'# Head-to-head: ours (attn+SPACE+clean@0.50) vs DeepLoc 2.1  (partition {args.partition})\n')
    md.append(f'Ours = tinker11 (single seed, single fold). DeepLoc = this run (mock data if --mock used).')
    md.append('')
    md.append('| Metric | Ours (tinker11) | DeepLoc 2.1 (t=0.5) | Δ (us − DL) |')
    md.append('|---|---:|---:|---:|')
    for k, ours_v in ours.items():
        dl_v = ov[ov_key_for[k]]
        d = ours_v - dl_v
        md.append(f'| {k} | {ours_v:.4f} | {dl_v:.4f} | {d:+.4f} |')
    md.append('')
    md.append('Note: positive Δ means our pipeline scores higher '
              '(better for accuracy/F1/recall/precision).')
    md.append('\n\nNote: A positive Δ means our pipeline is higher (better for accuracy/F1/recall/precision, lower for Hamming loss).')
    md.append('The Ours column reflects tinker11 (single seed, single fold, c=0.50 attn+SPACE+clean).')
    md.append('The DeepLoc 2.1 column reflects this script running on DTU\'s model output (or mock random data - see input).')
    md.append('')
    md.append('### Per-compartment breakdown with mapping type')
    md.append('| Compartment | Mapping | Ours (F1) | DeepLoc (F1) | Δ | Winner |')
    md.append('|---|---:|---:|---:|---:|:---|')
    per = metrics['per_compartment']
    # Re-read ours per-class F1 from tinker11 fallback values
    ours_perclass = {
        'Membrane': 0.8344, 'Cytoplasm': 0.7656, 'Nucleus': 0.8260,
        'Extracellular': 0.8886, 'Cell_surf': 0.7570, 'Mito': 0.8432,
        'Endom': 0.6926,
    }
    for j, c in enumerate(OUR_7):
        dl_f1 = per['f1'][j]
        our_f1 = ours_perclass[c]
        d = our_f1 - dl_f1
        star = ' ' if abs(d) > 0.05 else ''
        md.append(f'| {c} | {MAPPING[c]} | {our_f1:.4f} | {dl_f1:.4f} | {d:+.4f} | {"Us " if d > 0 else "DL"}{star} |')
    md.append('')
    md.append('**Mapping legend:**')
    md.append('- **direct**: 1-to-1 from a single DeepLoc probability column')
    md.append('- **derived**: Aggregated from multiple DL columns (Membrane = max(Transmembrane, Peripheral, Lipid-anchored) > t)')
    md_path = OUT_DIR / f'headline_table_{tag}.md'
    md_path.write_text('\n'.join(md))
    print(f'\nHeadline table: {md_path}\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
