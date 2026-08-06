#!/usr/bin/env python3
"""tinker11_diag_p4_metrics.py

Diagnostic: re-run the specific (pool, layer, cutoff, fold=4) configs that
produced the high scores, but LOG ALL METRICS (accuracy / recall / macro-F1)
per compartment + overall - since tinker9/tinker10 only logged F1.

Two configs:
  A. attn L22 + SPACE + cleanlab@0.50, partition 4  → wrote 0.8011 in tinker9
  B. mean L22 + SPACE + cleanlab@0.30, partition 4  → mean's best at 0.7926 (tinker10)

Wall time: ~3-5 min (18 MLP trainings total - 9 per config).
"""
import json, os, time, warnings
from pathlib import Path
import h5py, numpy as np, pandas as pd

import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cleanlab.multilabel_classification.rank import get_label_quality_scores

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

PROJ = Path("/Volumes/BOMBOCLAT/project_JL").resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
ATTN_H5 = PROJ / "data" / "prott5_attn_all_layers.h5"
MEAN_H5 = PROJ / "data" / "prott5_all_layers_dfadi-3.h5"
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
OUT_DIR = PROJ / "output_tinker11_diag_p4_metrics"
OUT_DIR.mkdir(exist_ok=True)

HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5
K_TARGET = 4   # only run partition 4

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]


class MLP(nn.Module):
    def __init__(self, indim, hdim, outdim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(indim, hdim), nn.ReLU(True),
                                 nn.Dropout(dropout), nn.Linear(hdim, outdim))
    def forward(self, x): return self.net(x)


def compute_pos_weight(Y):
    pw = np.ones(M, dtype=np.float32)
    for j in range(M):
        pos = float(Y[:, j].sum()); neg = float(Y.shape[0]) - pos
        pw[j] = 1.0 if pos <= 0 else min(20.0, neg / pos)
    return np.clip(pw, 1.0, 20.0)


def train_mlp(Xtr, Ytr, Xte, seed=42):
    sc = StandardScaler()
    Xts = sc.fit_transform(Xtr).astype(np.float32)
    Xtes = sc.transform(Xte).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    ti, ei = train_test_split(np.arange(len(Xts)), test_size=ES_FRAC, random_state=seed)
    pw = compute_pos_weight(Ytr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw.astype(np.float32)))
    model = MLP(Xts.shape[1], HIDDEN, M, DROPOUT)
    opt = optim.Adam(model.parameters(), lr=LR)
    Xt = torch.from_numpy(Xts); Yt = torch.from_numpy(Ytr.astype(np.float32))
    Xe = torch.from_numpy(Xts[ei])
    best_f1, best_state, stall = -1.0, None, 0
    for ep in range(1, MAX_EP + 1):
        model.train(); perm = torch.randperm(len(ti))
        for s in range(0, len(ti), BATCH_SIZE):
            ix = perm[s:s + BATCH_SIZE]
            criterion(model(Xt[ix]), Yt[ix]).backward(); opt.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            ep_ = torch.sigmoid(model(Xe)).numpy()
        ef = float(np.mean([
            f1_score(Ytr[ei, j].astype(int), (ep_[:, j] >= THR).astype(int), zero_division=0)
            for j in range(M)
        ]))
        if ef > best_f1 + 1e-6:
            best_f1 = ef
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= PATIENCE: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(Xtes))).numpy().astype(np.float32)
    return probs


def gen_oof(X, Y, seed=42, n_folds=4):
    n = len(X); oof = np.zeros((n, M), dtype=np.float32)
    rng = np.random.RandomState(seed); idx = np.arange(n); rng.shuffle(idx)
    fs = n // n_folds
    for f in range(n_folds):
        vs = f * fs; ve = n if f == n_folds - 1 else (f + 1) * fs
        vi = idx[vs:ve]; ti = np.concatenate([idx[:vs], idx[ve:]])
        oof[vi] = train_mlp(X[ti], Y[ti], X[vi], seed=seed + f)
    return oof


def cleanlab_step(Y, oof, cutoff):
    labs = [list(np.where(Y[i] == 1)[0]) for i in range(len(Y))]
    scores = get_label_quality_scores(labels=labs, pred_probs=oof.astype(np.float64),
                                       method="self_confidence", adjust_pred_probs=True)
    keep = scores >= cutoff
    return keep, scores


def evaluate_full(probs, Y):
    """Per-compartment AND overall: accuracy, recall, precision, F1.
    Returns: dict with per-compartment lists + overall numbers.

    The task is multi-label (7 compartments). Metrics are computed over the
    FLATTENED predictions vs FLATTENED labels (i.e. micro-averaged across all
    (row, compartment) pairs) for a single 'overall' number.

    'macro' here means mean over the 7 compartments.
    """
    preds = (probs >= THR).astype(int)
    per = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    for j in range(M):
        y_true = Y[:, j].astype(int); y_pred = preds[:, j].astype(int)
        per['accuracy'].append(float(accuracy_score(y_true, y_pred)))
        per['recall'].append(float(recall_score(y_true, y_pred, zero_division=0)))
        per['precision'].append(float(precision_score(y_true, y_pred, zero_division=0)))
        per['f1'].append(float(f1_score(y_true, y_pred, zero_division=0)))
    # Overall (micro over 7 × N samples)
    flat_t = Y.flatten().astype(int)
    flat_p = preds.flatten().astype(int)
    overall = {
        'accuracy':  float(accuracy_score(flat_t, flat_p)),
        'recall':    float(recall_score(flat_t, flat_p, zero_division=0)),
        'precision': float(precision_score(flat_t, flat_p, zero_division=0)),
        'f1_micro':  float(f1_score(flat_t, flat_p, zero_division=0, average='micro')),
    }
    # Macro average (mean of per-class metrics - same as F1-macro we already report)
    overall['f1_macro'] = float(np.mean(per['f1']))
    overall['accuracy_macro'] = float(np.mean(per['accuracy']))
    overall['recall_macro'] = float(np.mean(per['recall']))
    overall['precision_macro'] = float(np.mean(per['precision']))
    return {'per_compartment': per, 'overall': overall, 'preds': preds.tolist(), 'probs': probs.tolist()}


def load_X(pool, layer):
    if pool == 'attn':
        path, key = ATTN_H5, f'attn_layer_{layer:02d}'
    elif pool == 'mean':
        path, key = MEAN_H5, f'df_adi_layer_{layer:02d}'
    else:
        raise ValueError(pool)
    with h5py.File(path, 'r') as f:
        X = f[key][:].astype(np.float32)
    net_emb = np.load(SPACE_EMB).astype(np.float32)
    net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    return np.concatenate([X, net_filled], axis=1).astype(np.float32)


def run_one(pool, layer, cutoff, k, X_all, Y_all, parts):
    t = time.time()
    tr_mask = (parts != k); te_mask = (parts == k)
    X_tr, X_te = X_all[tr_mask], X_all[te_mask]
    Y_tr, Y_te = Y_all[tr_mask], Y_all[te_mask]
    n_tr0 = int(len(Y_tr))
    oof_r1 = gen_oof(X_tr, Y_tr)
    keep_r1, scores_r1 = cleanlab_step(Y_tr, oof_r1, cutoff)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2, scores_r2 = cleanlab_step(Y_r1, oof_r2, cutoff)
    X_r2 = X_r1[keep_r2]; Y_r2 = Y_r1[keep_r2]
    probs = train_mlp(X_r2, Y_r2, X_te)
    metrics = evaluate_full(probs, Y_te)
    return {
        'config': f'{pool}_L{layer:02d}+SPACE+clean@cutoff={cutoff:.2f} on partition {k}',
        'pool': pool, 'layer': layer, 'cutoff': cutoff, 'fold': k,
        'n_train': n_tr0, 'n_after_r1': int(keep_r1.sum()), 'n_after_r2': int(keep_r2.sum()),
        'n_test': int(len(Y_te)),
        'wall_s': round(time.time() - t, 1),
        'metrics': metrics,
    }


# ============================== MAIN ==============================
def main():
    t0 = time.time()
    print("=" * 65)
    print("  TINKER11 - partition-4 metric diagnostic")
    print("    accuracy + recall + precision + F1 per compartment & overall")
    print("=" * 65)

    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    print(f"  Labels: {Y_all.shape}  Partitions: {np.bincount(parts).tolist()}")
    print(f"  target fold = {K_TARGET}  (partition {K_TARGET} has {int((parts==K_TARGET).sum())} test rows)\n")

    # Configs to compute. Two folds ARE partition 4.
    # (Reference scores from tinker9/tinker10 are noted for sanity check.)
    configs = [
        # (label, pool, layer, cutoff)
        ("A_attn_L22_c050_p4",    "attn", 22, 0.50),  # wrote 0.8011 in tinker9
        ("B_mean_L22_c030_p4",    "mean", 22, 0.30),  # wrote 0.7926 in tinker10
    ]

    results = []
    for label, pool, layer, cutoff in configs:
        print(f"\n  >>> {label}  ({pool}, L22, cutoff={cutoff}, partition {K_TARGET})")
        X = load_X(pool, layer)
        r = run_one(pool, layer, cutoff, K_TARGET, X, Y_all, parts)
        r['label'] = label
        results.append(r)
        m = r['metrics']
        print(f"     n_train {r['n_train']} → R1 {r['n_after_r1']} → R2 {r['n_after_r2']} (test n={r['n_test']}, wall {r['wall_s']}s)")
        print(f"     OVERALL (micro):  accuracy={m['overall']['accuracy']:.4f}  recall={m['overall']['recall']:.4f}  "
              f"precision={m['overall']['precision']:.4f}  F1_micro={m['overall']['f1_micro']:.4f}  F1_macro={m['overall']['f1_macro']:.4f}")

    # Save
    out_json = OUT_DIR / "tinker11_diag_p4.json"
    out_json.write_text(json.dumps({'protocol': 'partition-4 only; full metric set; '
                                              'matches tinker9/10 protocol verbatim.',
                                     'wall_time_sec': round(time.time() - t0, 1),
                                     'results': results}, indent=2, default=str))

    PRINT = []
    PRINT.append('### TINKER11 - partition-4 metric diagnostic ###')
    PRINT.append('')
    PRINT.append(f'  Two configs on partition {K_TARGET} (3,276 held-out rows):')
    PRINT.append('')
    for r in results:
        m = r['metrics']
        PRINT.append(f'  ## {r["label"]} - {r["config"]}')
        PRINT.append(f'  train n: {r["n_train"]} -> R1: {r["n_after_r1"]} -> R2: {r["n_after_r2"]}')
        PRINT.append(f'  test  n: {r["n_test"]}    wall: {r["wall_s"]}s')
        PRINT.append('')
        PRINT.append(f'  OVERALL (micro over 7 compartments × {r["n_test"]} rows = {r["n_test"]*7} samples):')
        PRINT.append(f'    accuracy  : {m["overall"]["accuracy"]:.4f}')
        PRINT.append(f'    recall    : {m["overall"]["recall"]:.4f}   (= TP / (TP+FN), the fraction of true positives we caught)')
        PRINT.append(f'    precision : {m["overall"]["precision"]:.4f}  (= TP / (TP+FP), the fraction of predicted positives that are real)')
        PRINT.append(f'    F1_micro  : {m["overall"]["f1_micro"]:.4f}')
        PRINT.append(f'    F1_macro  : {m["overall"]["f1_macro"]:.4f}   ← this is the score we already reported (per-compartment mean)')
        PRINT.append('')
        PRINT.append(f'  MACRO (mean over the 7 compartments):')
        PRINT.append(f'    accuracy  : {m["overall"]["accuracy_macro"]:.4f}')
        PRINT.append(f'    recall    : {m["overall"]["recall_macro"]:.4f}')
        PRINT.append(f'    precision : {m["overall"]["precision_macro"]:.4f}')
        PRINT.append(f'    F1        : {m["overall"]["f1_macro"]:.4f}')
        PRINT.append('')
        PRINT.append(f'  PER-COMPARTMENT BREAKDOWN:')
        PRINT.append(f'  {"compartment":>15}  {"accuracy":>9}  {"recall":>9}  {"precisn":>9}  {"F1":>9}')
        for j, c in enumerate(COMPARTMENTS):
            PRINT.append(f'  {c:>15}  {m["per_compartment"]["accuracy"][j]:>9.4f}  '
                         f'{m["per_compartment"]["recall"][j]:>9.4f}  '
                         f'{m["per_compartment"]["precision"][j]:>9.4f}  '
                         f'{m["per_compartment"]["f1"][j]:>9.4f}')
        PRINT.append('')

    (OUT_DIR / 'tinker11_diag_p4.txt').write_text('\n'.join(PRINT))
    print('Saved:', out_json)
    print('\n' + '\n'.join(PRINT))
    print(f'\nTotal wall time: {time.time() - t0:.1f}s')


if __name__ == "__main__":
    main()
