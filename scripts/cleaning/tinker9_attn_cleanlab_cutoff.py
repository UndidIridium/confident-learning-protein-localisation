#!/usr/bin/env python3
"""tinker9_attn_cleanlab_cutoff.py

Sweep cleanlab self_confidence cutoff ∈ {0.30, 0.35, 0.40, 0.50, 0.60}
for attn L22 + SPACE in 5-fold CV.

Goal: determine whether a different cutoff value closes the −0.0065 gap
with mean + cleanlab (0.7672 vs attn + cleanlab = 0.7607) — i.e. is the
gap a calibration issue (which retuning can fix) or a structural issue
(which it can't)?

Protocol (verbatim from champion_5fold_cv.py / tinker8):
- Holdout by df_adi.partition, 5 folds.
- X = [attn_layer_22 || SPACE] (1536-d).
- 2 rounds of cleanlab with the SAME cutoff value (R1 applies, R2 applies again).
- Final MLP trained on cleaned data, t=0.5 fixed.
- StandardScaler, pos_weight clipped [1,20], Adam 1e-4, dropout 0.5.

Wall time estimate: ~25-35 min (5 cutoffs × 5 folds × 9 trainings/fold ≈ 225 trainings;
empirical rate from tinker8_four_conditions.py was ~7 s/training).
"""
import csv, json, os, time, warnings
from pathlib import Path
import h5py, numpy as np, pandas as pd

import torch, torch.nn as nn, torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cleanlab.multilabel_classification.rank import get_label_quality_scores

os.environ["OMP_NUM_THREADS"] = "4"
warnings.filterwarnings("ignore")

# -------------- paths --------------
PROJ = Path("/Volumes/BOMBOCLAT/project_JL").resolve()
SRC_CSV = PROJ / "data" / "df_adi.csv"
ATTN_H5 = PROJ / "data" / "prott5_attn_all_layers.h5"
SPACE_EMB = PROJ / "data" / "space_network_embeddings.npy"
SPACE_MASK = PROJ / "data" / "space_network_mask.npy"
OUT_DIR = PROJ / "output_tinker9_attn_cleanlab_cutoff"
OUT_DIR.mkdir(exist_ok=True)

# -------------- constants --------------
HIDDEN = 512; DROPOUT = 0.5; LR = 1e-4
MAX_EP = 50; PATIENCE = 5; BATCH_SIZE = 256; ES_FRAC = 0.10
THR = 0.5

LABEL_COLS = ["membrane","cytoplasm","nucleus","extracellular",
              "cell_surface","mitochondrion","endom"]
M = len(LABEL_COLS)
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]

CUTOFFS = [0.30, 0.35, 0.40, 0.50, 0.60]


# -------------- model (copy of champion) --------------
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


def evaluate(probs, Y):
    preds = (probs >= THR).astype(int)
    pc = [float(f1_score(Y[:, j].astype(int), preds[:, j], zero_division=0))
          for j in range(M)]
    return float(np.mean(pc)), pc


# -------------- data loader (cache X once, reuse across cutoffs) --------------
def load_X_attn_L22_SPACE():
    with h5py.File(ATTN_H5, 'r') as f:
        X = f["attn_layer_22"][:].astype(np.float32)
    net_emb = np.load(SPACE_EMB).astype(np.float32)
    net_mask = np.load(SPACE_MASK)
    net_filled = net_emb.copy(); net_filled[~net_mask] = 0.0
    X = np.concatenate([X, net_filled], axis=1).astype(np.float32)
    return X


# -------------- run a single (cutoff, fold) --------------
def run_one(cutoff, k, X_all, Y_all, parts):
    t = time.time()
    tr_mask = (parts != k); te_mask = (parts == k)
    X_tr, X_te = X_all[tr_mask], X_all[te_mask]
    Y_tr, Y_te = Y_all[tr_mask], Y_all[te_mask]
    n_tr0 = int(len(Y_tr))
    # R1
    oof_r1 = gen_oof(X_tr, Y_tr)
    keep_r1, scores_r1 = cleanlab_step(Y_tr, oof_r1, cutoff)
    X_r1, Y_r1 = X_tr[keep_r1], Y_tr[keep_r1]
    # R2
    oof_r2 = gen_oof(X_r1, Y_r1)
    keep_r2, scores_r2 = cleanlab_step(Y_r1, oof_r2, cutoff)
    X_r2 = X_r1[keep_r2]; Y_r2 = Y_r1[keep_r2]
    # Final
    probs = train_mlp(X_r2, Y_r2, X_te)
    f1, pc = evaluate(probs, Y_te)
    # Diagnostics: how aggressive was each step?
    avg_score_kept_r1 = float(scores_r1[keep_r1].mean()) if keep_r1.any() else 0.0
    avg_score_dropped_r1 = float(scores_r1[~keep_r1].mean()) if (~keep_r1).any() else 0.0
    return {
        'holdout': int(k), 'cutoff': float(cutoff),
        'macro_f1': float(f1), 'per_class': pc,
        'n_train': n_tr0, 'n_after_r1': int(keep_r1.sum()), 'n_after_r2': int(keep_r2.sum()),
        'drop_rate_r1_pct': 100 * (1 - keep_r1.mean()),
        'avg_score_kept_r1': avg_score_kept_r1,
        'avg_score_dropped_r1': avg_score_dropped_r1,
        'n_test': int(len(Y_te)), 'wall_s': round(time.time() - t, 1),
    }


# ============================== MAIN ==============================
def main():
    t0 = time.time()
    print("=" * 65)
    print("  TINKER9 — attn L22 + SPACE × cleanlab cutoff sweep")
    print("=" * 65)
    print(f"  Cutoffs: {CUTOFFS}")

    src = pd.read_csv(SRC_CSV)
    Y_all = src[LABEL_COLS].values.astype(np.int64)
    parts = src["partition"].to_numpy()
    print(f"  Labels: {Y_all.shape}  Partitions: {np.bincount(parts).tolist()}")

    X_all = load_X_attn_L22_SPACE()
    print(f"  X shape: {X_all.shape}  mean={X_all.mean():.2f}  std={X_all.std():.2f}")

    all_results = []
    for c in CUTOFFS:
        print(f"\n  >>> CUTOFF = {c}")
        cfg_results = []
        for k in range(5):
            res = run_one(c, k, X_all, Y_all, parts)
            cfg_results.append(res)
            print(f"     [c={c} fold {k}] F1={res['macro_f1']:.4f}  "
                  f"n={res['n_train']}→{res['n_after_r1']}→{res['n_after_r2']}  "
                  f"drop_r1={res['drop_rate_r1_pct']:.1f}%  ({res['wall_s']:.1f}s)")
        arr = np.array([r['macro_f1'] for r in cfg_results])
        drop_rates = [r['drop_rate_r1_pct'] for r in cfg_results]
        n_r1 = np.mean([r['n_after_r1'] for r in cfg_results])
        n_r2 = np.mean([r['n_after_r2'] for r in cfg_results])
        summary = {
            'cutoff': float(c),
            'mean_f1': float(arr.mean()),
            'std_f1':  float(arr.std(ddof=0)),
            'per_fold_f1': [float(x) for x in arr],
            'mean_drop_r1_pct': float(np.mean(drop_rates)),
            'mean_n_after_r1': float(n_r1),
            'mean_n_after_r2': float(n_r2),
            'per_fold': cfg_results,
        }
        all_results.append(summary)
        print(f"     [SUM c={c}]  mean={arr.mean():.4f} ± {arr.std(ddof=0):.4f}  "
              f"drop_r1≈{np.mean(drop_rates):.1f}%  R1→R2 n≈{n_r1:.0f}→{n_r2:.0f}")

    # ---- save ----
    summary_csv = OUT_DIR / "tinker9_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cutoff", "mean_f1", "std_f1",
                    "mean_drop_r1_pct", "mean_n_after_r1", "mean_n_after_r2"])
        for s in all_results:
            w.writerow([f"{s['cutoff']:.2f}", f"{s['mean_f1']:.4f}", f"{s['std_f1']:.4f}",
                        f"{s['mean_drop_r1_pct']:.2f}", f"{s['mean_n_after_r1']:.0f}",
                        f"{s['mean_n_after_r2']:.0f}"])

    out_json = OUT_DIR / "tinker9_report.json"
    out_json.write_text(json.dumps({
        'protocol': ('5-fold partition-aware CV; attn_L22+SPACE X (1536-d); '
                     'iterative cleanlab with self_confidence method; cutoff sweep '
                     '{0.30,0.35,0.40,0.50,0.60}; t=0.5 fixed.'),
        'wall_time_sec': round(time.time() - t0, 1),
        'reference_mean_L22_cleanlab': 0.7672,  # from layer_cleanlab_sweep.csv
        'reference_attn_L22_no_clean': 0.7680,   # from tinker8 5-fold CV
        'reference_attn_L22_c04_clean': 0.7835,  # from tinker7 (also affected by SPACE)
        'note': ('At cutoff=0.40, this sweep should reproduce tinker7 B_attn_L22 = '
                 '0.7835 ± 0.0145 (sanity check on protocol). The -0.0065 gap with mean '
                 'is at the no-SPACE +cleanlab cell of the 8-cell matrix; this sweep is on '
                 'the +SPACE cell which already wins by +0.0145.'),
        'cutoffs': all_results,
    }, indent=2, default=str))

    # ---- headline ----
    print("\n" + "=" * 65)
    print("  HEADLINE — cutoff sweep (attn L22 + SPACE)")
    print("=" * 65)
    print(f"  Reference: mean L22 + cleanlab = 0.7672  (the wall to beat)")
    print(f"  Reference: attn L22 + SPACE noClean = 0.7680  (tinker8)")
    print(f"  Reference: attn L22 + SPACE + clean@0.40 = 0.7835  (tinker7; for reference)")
    print()
    print(f"  {'cutoff':>8}  {'mean_f1':>9}  {'std_f1':>9}  {'drop_r1%':>9}  {'n_R1':>6}  {'n_R2':>6}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*6}")
    # Insert 'no cleanlab' as the floor reference
    print(f"  {'(no CL)':>8}  {0.7680:>9.4f}  {'   -- ':>9}  {'   -- ':>9}  {'   -- ':>6}  {'   -- ':>6}")
    for s in all_results:
        print(f"  {s['cutoff']:>8.2f}  {s['mean_f1']:>9.4f}  {s['std_f1']:>9.4f}  "
              f"{s['mean_drop_r1_pct']:>9.2f}  {s['mean_n_after_r1']:>6.0f}  "
              f"{s['mean_n_after_r2']:>6.0f}")

    # Best cutoff
    best_idx = int(np.argmax([s['mean_f1'] for s in all_results]))
    best = all_results[best_idx]
    best_no_clean = 0.7680
    print(f"\n  >>> BEST CUTOFF = {best['cutoff']}  →  {best['mean_f1']:.4f} ± {best['std_f1']:.4f}")
    print(f"      vs no-cleanlab (+0={best['mean_f1']-best_no_clean:+.4f})")
    print(f"      vs mean L22 + cleanlab (target 0.7672, gap =  {best['mean_f1']-0.7672:+.4f})")

    print(f"\n  Total wall time: {time.time() - t0:.1f}s")
    print(f"  Saved: {summary_csv}  /  {out_json}")


if __name__ == "__main__":
    main()
