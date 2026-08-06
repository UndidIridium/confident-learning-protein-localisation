#!/usr/bin/env python3
"""compare_p4_vs_deeploc.py

Apples-to-apples comparison: DeepLoc 2.1 (Accurate / ProtT5-XL) vs our
champion pipeline on df_adi partition 4 (3,276 proteins, 7 binary labels).

Inputs:
  - /Users/aditya/Downloads/results_partition4.csv
      DeepLoc 2.1 Accurate output (18 cols; Protein_ID, Localizations, Signals,
      Membrane types, + 14 probability columns)
  - output_champion_5fold_cv/test_probs_p4.npy  (3276, 7) float32
  - output_champion_5fold_cv/test_labels_p4.npy (3276, 7) int
  - data/df_adi.csv  ground truth  (filter to partition==4)

Outputs (in output_comparison_deeploc_accurate_vs_ours/):
  - headline_table.md            one-page table (5 metrics × 2 methods + Δ)
  - per_compartment.csv          raw per-compartment breakdown
  - per_compartment.md           same in markdown
  - fig1_f1_per_compartment.png
  - fig2_precision_recall_bars.png
  - fig3_score_distributions.png
  - fig4_win_loss_heatmap.png
  - comparison_metrics.json

Usage:
  python3 compare_p4_vs_deeploc.py
"""

import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, hamming_loss,
)

import eval_against_deeploc_2_1 as eval_dl  # reuse existing mapping helpers

PROJ = Path(__file__).parent.resolve()
SRC_CSV    = PROJ / "data" / "df_adi.csv"
DL_CSV     = Path("/Users/aditya/Downloads/results_partition4.csv")
OUR_PROBS  = PROJ / "output_champion_5fold_cv" / "test_probs_p4.npy"
OUR_LBLS   = PROJ / "output_champion_5fold_cv" / "test_labels_p4.npy"
OUT_DIR    = PROJ / "output_comparison_deeploc_accurate_vs_ours"
OUT_DIR.mkdir(exist_ok=True)

LABELS_7 = [
    "membrane","cytoplasm","nucleus","extracellular",
    "cell_surface","mitochondrion","endom",
]
COMPARTMENTS = ["Membrane","Cytoplasm","Nucleus","Extracellular",
                "Cell_surf","Mito","Endom"]
MAPPING = {
    "Membrane":      "derived",   # max(TM, Peri, LA)
    "Cytoplasm":     "direct",
    "Nucleus":       "direct",
    "Extracellular": "direct",
    "Cell_surf":     "direct",
    "Mito":          "direct",
    "Endom":         "direct",
}
THRESHOLD = 0.5

# matplotlib polish ----------------------------------------------------
rcParams["font.family"] = "DejaVu Sans"
rcParams["axes.spines.top"]   = False
rcParams["axes.spines.right"] = False
rcParams["axes.grid"]         = True
rcParams["grid.alpha"]        = 0.25
rcParams["axes.titleweight"]  = "bold"
rcParams["savefig.bbox"]      = "tight"

# Color-blind-safe palette: teal = ours, red = DeepLoc
OURS_COLOR = "#0e7490"   # teal
DL_COLOR   = "#b91c1c"   # red
NEUTRAL    = "#444444"


def load_dl_predictions(dl_csv: Path):
    """Load DeepLoc CSV and map to 7-compartment binary predictions.

    Reuses `eval_dl.map_dl_to_ours` so the mapping is identical to the
    apples-to-apples comparison script that already lives in the codebase.
    """
    rows, hdr = eval_dl.load_deeploc_csv(str(dl_csv))
    preds = np.array(
        [eval_dl.map_dl_to_ours(r, threshold=THRESHOLD) for r in rows],
        dtype=int,
    )
    return rows, preds


def load_our_predictions(probs_path: Path, labels_path: Path):
    """Load cached champion P4 probs; threshold to binary predictions."""
    probs  = np.load(probs_path).astype(np.float32)
    labels = np.load(labels_path).astype(int)
    preds  = (probs >= THRESHOLD).astype(int)
    # sanity: lengths match
    assert probs.shape == labels.shape == (len(probs), 7), \
        f"shape mismatch probs={probs.shape} labels={labels.shape}"
    return probs, labels, preds


def per_compartment_metrics(Y_true, Y_pred):
    per = {"accuracy":[], "recall":[], "precision":[], "f1":[]}
    for j in range(7):
        yt, yp = Y_true[:, j], Y_pred[:, j]
        per["accuracy"].append(float(accuracy_score(yt, yp)))
        per["recall"].append(float(recall_score(yt, yp, zero_division=0)))
        per["precision"].append(float(precision_score(yt, yp, zero_division=0)))
        per["f1"].append(float(f1_score(yt, yp, zero_division=0)))
    flat_t, flat_p = Y_true.flatten(), Y_pred.flatten()
    overall = {
        "accuracy":         float(accuracy_score(flat_t, flat_p)),
        "recall":           float(recall_score(flat_t, flat_p, zero_division=0)),
        "precision":        float(precision_score(flat_t, flat_p, zero_division=0)),
        "f1_micro":         float(f1_score(flat_t, flat_p, average="micro", zero_division=0)),
        "f1_macro":         float(np.mean(per["f1"])),
        "accuracy_macro":   float(np.mean(per["accuracy"])),
        "recall_macro":     float(np.mean(per["recall"])),
        "precision_macro":  float(np.mean(per["precision"])),
        "hamming_loss":     float(hamming_loss(flat_t, flat_p)),
        "n_proteins":       int(len(Y_true)),
    }
    return per, overall


def fig1_f1_bars(per_ours, per_dl):
    x = np.arange(len(COMPARTMENTS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, per_dl["f1"],    w, color=DL_COLOR,   label="DeepLoc 2.1 (Accurate / ProtT5-XL)")
    ax.bar(x + w/2, per_ours["f1"],  w, color=OURS_COLOR, label="Ours (champion, attn-pool L22 + SPACE + aux)")
    for j in range(len(COMPARTMENTS)):
        d = per_ours["f1"][j] - per_dl["f1"][j]
        ax.text(x[j], max(per_ours["f1"][j], per_dl["f1"][j]) + 0.012,
                f"Δ {d:+.3f}", ha="center", fontsize=8,
                color=OURS_COLOR if d > 0 else DL_COLOR)
    ax.set_xticks(x); ax.set_xticklabels(COMPARTMENTS, rotation=15, ha="right")
    ax.set_ylabel("F1 score"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-compartment F1 - partition 4 holdout (3,276 proteins)")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_f1_per_compartment.png", dpi=300)
    plt.close(fig)


def fig2_pr_bars(per_ours, per_dl):
    x = np.arange(len(COMPARTMENTS))
    w = 0.20
    fig, ax = plt.subplots(figsize=(11, 5.5))
    # ours - precision + recall
    ax.bar(x - 1.5*w, per_ours["precision"], w, color=OURS_COLOR, alpha=0.85,
           label="Ours - precision")
    ax.bar(x - 0.5*w, per_ours["recall"],    w, color=OURS_COLOR, alpha=0.55,
           label="Ours - recall")
    # DeepLoc - precision + recall
    ax.bar(x + 0.5*w, per_dl["precision"],   w, color=DL_COLOR,   alpha=0.85,
           label="DeepLoc - precision")
    ax.bar(x + 1.5*w, per_dl["recall"],      w, color=DL_COLOR,   alpha=0.55,
           label="DeepLoc - recall")
    ax.set_xticks(x); ax.set_xticklabels(COMPARTMENTS, rotation=15, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Per-compartment Precision & Recall - partition 4 holdout (3,276 proteins)")
    ax.axhline(0.5, color=NEUTRAL, lw=0.6, ls="--")
    ax.legend(loc="lower right", ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_precision_recall_bars.png", dpi=300)
    plt.close(fig)


def fig3_score_distributions(probs_ours, probs_dl_rows, Y_true):
    """6 separate panels (skip Membrane - derived mapping makes comparison apples-to-oranges).
    Each panel: histogram of predicted probability on positives vs negatives
    for both methods.
    """
    # Build DL probability vectors aligned to ground truth ordering
    # (eval_dl returns rows as dicts; we need raw probabilities for each compartment)
    dl_probs_matrix = np.zeros((len(probs_dl_rows), 7), dtype=np.float32)
    for i, r in enumerate(probs_dl_rows):
        # Membrane = max(TM, Peri, LA)
        dl_probs_matrix[i, 0] = max(eval_dl._resolve(r, "Transmembrane"),
                                   eval_dl._resolve(r, "Peripheral"),
                                   eval_dl._resolve(r, "Lipid.anchored"))
        dl_probs_matrix[i, 1] = eval_dl._resolve(r, "Cytoplasm")
        dl_probs_matrix[i, 2] = eval_dl._resolve(r, "Nucleus")
        dl_probs_matrix[i, 3] = eval_dl._resolve(r, "Extracellular")
        dl_probs_matrix[i, 4] = eval_dl._resolve(r, "Cell.membrane")
        dl_probs_matrix[i, 5] = eval_dl._resolve(r, "Mitochondrion")
        dl_probs_matrix[i, 6] = eval_dl._resolve(r, "Endoplasmic.reticulum")

    # Skip Membrane panel (derived aggregation changes its probability semantics)
    panels = COMPARTMENTS[1:]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.flatten()
    bins = np.linspace(0, 1, 21)
    for j, comp in enumerate(panels):
        ax = axes[j]
        yt = Y_true[:, j+1]
        # DeepLoc - positives
        ax.hist(dl_probs_matrix[yt==1, j+1], bins=bins,
                color=DL_COLOR, alpha=0.55, label="DeepLoc - true positive",
                density=True, edgecolor="white", linewidth=0.4)
        ax.hist(dl_probs_matrix[yt==0, j+1], bins=bins,
                color=DL_COLOR, alpha=0.20, label="DeepLoc - true negative",
                density=True, edgecolor="white", linewidth=0.4, hatch="//")
        # Ours - positives
        ax.hist(probs_ours[yt==1, j+1], bins=bins,
                color=OURS_COLOR, alpha=0.55, label="Ours - true positive",
                density=True, edgecolor="white", linewidth=0.4)
        ax.hist(probs_ours[yt==0, j+1], bins=bins,
                color=OURS_COLOR, alpha=0.20, label="Ours - true negative",
                density=True, edgecolor="white", linewidth=0.4, hatch="\\\\")
        ax.axvline(THRESHOLD, color=NEUTRAL, lw=0.8, ls="--")
        ax.set_title(comp, fontsize=11)
        ax.set_xlabel("predicted probability")
        if j % 3 == 0:
            ax.set_ylabel("density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle(
        "Score distributions of true positives vs negatives - "
        "DeepLoc 2.1 Accurate ‖ ours\n"
        "(Membrane omitted: derived mapping makes raw probability "
        "distributions non-comparable)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT_DIR / "fig3_score_distributions.png", dpi=300)
    plt.close(fig)


def fig4_win_loss_heatmap(per_ours, per_dl):
    delta = np.array(per_ours["f1"]) - np.array(per_dl["f1"])
    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "ours_dl", [DL_COLOR, "#f3f4f6", OURS_COLOR])
    vmax = max(0.05, abs(delta).max() + 0.01)
    im = ax.imshow(delta.reshape(1, -1), cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="auto")
    for j, comp in enumerate(COMPARTMENTS):
        d = delta[j]
        color = "white" if abs(d) > vmax * 0.55 else NEUTRAL
        ax.text(j, 0, f"{d:+.3f}", ha="center", va="center",
                color=color, fontweight="bold")
    ax.set_xticks(range(len(COMPARTMENTS)))
    ax.set_xticklabels([f"{c}\n({MAPPING[c]})" for c in COMPARTMENTS],
                       rotation=0, ha="center", fontsize=9)
    ax.set_yticks([])
    ax.set_title("F1 delta per compartment (ours − DeepLoc 2.1 Accurate)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Δ F1 (teal = ours wins, red = DeepLoc wins)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_win_loss_heatmap.png", dpi=300)
    plt.close(fig)


def write_outputs(per_ours, per_dl, ov_ours, ov_dl):
    # JSON
    json.dump(
        {
            "ours":     {"per_compartment": per_ours, "overall": ov_ours},
            "deeploc":  {"per_compartment": per_dl,   "overall": ov_dl},
            "mapping":  MAPPING,
            "threshold": THRESHOLD,
            "n_proteins": ov_ours["n_proteins"],
        },
        open(OUT_DIR / "comparison_metrics.json", "w"),
        indent=2,
    )

    # Per-compartment CSV
    with open(OUT_DIR / "per_compartment.csv", "w") as f:
        f.write("compartment,mapping,dl_f1,dl_precision,dl_recall,"
                "ours_f1,ours_precision,ours_recall,delta_f1,winner\n")
        for j, c in enumerate(COMPARTMENTS):
            d_f1  = per_ours["f1"][j] - per_dl["f1"][j]
            winner = "OURS" if d_f1 > 0 else ("DEEPLOC" if d_f1 < 0 else "TIE")
            f.write(f"{c},{MAPPING[c]},"
                    f"{per_dl['f1'][j]:.4f},{per_dl['precision'][j]:.4f},"
                    f"{per_dl['recall'][j]:.4f},"
                    f"{per_ours['f1'][j]:.4f},{per_ours['precision'][j]:.4f},"
                    f"{per_ours['recall'][j]:.4f},"
                    f"{d_f1:+.4f},{winner}\n")

    # Per-compartment Markdown
    md_lines = [
        "## Per-compartment F1 / precision / recall",
        "",
        "| Compartment | Mapping | DL F1 | DL P | DL R | Ours F1 | Ours P | Ours R | Δ F1 | Winner |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for j, c in enumerate(COMPARTMENTS):
        d_f1 = per_ours["f1"][j] - per_dl["f1"][j]
        winner = " OURS" if d_f1 > 0 else ("DeepLoc" if d_f1 < 0 else "-")
        star = "  (Δ>0.05)" if abs(d_f1) > 0.05 else ""
        md_lines.append(
            f"| {c} | {MAPPING[c]} | {per_dl['f1'][j]:.4f} | "
            f"{per_dl['precision'][j]:.4f} | {per_dl['recall'][j]:.4f} | "
            f"{per_ours['f1'][j]:.4f} | {per_ours['precision'][j]:.4f} | "
            f"{per_ours['recall'][j]:.4f} | {d_f1:+.4f} | {winner}{star} |"
        )
    md_lines += ["",
                 "**Mapping legend:** ‘direct’ = 1-to-1 from a single DL prob column. "
                 "**‘derived’** = Membrane = max(Transmembrane, Peripheral, Lipid-anchored) > t.",
                 ""]
    (OUT_DIR / "per_compartment.md").write_text("\n".join(md_lines))

    # Headline table
    hd = ["# Headline: DeepLoc 2.1 (Accurate / ProtT5-XL) vs Ours (attn-pool L22 + SPACE + aux)",
          "",
          f"**Task:** df_adi partition 4 holdout, 3,276 proteins, 7 binary labels.",
          f"**Threshold:** t = 0.50 for both methods. "
          "Direct-mapped compartments (Cytoplasm, Nucleus, Extracellular, Cell_surf, Mito, Endom) "
          "are an apples-to-apples comparison. Membrane is *derived* (max of three DL probs).",
          "",
          "## Overall metrics (micro / macro over 7 compartments × 3,276 = 22,932 binary samples)",
          "",
          "| Metric | DeepLoc 2.1 (Accurate) | Ours (Champion) | Δ (Ours − DL) |",
          "|---|---:|---:|---:|"]
    metric_pairs = [
        ("F1 (macro) - headline",   ov_dl["f1_macro"],         ov_ours["f1_macro"]),
        ("F1 (micro)",              ov_dl["f1_micro"],         ov_ours["f1_micro"]),
        ("Accuracy (micro)",        ov_dl["accuracy"],         ov_ours["accuracy"]),
        ("Precision (macro)",       ov_dl["precision_macro"],  ov_ours["precision_macro"]),
        ("Recall (macro)",          ov_dl["recall_macro"],     ov_ours["recall_macro"]),
        ("Hamming loss ↓",          ov_dl["hamming_loss"],     ov_ours["hamming_loss"]),
    ]
    for name, dv, ov in metric_pairs:
        delta = ov - dv
        sign  = " (better ↓)" if "loss" in name else ""
        hd.append(f"| {name}{sign} | {dv:.4f} | {ov:.4f} | {delta:+.4f} |")
    hd += ["",
           "## Per-compartment F1 (full breakdown)",
           ""]
    for j, c in enumerate(COMPARTMENTS):
        d_f1 = per_ours["f1"][j] - per_dl["f1"][j]
        verdict = "OURS wins" if d_f1 > 0 else ("DeepLoc wins" if d_f1 < 0 else "tie")
        hd.append(f"- **{c}** ({MAPPING[c]}):  DL {per_dl['f1'][j]:.4f}  ‖  "
                  f"Ours {per_ours['f1'][j]:.4f}  →  Δ {d_f1:+.4f}  ({verdict})")
    hd += ["",
           "## Artifacts in this folder",
           "",
           "- `headline_table.md` (this file)",
           "- `per_compartment.csv` / `.md`  raw breakdown",
           "- `fig1_f1_per_compartment.png`  per-compartment F1 side-by-side bars",
           "- `fig2_precision_recall_bars.png`  precision + recall grouped bars",
           "- `fig3_score_distributions.png`  score histograms per compartment (pos vs neg)",
           "- `fig4_win_loss_heatmap.png`  Δ F1 heatmap across compartments",
           "- `comparison_metrics.json`  full numerical output",
           ""]
    (OUT_DIR / "headline_table.md").write_text("\n".join(hd))


def main():
    print("=" * 72)
    print("  COMPARISON - DeepLoc 2.1 (Accurate / ProtT5-XL) vs Our Champion")
    print("  df_adi partition 4 - apples-to-apples (same GT, same threshold t=0.5)")
    print("=" * 72)

    # 1. Ground truth - load from df_adi partition-4
    src = pd.read_csv(SRC_CSV)
    p4 = src[src["partition"] == 4].reset_index(drop=True)
    Y_true = p4[LABELS_7].values.astype(int)
    n = len(p4)
    print(f"\nGround truth loaded:  {n} proteins × 7 binary labels (from df_adi.csv)")

    # 2. Our P4 probs
    probs_ours, Y_ours_lbl, preds_ours = load_our_predictions(OUR_PROBS, OUR_LBLS)
    assert probs_ours.shape == (n, 7), \
        f"OURS probs shape {probs_ours.shape} != ({n}, 7)"
    assert (Y_ours_lbl == Y_true).all(), \
        "OURS labels don't match df_adi ground truth - check partition filter"
    print(f"Our champion P4 probs:    shape {probs_ours.shape} (from {OUR_PROBS.name})")

    # 3. DeepLoc preds
    dl_rows, preds_dl = load_dl_predictions(DL_CSV)
    assert preds_dl.shape == (n, 7), \
        f"DL preds shape {preds_dl.shape} != ({n}, 7)"
    # ID alignment sanity
    dl_ids = pd.read_csv(DL_CSV).iloc[:, 0].astype(str).values
    p4_ids = p4["acc"].astype(str).values
    pos_match = int((dl_ids == p4_ids).sum())
    print(f"DeepLoc Accurate preds:   shape {preds_dl.shape} (from {DL_CSV.name})")
    print(f"ID alignment check:      {pos_match}/{n} positional matches in accession order")
    assert pos_match == n, (
        f"DeepLoc CSV is not in the same accession order as df_adi partition 4 "
        f"(only {pos_match}/{n} matches). Re-align by accession before evaluation."
    )

    # 4. Metrics
    per_ours, ov_ours = per_compartment_metrics(Y_true, preds_ours)
    per_dl,   ov_dl   = per_compartment_metrics(Y_true, preds_dl)
    print(f"\nDeepLoc 2.1 Accurate:   F1_macro={ov_dl['f1_macro']:.4f}  F1_micro={ov_dl['f1_micro']:.4f}  acc={ov_dl['accuracy']:.4f}")
    print(f"Ours (champion):        F1_macro={ov_ours['f1_macro']:.4f}  F1_micro={ov_ours['f1_micro']:.4f}  acc={ov_ours['accuracy']:.4f}")
    print(f"ΔF1_macro: {ov_ours['f1_macro']-ov_dl['f1_macro']:+.4f}")

    # 5. Figures
    print("\nRendering figures...")
    fig1_f1_bars(per_ours, per_dl)
    fig2_pr_bars(per_ours, per_dl)
    fig3_score_distributions(probs_ours, dl_rows, Y_true)
    fig4_win_loss_heatmap(per_ours, per_dl)
    print(f"  → {OUT_DIR}/fig[1-4]_*.png")

    # 6. Outputs
    write_outputs(per_ours, per_dl, ov_ours, ov_dl)
    print(f"  → {OUT_DIR}/headline_table.md")
    print(f"  → {OUT_DIR}/per_compartment.{{csv,md}}")
    print(f"  → {OUT_DIR}/comparison_metrics.json")

    # 7. Per-compartment wins/losses summary
    wins_ours  = sum(1 for j in range(7) if per_ours["f1"][j] > per_dl["f1"][j])
    wins_dl    = 7 - wins_ours
    print(f"\nPer-compartment head-to-head: ours wins {wins_ours}/7, DeepLoc wins {wins_dl}/7.")


if __name__ == "__main__":
    main()
