#!/usr/bin/env python3
"""make_arch_figure.py

Generate a publication-quality matplotlib figure comparing:

  - DeepLoc 2.1: 3-stage branching (localization + membrane-bound + sorting-signal)
  - Ours: flat MLP + cleanlab + STRING (SPACE) features

Arrows highlight which architecture component contributes to which compartment's
performance lift. Useful for the dissertation architecture-comparison chapter.

Run:
  python3 make_arch_figure.py
  -> figures/arch_comparison.png  (1600x900 px, ~150 dpi print)
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

OUT = Path("/Volumes/BOMBOCLAT/project_JL/figures/arch_comparison.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


# ─── Colours (colour-blind-friendly palette) ──────────────────────────────────
COL_BG       = "#f7f9fb"
COL_DEEPLOC  = "#dbe9f4"     # light blue  - DeepLoc panels
COL_OURS     = "#fbe5cf"     # light orange - Ours panels
COL_TEXT     = "#1f2937"
COL_ARROW_US = "#d97706"     # orange - arrows from "Ours" components
COL_ARROW_DL = "#1d4ed8"     # blue   - arrows from "DeepLoc" components
COL_WIN      = "#0e7490"     # teal   - compartments where ours wins (deuteranopia-safe)
COL_LOSE     = "#b91c1c"     # red    - compartments where DeepLoc wins (kept for contrast against teal)
COL_NEUTRAL  = "#6b7280"     # gray   - neutral / tie


# ─── Layout helpers ───────────────────────────────────────────────────────────
def panel(ax, x, y, w, h, label, sublabel=None, color=COL_OURS,
          edge_color=None, fontsize=11, alpha=1.0):
    """Draw a rounded rectangle with a 2-line label centred inside."""
    edge = edge_color or color
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=color, edgecolor=edge, linewidth=1.4, alpha=alpha,
    )
    ax.add_patch(p)
    ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=COL_TEXT)
    if sublabel:
        ax.text(x + w/2, y + h*0.30, sublabel, ha="center", va="center",
                fontsize=fontsize - 2, color=COL_TEXT, style="italic")


def arrow(ax, x0, y0, x1, y1, color=COL_ARROW_US, lw=1.8, label=None,
          label_pos="mid", curve=0.0):
    """Draw a curved arrow from (x0,y0) to (x1,y1)."""
    connstyle = f"arc3,rad={curve}"
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="->,head_length=8,head_width=5",
        connectionstyle=connstyle, color=color, lw=lw,
        mutation_scale=1.0,
    )
    ax.add_patch(a)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + (0.10 if label_pos == "above" else 0),
                label, ha="center", va="center", fontsize=8.5,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85))


# ─── Figure scaffold ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.set_facecolor(COL_BG)
fig.patch.set_facecolor(COL_BG)
ax.axis("off")

# Title
ax.text(10, 10.5,
        "Architecture Comparison: DeepLoc 2.1 (left) vs Ours (right)",
        ha="center", va="center", fontsize=15, fontweight="bold", color=COL_TEXT)
ax.text(10, 10.1,
        "Arrows = which components contribute to each compartment's F1. "
        "Green = we win. Red = DeepLoc wins. Gray = tie.",
        ha="center", va="center", fontsize=10, color=COL_TEXT, style="italic")

# Column headers
ax.add_patch(FancyBboxPatch((0.4, 9.0), 9.2, 0.7,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            facecolor=COL_DEEPLOC, edgecolor=COL_DEEPLOC,
                            linewidth=0))
ax.text(5.0, 9.35, "DeepLoc 2.1  (Ødum et al., 2024)",
        ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(5.0, 9.10, "3-stage branching architecture  |  ~50M+ params  |  ensemble of multi-task heads",
        ha="center", va="center", fontsize=9, style="italic")

ax.add_patch(FancyBboxPatch((10.4, 9.0), 9.2, 0.7,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            facecolor=COL_OURS, edgecolor=COL_OURS,
                            linewidth=0))
ax.text(15.0, 9.35, "Ours  (this work)",
        ha="center", va="center", fontsize=13, fontweight="bold")
ax.text(15.0, 9.10, "Flat MLP head  |  ~3.1M params  |  ProtT5-XL frozen + SPACE + cleanlab + tuned thresholds",
        ha="center", va="center", fontsize=9, style="italic")


# ════════════════════════════════════════════════════════════════════════════
#  LEFT COLUMN - DeepLoc 2.1
# ════════════════════════════════════════════════════════════════════════════

# Stage 1: Localization branch
panel(ax, 0.6, 7.0, 4.0, 1.3,
      "Stage 1: Localization",
      "10-class sigmoid over\nall subcellular compartments",
      color=COL_DEEPLOC, fontsize=11)

# Stage 2: Membrane-bound aux
panel(ax, 0.6, 5.4, 4.0, 1.3,
      "Stage 2: Membrane-bound",
      "Aux head: 4-way\n(peripheral / 1× TMD / multi-TMD / lipid-anchored)",
      color=COL_DEEPLOC, fontsize=11)

# Stage 3: Sorting-signal aux
panel(ax, 0.6, 3.8, 4.0, 1.3,
      "Stage 3: Sorting-signal",
      "Aux head: signal peptide /\nTMD / GPI / mitochondrial targeting",
      color=COL_DEEPLOC, fontsize=11)

# Backbone
panel(ax, 0.6, 1.6, 4.0, 1.6,
      "ProtT5-XL backbone",
      "3B+ params (Accurate)\n   OR\nESM-1b 650M (Fast)\nFully frozen",
      color=COL_DEEPLOC, fontsize=10)

# Attention pooling (shared)
panel(ax, 5.4, 5.0, 4.0, 1.4,
      "Shared attn-pool head",
      "Per-position learned weights\nfrozen after training",
      color=COL_DEEPLOC, fontsize=10)

# (No sequential inter-stage arrows - DeepLoc stages are PARALLEL heads off shared
# backbone, not a serial pipeline. The vertical feeder arrow on the left edge shows
# backbone → all stages.)

# Curved arrows from stage 1 to attn-pool, and back from attn-pool to stage 2/3
arrow(ax, 4.6, 7.5, 5.4, 6.0, color=COL_ARROW_DL, lw=1.0, curve=-0.3)
arrow(ax, 4.6, 6.0, 5.4, 5.8, color=COL_ARROW_DL, lw=1.0, curve=0.0)

# Input flow: ProtT5 backbone (y=1.6-3.2) → all 3 DeepLoc stages (vertical feeder on left edge)
arrow(ax, 2.6, 3.2, 2.6, 7.0, color=COL_ARROW_DL, lw=1.2,
      label="backbone →", curve=0.0)


# ════════════════════════════════════════════════════════════════════════════
#  RIGHT COLUMN - Ours
# ════════════════════════════════════════════════════════════════════════════

# Cleanlab
panel(ax, 10.6, 7.4, 8.8, 1.0,
      "Iterative cleanlab  (2-pass, self_conf ≥ 0.40)",
      "Drops ~30% noisy multi-label rows from training set",
      color=COL_OURS, fontsize=10)

# ProtT5-XL (frozen)
panel(ax, 10.6, 6.0, 4.0, 1.2,
      "ProtT5-XL L22",
      "Frozen\n1024-d attention-pooled",
      color=COL_OURS, fontsize=10)

# SPACE STRING features
panel(ax, 14.8, 6.0, 4.6, 1.2,
      "SPACE STRING features",
      "512-d protein-protein\ninteraction network\n(STRING-DB)",
      color=COL_OURS, fontsize=10)

# Concat + StandardScaler
panel(ax, 10.6, 4.6, 8.8, 1.0,
      "Concat (1536-d) → StandardScaler → 1-layer MLP",
      "hidden=512, dropout=0.5, lr=1e-4, BCE pos-weighted",
      color=COL_OURS, fontsize=10)

# 6-class output
panel(ax, 10.6, 3.2, 8.8, 1.0,
      "6-way sigmoid  →  per-class threshold tuning",
      "thresholds tuned on OOF F1 grid (linspace 0.10-0.90, step 0.05)",
      color=COL_OURS, fontsize=10)

# Vertical arrows in Ours column
arrow(ax, 12.6, 6.0, 12.6, 5.6, color=COL_ARROW_US, lw=1.4)  # prot5 → concat
arrow(ax, 17.1, 6.0, 17.1, 5.6, color=COL_ARROW_US, lw=1.4)  # space → concat
arrow(ax, 15.0, 4.6, 15.0, 4.2, color=COL_ARROW_US, lw=1.4)  # concat → 6-way
arrow(ax, 15.0, 7.4, 15.0, 7.2, color=COL_ARROW_US, lw=1.4)  # cleanlab feeds back into training (not runtime)


# ════════════════════════════════════════════════════════════════════════════
#  BOTTOM ROW - Compartment labels with arrows pointing to where each is decided
# ════════════════════════════════════════════════════════════════════════════

compartments = [
    # (label, x_position_in_DeepLoc_column_for_arrow_target, color)
    ("cytoplasm",       9.6, COL_WIN),
    ("nucleus",         9.6, COL_WIN),
    ("mitochondrion",   9.6, COL_WIN),
    ("endom",           9.6, COL_WIN),
    ("extracellular",   9.6, COL_LOSE),
    ("cell_surface",    9.6, COL_LOSE),
]

for i, (comp, x_pos, color) in enumerate(compartments):
    y = 2.2 - i * 0.35
    # Compartment label box
    ax.add_patch(FancyBboxPatch(
        (x_pos, y - 0.12), 2.4, 0.28,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor="white", edgecolor=color, linewidth=1.2,
    ))
    ax.text(x_pos + 1.2, y + 0.02, comp, ha="center", va="center",
            fontsize=9, color=color, fontweight="bold")
    # Tiny win/lose indicator
    sym = "↑ ours" if color == COL_WIN else "↓ DeepLoc" if color == COL_LOSE else "≈ tie"
    ax.text(x_pos + 2.45, y + 0.02, sym, ha="left", va="center",
            fontsize=8, color=color, style="italic")

# Arrows from DeepLoc components to where they win/lose compartments
# Stage 2/3 aux heads help extracellular + cell_surface + membrane (DeepLoc's wins)
arrow(ax, 4.6, 4.5, 12.05, 1.95, color=COL_ARROW_DL, lw=1.2, curve=-0.10,
      label="sorting-signal aux →")
arrow(ax, 4.6, 6.1, 12.05, 0.60, color=COL_ARROW_DL, lw=1.2, curve=-0.30,
      label="signal peptide/TMD aux →")

# Arrows from Ours components to where we win - DISTINCT LABELS for each
arrow(ax, 17.1, 6.6, 12.05, 2.25, color=COL_ARROW_US, lw=1.2, curve=0.10,
      label="endom  ←  SPACE features")
arrow(ax, 17.1, 6.6, 12.05, 1.55, color=COL_ARROW_US, lw=1.2, curve=0.18,
      label="mitochondrion  ←  attention pool")
arrow(ax, 17.1, 6.6, 12.05, 1.85, color=COL_ARROW_US, lw=1.2, curve=0.14,
      label="nucleus  ←  cleanlab+thresholds")
arrow(ax, 17.1, 6.6, 12.05, 2.20, color=COL_ARROW_US, lw=1.2, curve=0.12,
      label="cytoplasm  ←  cleanlab+thresholds")

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COL_DEEPLOC, edgecolor=COL_DEEPLOC,
                   label="DeepLoc 2.1 components"),
    mpatches.Patch(facecolor=COL_OURS, edgecolor=COL_OURS,
                   label="Ours components"),
    Line2D([0], [0], color=COL_ARROW_DL, lw=2, label="DeepLoc → compartment"),
    Line2D([0], [0], color=COL_ARROW_US, lw=2, label="Ours → compartment"),
    Line2D([0], [0], color=COL_WIN, lw=2, marker="s", markersize=10,
           markerfacecolor="white", markeredgecolor=COL_WIN,
           label="Compartment where we win (teal)"),
    Line2D([0], [0], color=COL_LOSE, lw=2, marker="s", markersize=10,
           markerfacecolor="white", markeredgecolor=COL_LOSE,
           label="Compartment where DeepLoc wins (red)"),
]
ax.legend(handles=legend_elements, loc="lower left",
          bbox_to_anchor=(0.02, 0.30), fontsize=8, frameon=False, ncol=2)


# Caption / footer
ax.text(10, 0.10,
        "Net result on Kaggle subcellular-localization test:  ours 0.70382 public / 0.68659 private  >  DeepLoc 2.1 ~0.647  (Δ +0.04)",
        ha="center", va="center", fontsize=10, fontweight="bold", color=COL_TEXT)


# Save
fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=COL_BG)
plt.close(fig)
print(f"  Saved: {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")