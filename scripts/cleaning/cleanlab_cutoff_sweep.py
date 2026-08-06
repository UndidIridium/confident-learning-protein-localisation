#!/usr/bin/env python3
"""cleanlab_cutoff_sweep.py

Sweep CL_CUTOFF ∈ {0.30, 0.35, 0.45} over the 5-fold CV pipeline to find
the best cleanlab self_confidence cutoff. The 0.40 default is part of v1 (the
champion). We test:
   0.30 : drop fewer rows (more aggressive data retention; less cleanlab effect)
   0.35 : slight relaxation from default
   0.45 : drop more rows (more aggressive pruning)

For each cutoff we run the full 5-fold CV and save its report under
output_cleanlab_cutoff_sweep/cutoff_*.json. After the sweep, we aggregate
into a single comparison table.

Usage:
    python3 cleanlab_cutoff_sweep.py
"""
import shutil, subprocess, json, time
from pathlib import Path
import numpy as np

PROJ = Path(__file__).parent.resolve()
OUT_DIR = PROJ / "output_cleanlab_cutoff_sweep"
OUT_DIR.mkdir(exist_ok=True)

# 0.40 is the v1 champion (already cached in output_champion_5fold_cv.heuristic.json)
# 0.30, 0.35, 0.45 are the new values to test
CUTOFFS = [0.30, 0.35, 0.45]


def main():
    print("=" * 72)
    print("  Cleanlab cutoff sweep — 5-fold CV at each CL_CUTOFF")
    print("=" * 72)
    print(f"  cutoffs to test: {CUTOFFS}")
    print()

    # Save current champion json so we can restore at the end
    canon_champ_json = PROJ / "output_champion_5fold_cv.json"
    if canon_champ_json.exists():
        bck = PROJ / "output_champion_5fold_cv.backup_pre_sweep.json"
        shutil.copy(canon_champ_json, bck)
        print(f"Backed up canonical champion JSON to {bck.name}\n")

    base_log = OUT_DIR / "sweep.log"
    results = {}
    t_start = time.time()
    for c in CUTOFFS:
        t_one = time.time()
        print(f"\n========= CL_CUTOFF = {c} =========", flush=True)
        proc = subprocess.run(
            ["python3", "champion_5fold_cv.py", "--cl-cutoff", str(c)],
            cwd=str(PROJ),
            capture_output=True, text=True, timeout=1800,  # bumped from 900s (code-reviewer)
        )
        if proc.returncode != 0:
            print(f"  ❌ FAIL — last stderr: {proc.stderr[-1500:]}")
            continue
        # Save the resulting per-cutoff report
        tag = f"cutoff_{c:.2f}".replace(".", "p")
        dst = OUT_DIR / f"{tag}.json"
        shutil.copy(canon_champ_json, dst)
        d = json.load(open(dst))
        d["cl_cutoff"] = c
        json.dump(d, open(dst, "w"), indent=2)
        results[c] = d
        print(f"  ✓ champion_mean={d['champion_mean']:.4f}  baseline_mean={d['baseline_mean']:.4f}  "
              f"gain={d['overall_gain']:+.4f}  per-fold std={d['champion_std']:.4f}  "
              f"({time.time()-t_one:.0f}s)")

    # ---- Add the v1 champion as a reference (CL_CUTOFF=0.40) ----
    v1_path = PROJ / "output_champion_5fold_cv.heuristic.json"
    if v1_path.exists():
        v1 = json.load(open(v1_path))
        v1["cl_cutoff"] = 0.40
        results[0.40] = v1
        json.dump(v1, open(OUT_DIR / "cutoff_0p40.json", "w"), indent=2)
        print(f"\n(v1 champion loaded as reference: cutoff=0.40  champion_mean={v1['champion_mean']:.4f})")

    # ---- Aggregate ----
    print("\n" + "=" * 72)
    print("  SUMMARY — cleanlab cutoff sweep")
    print("=" * 72)
    print(f"  {'cutoff':>8}  {'baseline':>10}  {'champion':>10}  {'gain':>8}  {'per-fold std':>12}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*12}")
    for c in sorted(results.keys()):
        d = results[c]
        print(f"  {c:>8.2f}  {d['baseline_mean']:>10.4f}  "
              f"{d['champion_mean']:>10.4f}  {d['overall_gain']:>+8.4f}  "
              f"{d['champion_std']:>12.4f}")

    # ---- Per-compartment breaks at each cutoff ----
    comps = ["Membrane","Cytoplasm","Nucleus","Extracell","Cell_surf","Mito","Endom"]
    print("\nPer-compartment F1 (mean over 5 folds):")
    print(f"  {'cutoff':>8}  " + "  ".join(f"{c[:7]:>8s}" for c in comps))
    for c in sorted(results.keys()):
        d = results[c]
        pc = []
        for j in range(7):
            pc.append(np.mean([r['champion_per_class'][j] for r in d['per_fold']]))
        print(f"  {c:>8.2f}  " + "  ".join(f"{x:>8.4f}" for x in pc))

    # ---- Find the best cutoff ----
    best_cutoff = max(results.keys(), key=lambda k: results[k]['champion_mean'])
    print(f"\n  ⭐ Best cutoff: {best_cutoff}  →  champion_mean={results[best_cutoff]['champion_mean']:.4f}")

    # ---- Save aggregate report ----
    summary_path = OUT_DIR / "summary.json"
    summary = {
        "sweep_cutoffs": CUTOFFS,
        "v1_champion_cutoff": 0.40,
        "results": {
            str(c): {
                "champion_mean": results[c]["champion_mean"],
                "champion_std":  results[c]["champion_std"],
                "baseline_mean": results[c]["baseline_mean"],
                "overall_gain":  results[c]["overall_gain"],
            }
            for c in results
        },
        "best_cutoff": best_cutoff,
    }
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"\n  → {summary_path}")
    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
