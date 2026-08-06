#!/usr/bin/env python3
"""build_deeploc_accurate_p4_colab.py

Generates a Colab notebook (.ipynb) for running DeepLoc 2.1 Accurate
(ProtT5-XL) on just partition 4 — for the definitive apples-to-apples
head-to-head against our ProtT5 champion (0.8011).

Runtime: ~10-15 min on A100 GPU (first run also downloads ~10 GB ProtT5-XL
weights). Much faster than the full 5-fold notebook (~50 min).

Upload to Colab:
  1.  deeploc-2.1.All.tar  (from inputs/ — ~50 MB)
  2.  df_adi.csv           (from data/ — ~10 MB)

Or use the notebook's built-in FASTA generator from df_adi.csv.
"""

import json, textwrap

PROJ = "/Volumes/BOMBOCLAT/project_JL"


def cell(source, cell_type="code"):
    """Build a notebook cell handling mixed indentation robustly."""
    lines = source.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return {"cell_type": cell_type, "metadata": {}, "source": [],
                "outputs": [] if cell_type == "code" else None,
                "execution_count": None if cell_type == "code" else None}

    nonempty = [l for l in lines if l.strip()]
    max_indent = max(len(l) - len(l.lstrip()) for l in nonempty)
    padded = []
    for l in lines:
        if not l.strip():
            padded.append("")
            continue
        cur_indent = len(l) - len(l.lstrip())
        if cur_indent < max_indent:
            padded.append(" " * max_indent + l)
        else:
            padded.append(l)

    joined = "\n".join(padded)
    dedented = textwrap.dedent(joined)
    final = dedented.split("\n")
    while final and not final[-1].strip():
        final.pop()
    src_lines = [l + "\n" for l in final]
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": src_lines,
        "outputs": [] if cell_type == "code" else None,
        "execution_count": None if cell_type == "code" else None,
    }


def build():
    cells = []

    cells.append(cell(
        """# DeepLoc 2.1 Accurate (ProtT5-XL) — Partition 4 Inference
        #
        # Runs deeploc2 on partition 4 only for an apples-to-apples head-to-head
        # against our ProtT5 champion pipeline (0.8011 F1 on this same partition).
        #
        # **Runtime**: ~10-15 min on A100 GPU.
        #   - First run: ~3-5 min to download ProtT5-XL weights (~10 GB)
        #   - Inference: ~0.3s/protein × 3,276 proteins ≈ 15 min
        #
        # **Upload these files via 📁 sidebar** before running:
        #   1. `/content/deeploc-2.1.All.tar` (~50 MB)
        #   2. `/content/df_adi.csv` (~10 MB)
        #
        # **After completion**: download from 📁 → `/content/deeploc_p4_results/`
        #
        # **Then locally**: eval_against_deeploc_2_1.py on the CSV
        """ , "markdown"))

    cells.append(cell(
        """# Mount Google Drive (optional — for backup)
        from google.colab import drive
        drive.mount('/content/drive')
        print("Drive mounted.")
        """ ))

    cells.append(cell(
        """# Upload files — check they exist or prompt user
        import os, shutil

        UPLOAD_DIR = "/content/uploads"
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Check for tar
        tar_src = "/content/deeploc-2.1.All.tar"
        if os.path.exists(tar_src):
            shutil.copy(tar_src, f"{UPLOAD_DIR}/deeploc-2.1.All.tar")
            print(f"✅ Found tar: {tar_src} ({os.path.getsize(tar_src)/1024**2:.0f} MB)")
        else:
            print("❌ TARBALL NOT FOUND — upload deeploc-2.1.All.tar via sidebar.")
            raise FileNotFoundError("Upload deeploc-2.1.All.tar first")

        # Check for CSV
        csv_src = "/content/df_adi.csv"
        if os.path.exists(csv_src):
            shutil.copy(csv_src, f"{UPLOAD_DIR}/df_adi.csv")
            import pandas as pd
            df = pd.read_csv(csv_src)
            n = len(df)
            n_p4 = (df['partition'] == 4).sum()
            print(f"✅ Found df_adi.csv: {n} rows, {n_p4} in partition 4")
        else:
            print("❌ df_adi.csv NOT FOUND — upload via sidebar.")
            raise FileNotFoundError("Upload df_adi.csv first")
        """ ))

    cells.append(cell(
        """# Extract DeepLoc 2.1 tarball
        import tarfile, os
        tar_path = "/content/uploads/deeploc-2.1.All.tar"
        extract_dir = "/content/deeploc2"
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(extract_dir)
        print("Extracted contents:", os.listdir(extract_dir))
        pkg_dirs = [d for d in os.listdir(extract_dir) if d.startswith("deeploc")]
        pkg_path = os.path.join(extract_dir, pkg_dirs[0] if pkg_dirs else "deeploc2_package")
        print(f"Package path: {pkg_path}")
        """ ))

    cells.append(cell(
        """# Generate partition 4 FASTA from df_adi.csv
        import pandas as pd, os

        df = pd.read_csv("/content/uploads/df_adi.csv")
        p4 = df[df['partition'] == 4]
        fasta_dir = "/content/partition_fastas"
        os.makedirs(fasta_dir, exist_ok=True)
        fasta_path = f"{fasta_dir}/partition_4.fasta"

        with open(fasta_path, 'w') as f:
            for _, r in p4.iterrows():
                seq = str(r['sequence']).replace(' ', '').replace('\\n', '')
                f.write(f'>{r["acc"]}\\\n{seq}\\\n')

        n_prots = sum(1 for l in open(fasta_path) if l.startswith('>'))
        size = os.path.getsize(fasta_path) / 1024
        print(f"Partition 4: {n_prots} proteins, {size:.1f} KB")
        print(f"FASTA saved to: {fasta_path}")

        # Quick sanity check
        with open(fasta_path) as f:
            first = f.readline().strip()
            second = f.readline().strip()
            print(f"First protein: {first}")
            print(f"Sequence length: {len(second)} aa")
        """ ))

    cells.append(cell(
        """# Install dependencies + DeepLoc2
        import subprocess, sys

        print("Installing dependencies...")
        deps = [
            "biopython", "fair-esm", "onnxruntime", "pytorch_lightning",
            "transformers<4.36.0", "sentencepiece", "setuptools",
        ]
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + deps,
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("STDERR:", result.stderr[-3000:])
            raise RuntimeError(f"pip install failed with code {result.returncode}")
        print("✅ Dependencies installed.")

        print("Installing DeepLoc2...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--no-build-isolation", "-e",
             "/content/deeploc2/deeploc2_package"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("STDERR:", result.stderr[-3000:])
            raise RuntimeError(f"DeepLoc2 install failed with code {result.returncode}")
        print("✅ DeepLoc2 installed.")

        # Verify
        import shutil
        deeploc2_path = shutil.which("deeploc2")
        if deeploc2_path:
            print(f"✅ deeploc2 found at {deeploc2_path}")
            result = subprocess.run(["deeploc2", "--help"], capture_output=True, text=True)
            print(result.stdout[:300])
        else:
            raise RuntimeError("deeploc2 not found on PATH after install")
        """ ))

    cells.append(cell(
        """# RUN DEEPLOC 2.1 ACCURATE (ProtT5-XL) ON PARTITION 4
        #
        # This cell does all the work:
        #   - Downloads ~10 GB ProtT5-XL weights on first run (3-5 min)
        #   - Runs inference at ~0.3s/protein on T4/A100
        #   - Total: ~10-15 min on A100, ~20-25 min on T4
        #
        # The output is streamed live so you can see progress.

        import subprocess, time, os, sys

        fasta_path = "/content/partition_fastas/partition_4.fasta"
        output_dir = "/content/deeploc_p4_results"
        os.makedirs(output_dir, exist_ok=True)

        n_prots = sum(1 for l in open(fasta_path) if l.startswith('>'))
        print(f"Starting inference on {n_prots} proteins")
        print(f"Model: Accurate (ProtT5-XL), device: cuda")
        print(f"Output: {output_dir}")
        print(f"NOTE: First run downloads ~10 GB ProtT5-XL weights (3-5 min).")
        print(f"Inference: ~0.3s/protein on T4 = ~{n_prots * 0.3 / 60:.0f} min total.", flush=True)
        print()

        t0 = time.time()

        proc = subprocess.Popen(
            ["deeploc2", "-f", fasta_path, "-o", output_dir,
             "-m", "Accurate", "-d", "cuda"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )

        for line in iter(proc.stdout.readline, ''):
            print(line, end='', flush=True)

        proc.wait()
        dt = time.time() - t0

        if proc.returncode != 0:
            print(f"\n❌ Inference FAILED with code {proc.returncode}.")
            raise RuntimeError(f"deeploc2 exited with code {proc.returncode}")

        print(f"\n{'='*60}")
        print(f"✅ INFERENCE COMPLETED in {dt:.0f}s ({dt/60:.1f}m)")
        print(f"{'='*60}")
        print()

        # Show output files
        csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
        if csv_files:
            for fname in csv_files:
                fpath = os.path.join(output_dir, fname)
                size = os.path.getsize(fpath) / 1024
                print(f"  {fname}: {size:.1f} KB")
                rows = sum(1 for _ in open(fpath))
                print(f"    {rows-1} predictions (excluding header)")
        else:
            print(f"  All files: {os.listdir(output_dir)}")

        # Save to Drive too
        drive_dir = "/content/drive/MyDrive/deeploc_p4_accurate"
        os.makedirs(drive_dir, exist_ok=True)
        for fname in csv_files:
            src = os.path.join(output_dir, fname)
            dst = os.path.join(drive_dir, fname)
            import shutil
            shutil.copy2(src, dst)
            print(f"  Also saved to Drive: {dst}")
        """ ))

    cells.append(cell(
        """# --- DOWNLOAD INSTRUCTIONS ---
        #
        # The prediction CSV is in TWO places:
        #
        #   1. Colab 📁 sidebar:  /content/deeploc_p4_results/
        #   2. Google Drive:      MyDrive/deeploc_p4_accurate/
        #
        # Download the CSV file from either location.
        #
        # === THEN LOCALLY ===
        # Run the evaluation:
        #
        #   python3 /Volumes/BOMBOCLAT/project_JL/eval_against_deeploc_2_1.py \\
        #       ~/Downloads/results_*.csv
        #
        # This will produce a head-to-head table showing per-compartment F1
        # for both DeepLoc Accurate and our best ProtT5 champion (0.8011).
        """ , "markdown"))

    cells.append(cell(
        """# Quick peek at the predictions before downloading
        import pandas as pd, os

        csv_files = [f for f in os.listdir("/content/deeploc_p4_results")
                     if f.endswith(".csv")]
        if csv_files:
            df = pd.read_csv(os.path.join("/content/deeploc_p4_results", csv_files[0]))
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nHead (first 5):")
            print(df.head().to_string())
            print(f"\nTail (last 5):")
            print(df.tail().to_string())
            print(f"\nPrediction stats:")
            # Show probability distribution per compartment
            prob_cols = [c for c in df.columns if c != 'ID']
            for c in prob_cols:
                if df[c].dtype in ['float64', 'float32']:
                    print(f"  {c:>20s}: mean={df[c].mean():.3f}, "
                          f">=0.5={(df[c] >= 0.5).sum()}")
            print(f"\n✅ Ready for download.")
        else:
            print("No CSV found yet — inference may still be running.")
        """ ))

    # ─── Compile notebook ──────────────────────────
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }

    out_path = f"{PROJ}/inputs/deeploc_accurate_p4_colab.ipynb"
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Wrote {out_path}")
    print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
          f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
    print(f"\nUpload to Colab:")
    print(f"  1. {PROJ}/inputs/deeploc-2.1.All.tar  (~50 MB)")
    print(f"  2. data/df_adi.csv                    (~10 MB)")
    print(f"\nThen run all cells in order. Takes ~10-15 min on A100.")


if __name__ == "__main__":
    build()
