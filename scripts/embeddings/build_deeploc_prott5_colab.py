#!/usr/bin/env python3
"""build_deeploc_prott5_colab.py

Generates a Colab notebook (.ipynb) that:
  1. Installs DeepLoc 2.1 + dependencies
  2. Generates 5 FASTA files (partitions 0-4) from df_adi.csv
  3. Runs deeploc2 -m Accurate -d cuda on each partition sequentially
  4. Saves all 5 prediction CSVs to Google Drive

Usage:
  python3 build_deeploc_prott5_colab.py
  → outputs notebooks/deeploc_prott5_colab.ipynb

Upload this notebook + deeploc-2.1.All.tar + df_adi.csv to Colab with A100 GPU.
Wall time: ~8-12 min per partition = ~50 min total.
"""

import json, textwrap
from pathlib import Path

def cell(source, cell_type="code"):
    """Build a notebook cell with robust indentation handling.

    Cell sources here use a 0-indent header comment on the first line
    followed by body lines indented 8+ spaces (a header line directly after
    the opening triple-quote, then indented body). textwrap.dedent() alone
    finds common prefix = 0 (due to the header) and strips nothing, so we
    dedent the BODY by its own common indent and leave the header at
    column 0.

    Result: header comments at column 0, top-level code at column 0,
    function bodies at column 4, nested blocks deeper. Always valid Python.
    """
    lines = source.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return {"cell_type": cell_type, "metadata": {}, "source": [],
                "outputs": [] if cell_type == "code" else None,
                "execution_count": None if cell_type == "code" else None}

    header = lines[0]
    body = lines[1:]
    body_indents = [len(l) - len(l.lstrip()) for l in body if l.strip()]
    if body_indents:
        common = min(body_indents)
        body = [l[common:] if l.strip() else "" for l in body]
    final = [header] + body
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
        """# DeepLoc 2.1 - ProtT5-XL (Accurate) 5-fold CV
        #
        # Runs inference on ALL 5 partitions (0-4) for an apples-to-apples
        # 5-fold cross-validation comparison against our pipeline.
        #
        # **Runtime**: A100 GPU (~50 min total, ~10 min per partition).
        # First run also downloads ~10 GB ProtT5-XL weights (3-5 min).
        #
        # **Step 1**: Upload these 2 files via sidebar ():
        #   - `/content/deeploc-2.1.All.tar`  (~50 MB, from colab_uploads/)
        #   - `/content/df_adi.csv`           (~10 MB, from colab_uploads/)
        #
        # Upload both to Colab's root dir. Then run all cells in order.
        #
        # **Step 2**: Run all cells in order (~50 min total on A100).
        #
        # **Step 3**: After completion, download all 5 CSVs from Drive:
        #   MyDrive/deeploc_5fold/results_partition{0..4}.csv
        #
        # **Step 4**: Locally run eval on each:
        #   for part in 0 1 2 3 4; do
        #       python3 eval_against_deeploc_2_1.py \\
        #           /path/to/results_partition${part}.csv \\
        #           --partition ${part}
        #   done
        """ , "markdown"))

    cells.append(cell(
        """# Mount Google Drive (where outputs will be saved)
        from google.colab import drive
        drive.mount('/content/drive')
        print("Drive mounted.")
        """ ))

    cells.append(cell(
        """# Upload the DeepLoc 2.1 tarball + df_adi.csv
        #
        # Use the  sidebar to upload:
        #   1. deeploc-2.1.All.tar  (from Downloads or colab_uploads/)
        #   2. df_adi.csv           (from data/df_adi.csv)
        #
        # Or if they're on Drive, just copy from there.

        import os, shutil

        UPLOAD_DIR = "/content/uploads"
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Check for tar
        tar_src = "/content/deeploc-2.1.All.tar"
        if os.path.exists(tar_src):
            shutil.copy(tar_src, f"{UPLOAD_DIR}/deeploc-2.1.All.tar")
            print(f"Found tar: {tar_src}")
        else:
            print("TARBALL NOT FOUND - upload via sidebar.")
            raise FileNotFoundError("Upload deeploc-2.1.All.tar first")

        # Check for CSV
        csv_src = "/content/df_adi.csv"
        if os.path.exists(csv_src):
            shutil.copy(csv_src, f"{UPLOAD_DIR}/df_adi.csv")
            import pandas as pd
            df = pd.read_csv(csv_src)
            n = len(df)
            n_parts = df['partition'].nunique() if 'partition' in df else 0
            print(f"Found df_adi.csv: {n} rows, {n_parts} partitions")
        else:
            print("df_adi.csv NOT FOUND - upload via sidebar.")
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
        print("Extracted:", os.listdir(extract_dir))
        pkg_dirs = [d for d in os.listdir(extract_dir) if d.startswith("deeploc")]
        pkg_path = os.path.join(extract_dir, pkg_dirs[0] if pkg_dirs else "deeploc2_package")
        print("Package path:", pkg_path)
        """ ))

    cells.append(cell(
        """# Generate FASTA files for all 5 partitions from df_adi.csv
        import pandas as pd, os

        df = pd.read_csv("/content/uploads/df_adi.csv")
        fasta_dir = "/content/partition_fastas"
        os.makedirs(fasta_dir, exist_ok=True)

        for part in sorted(df['partition'].unique()):
            p_df = df[df['partition'] == part]
            path = f"{fasta_dir}/partition_{part}.fasta"
            with open(path, 'w') as f:
                for _, r in p_df.iterrows():
                    seq = str(r['sequence']).replace(' ', '')
                    f.write(f'>{r["acc"]}\\n{seq}\\n')
            n = sum(1 for l in open(path) if l.startswith('>'))
            size = os.path.getsize(path) / 1024
            print(f"Partition {part}: {n} proteins, {size:.1f} KB")
        """ ))

    cells.append(cell(
        """# Install dependencies + DeepLoc2
        import subprocess, sys

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
        print("Dependencies installed.")

        # Editable install of DeepLoc2
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--no-build-isolation", "-e",
             "/content/deeploc2/deeploc2_package"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("STDERR:", result.stderr[-3000:])
            raise RuntimeError(f"DeepLoc2 install failed with code {result.returncode}")
        print("DeepLoc2 installed.")

        # Verify
        import shutil
        deeploc2_path = shutil.which("deeploc2")
        if deeploc2_path:
            print(f"deeploc2 found at {deeploc2_path}")
            result = subprocess.run(["deeploc2", "--help"], capture_output=True, text=True)
            print(result.stdout[:300])
        else:
            print("deeploc2 not on PATH - will use python -m for inference")
            result = subprocess.run(
                [sys.executable, "-m", "DeepLoc2.deeploc2", "--help"],
                capture_output=True, text=True
            )
            print(result.stdout[:300] if result.returncode == 0 else "Module also unavailable")
        """ ))

    cells.append(cell(
        """# Define the inference helper function (used by all 5 partition cells)
        import subprocess, time, os, sys, shutil

        def run_partition(partition: int):
            \"\"\"Run deeploc2 -m Accurate on a single partition FASTA.\"\"\"
            fasta_path = f"/content/partition_fastas/partition_{partition}.fasta"
            output_dir = f"/content/deeploc_results_p{partition}"
            os.makedirs(output_dir, exist_ok=True)

            # Build the deeploc2 command
            deeploc_cmd = shutil.which("deeploc2") or [sys.executable, "-m", "DeepLoc2.deeploc2"]
            if isinstance(deeploc_cmd, str):
                deeploc_cmd = [deeploc_cmd]
            deeploc_cmd += ["-f", fasta_path, "-o", output_dir, "-m", "Accurate", "-d", "cuda"]

            n_prots = sum(1 for l in open(fasta_path) if l.startswith('>'))
            print(f"\\n{'='*60}")
            print(f"PARTITION {partition}: {n_prots} proteins")
            print(f"Command: {' '.join(deeploc_cmd)}")
            print(f"{'='*60}\\n", flush=True)

            t0 = time.time()
            proc = subprocess.Popen(deeploc_cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in iter(proc.stdout.readline, ''):
                print(line, end='', flush=True)
            proc.wait()
            dt = time.time() - t0

            if proc.returncode != 0:
                print(f"\\n Partition {partition} FAILED with code {proc.returncode}")
                return False

            csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
            if csv_files:
                fpath = os.path.join(output_dir, csv_files[0])
                size = os.path.getsize(fpath) / 1024
                print(f"\\n Partition {partition} done in {dt:.0f}s ({dt/60:.1f}m)")
                print(f"   CSV: {csv_files[0]} ({size:.1f} KB)")
            else:
                print(f"\\nWARNING:  Partition {partition} done but no CSV found")
                print(f"   Files in output: {os.listdir(output_dir)}")
            return True

        print("Helper function defined. Ready to run partitions 0-4.")
        """ ))

    for part in range(5):
        cells.append(cell(
            f"""# Run partition {part} inference (~8-12 min on A100)
            import sys
            sys.stdout.flush()
            ok = run_partition({part})
            if not ok:
                raise RuntimeError(f"Partition {part} failed - stopping.")
            print(f"Finished partition {part}.")
            """ ))

    cells.append(cell(
        """# Copy all 5 result CSVs to Google Drive
        import shutil, os

        drive_dir = "/content/drive/MyDrive/deeploc_5fold"
        os.makedirs(drive_dir, exist_ok=True)

        for part in range(5):
            output_dir = f"/content/deeploc_results_p{part}"
            csv_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
            if csv_files:
                src = os.path.join(output_dir, csv_files[0])
                dst = os.path.join(drive_dir, f"results_partition{part}.csv")
                shutil.copy2(src, dst)
                rows = sum(1 for _ in open(src))
                print(f"Partition {part}: {src} → {dst}  ({rows-1} proteins, {os.path.getsize(dst)/1024:.1f} KB)")
            else:
                print(f"Partition {part}: NO CSV FOUND in {output_dir}")

        print(f"\\nAll CSVs saved to {drive_dir}/")
        print("Summary of partitions:")
        for part in range(5):
            path = os.path.join(drive_dir, f"results_partition{part}.csv")
            if os.path.exists(path):
                rows = sum(1 for _ in open(path))
                print(f"  Partition {part}: {rows-1} predictions ")
            else:
                print(f"  Partition {part}: MISSING ")
        """ ))

    cells.append(cell(
        """# DOWNLOAD THE CSVs MANUALLY
        #
        # Option A: From Google Drive
        #   My Drive → deeploc_5fold/ → download all 5 CSV files
        #
        # Option B: From Colab sidebar ()
        #   /content/deeploc_results_p{0..4}/ → download CSV from each
        #
        # === THEN LOCALLY ===
        # Move CSVs to ~/Downloads, then:
        #
        # for part in 0 1 2 3 4; do
        #     python3 scripts/evaluation/eval_against_deeploc_2_1.py \\
        #         ~/Downloads/results_partition${part}.csv
        # done
        #
        # This will produce 5 head-to-head tables in:
        #   output_eval_against_deeploc/headline_table_t05.md
        """ , "markdown"))

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

    out_path = Path(__file__).resolve().parents[2] / "notebooks" / "deeploc_prott5_colab.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Wrote {out_path}")
    print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
          f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")


if __name__ == "__main__":
    build()
