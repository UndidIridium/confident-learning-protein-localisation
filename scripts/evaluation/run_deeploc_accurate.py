#!/usr/bin/env python3
"""run_deeploc_accurate.py

Wrapper that properly runs DeepLoc 2.1 Accurate (ProtT5-XL) on partition 4,
avoiding the -m submodule path issue by calling predict() directly.
"""
import os, sys

# Must be set BEFORE any matplotlib/torch imports
os.environ["MPLBACKEND"] = "Agg"

import argparse
from DeepLoc2.deeploc2 import predict

if __name__ == "__main__":
    # Build args matching what deeploc2 CLI expects
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-m", "--model", default="Accurate", choices=["Fast", "Accurate"])
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()

    print(f"DeepLoc 2.1 — Accurate (ProtT5-XL)")
    print(f"  FASTA:  {args.fasta}")
    print(f"  Output: {args.output}")
    print(f"  Model:  {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Plot:   {args.plot}")
    print(f"  Note: First run downloads ~10 GB ProtT5-XL weights")
    print(f"  Expected: ~2-4 hours on MPS, ~15 min on GPU")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Run prediction
    exit_code = predict()
    sys.exit(exit_code)
