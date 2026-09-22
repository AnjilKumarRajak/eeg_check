#!/usr/bin/env python3
"""Render all ICLR figures from gates + ledger.   python experiments/make_figures.py --runs-dir runs"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cprd.figures import make_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", default="runs")
args = ap.parse_args()
paths = make_all(args.runs_dir)
print(f"{len(paths)} figures written to {os.path.join(args.runs_dir, 'figures')}")
