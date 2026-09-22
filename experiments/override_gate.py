#!/usr/bin/env python3
"""Sanctioned gate override — the replacement for hand-editing gate JSON.

    python experiments/override_gate.py runs estimator --reason "proceeding exploratory; \
        E1 failed at b>=0.5, validated range 0-0.25"

Preserves the original artifact inside the override, logs the deviation, and makes
every downstream ledger record carry '<gate>_OVERRIDDEN' instead of '<gate>'.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cprd.prereg import apply_gate_override  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("gate", help="gate name, e.g. estimator")
    ap.add_argument("--reason", required=True)
    args = ap.parse_args()
    print(apply_gate_override(args.runs_dir, args.gate, args.reason))


if __name__ == "__main__":
    main()
