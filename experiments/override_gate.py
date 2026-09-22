
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
