#!/usr/bin/env python3
"""Save a float32 copy of the COFETT prior (Qwen2.5-0.5B).

The hub checkpoint is stored in bfloat16; the frozen-prior code computes the exact partition
function in float32/float64 and expects float32 weights, so the paper's COFETT runs used a
float32 copy of Qwen/Qwen2.5-0.5B. Weights and tokenizer are otherwise unchanged.

    python cofett/make_fp32_prior.py <OUT_DIR> [--model Qwen/Qwen2.5-0.5B]
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    a = ap.parse_args()
    AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).save_pretrained(a.out)
    AutoTokenizer.from_pretrained(a.model).save_pretrained(a.out)
    print(f"float32 prior written to {a.out}")


if __name__ == "__main__":
    main()
