
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
