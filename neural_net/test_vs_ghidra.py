#!/usr/bin/env python3
"""
test_vs_ghidra.py

Compare your model vs Ghidra on the ORIGINAL test set:

For each stem in test_stems.txt:
  - Load assembly:   ../training_data/<stem>.s   (model input)
  - Load original C: ../training_data/<stem>.c   (ground truth)
  - Load Ghidra C:   ../training_data/<stem>.ghidra.c (from generate_ghidra.py)
  - Generate model C from assembly using codet5_poc_model
  - Normalize all C texts
  - Compute token-level similarity to original C for:
      * Ghidra
      * model

Outputs:
  - Average similarity for Ghidra vs original
  - Average similarity for model vs original
"""

import math
import pathlib
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

TRAINING_DATA_DIR = (ROOT / "training_data").resolve()
TEST_STEMS_FILE = HERE / "test_stems.txt"
MODEL_DIR = HERE / "codet5_poc_model"

# ---- Normalization (copied from your training script style) ----
import re

def normalize_c_style(c_src: str) -> str:
    # Canonicalize brace placement: )\n{ or ){ -> ) {
    c_src = re.sub(r"\)\s*\n\s*\{", r") {", c_src)
    c_src = re.sub(r"\)\s*\{", r") {", c_src)

    includes = []
    body = []
    for ln in c_src.splitlines():
        if ln.strip().startswith("#include"):
            key = re.sub(r"\s+", " ", ln.strip())
            if key not in includes:
                includes.append(key)
        else:
            body.append(ln)

    prefer = ["<stdio.h>", "<stdlib.h>", "<string.h>", "<ctype.h>", "<math.h>"]
    ordered = []
    for p in prefer:
        for inc in includes:
            if p in inc and inc not in ordered:
                ordered.append(inc)
    for inc in includes:
        if inc not in ordered:
            ordered.append(inc)
    ordered = ordered[:6]

    out = (("\n".join(ordered) + "\n\n") if ordered else "") + "\n".join(body)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

def normalize_for_eval(c_src: str) -> str:
    return "\n".join(line.rstrip() for line in normalize_c_style(c_src).splitlines())


def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_test_stems(path: pathlib.Path) -> List[str]:
    if not path.exists():
        raise RuntimeError(f"Missing test_stems.txt at {path}")
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

# ---- Simple token-level Levenshtein similarity ----

def levenshtein(a: List[str], b: List[str]) -> int:
    """Compute edit distance between two token lists."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = temp
    return dp[m]


def similarity(a: str, b: str) -> float:
    """
    1 - (edit distance / max length), using whitespace-tokenized sequences.
    Returns 0..1; 1 means identical.
    """
    a_tok = a.split()
    b_tok = b.split()
    if not a_tok and not b_tok:
        return 1.0
    dist = levenshtein(a_tok, b_tok)
    denom = max(len(a_tok), len(b_tok), 1)
    return 1.0 - (dist / float(denom))


# ---- Model loading / generation ----

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_one(tokenizer, model, device, src_text: str) -> str:
    inputs = tokenizer(
        src_text,
        return_tensors="pt",
        truncation=True,
        max_length=384,  # same default as training if not overridden
    ).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


def main() -> None:
    print("[test_vs_ghidra] Loading model from", MODEL_DIR)
    tokenizer, model, device = load_model()

    stems = read_test_stems(TEST_STEMS_FILE)
    print(f"[test_vs_ghidra] Found {len(stems)} test stems")

    ghidra_sims: List[float] = []
    model_sims: List[float] = []
    used = 0

    for idx, stem in enumerate(stems, 1):
        c_path = TRAINING_DATA_DIR / f"{stem}.c"
        s_path = TRAINING_DATA_DIR / f"{stem}.s"
        ghidra_path = TRAINING_DATA_DIR / f"{stem}.ghidra.c"

        if not (c_path.exists() and s_path.exists() and ghidra_path.exists()):
            print(f"[{idx}] SKIP (missing one of .c/.s/.ghidra.c) for stem {stem}")
            continue

        print(f"\n[{idx}/{len(stems)}] stem={stem}")

        orig_c = normalize_for_eval(read_text(c_path))
        ghidra_c = normalize_for_eval(read_text(ghidra_path))

        asm_src = read_text(s_path)
        model_c_raw = generate_one(tokenizer, model, device, asm_src)
        model_c = normalize_for_eval(model_c_raw)

        sim_gh = similarity(ghidra_c, orig_c)
        sim_model = similarity(model_c, orig_c)

        print(f"  Ghidra vs original similarity: {sim_gh:.3f}")
        print(f"  Model  vs original similarity: {sim_model:.3f}")

        ghidra_sims.append(sim_gh)
        model_sims.append(sim_model)
        used += 1

    if used == 0:
        print("[test_vs_ghidra] No usable samples found.")
        return

    avg_gh = sum(ghidra_sims) / used
    avg_model = sum(model_sims) / used

    print("\n[test_vs_ghidra] Summary over", used, "samples:")
    print(f"  Avg Ghidra similarity to original: {avg_gh:.3f}")
    print(f"  Avg Model  similarity to original: {avg_model:.3f}")


if __name__ == "__main__":
    main()

