#!/usr/bin/env python3
"""
Evaluate how "confused" the trained CodeT5 model is on validation and test splits.

Metrics per split:
  - avg_neg_log_likelihood (per token)
  - perplexity
  - token_accuracy (teacher-forced next-token accuracy)

Assumes:
  - train_codet5-small.py has already been run
  - codet5_poc_model/ exists next to this file
  - val_stems.txt and test_stems.txt exist next to this file
  - training_data/ contains the *.s / *.c pairs
"""

import os
import pathlib
import re
from typing import Dict, List, Tuple

import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "training_data").resolve()

MODEL_DIR = HERE / "codet5_poc_model"
VAL_STEMS = HERE / "val_stems.txt"
TEST_STEMS = HERE / "test_stems.txt"

# Keep these consistent with your training script
MAX_SRC_LEN = int(os.environ.get("MAX_SRC_LEN", "384"))
MAX_TGT_LEN = int(os.environ.get("MAX_TGT_LEN", "256"))
BATCH_EVAL  = int(os.environ.get("BATCH_EVAL", "4"))


def read_list(path: pathlib.Path) -> List[str]:
    if not path.exists():
        raise RuntimeError(f"Missing split file: {path}")
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def load_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_c_style(c_src: str) -> str:
    """
    Same as in train_codet5-small.py:
    Canonicalize brace placement and dedupe/normalize includes.
    """
    # Canonicalize brace position: )\\n{ or ){ -> ) {
    c_src = re.sub(r"\)\s*\n\s*\{", r") {", c_src)
    c_src = re.sub(r"\)\s*\{", r") {", c_src)

    # Collect unique includes
    includes = []
    body = []
    for ln in c_src.splitlines():
        if ln.strip().startswith("#include"):
            key = re.sub(r"\s+", " ", ln.strip())
            if key not in includes:
                includes.append(key)
        else:
            body.append(ln)

    # Prioritize common headers, limit count
    prefer = ["<stdio.h>", "<stdlib.h>", "<string.h>", "<ctype.h>", "<math.h>"]
    ordered = []
    for p in prefer:
        for inc in includes:
            if p in inc and inc not in ordered:
                ordered.append(inc)
    for inc in includes:
        if inc not in ordered:
            ordered.append(inc)
    ordered = ordered[:6]  # hard cap to avoid spam

    out = (("\n".join(ordered) + "\n\n") if ordered else "") + "\n".join(body)
    out = re.sub(r"\n{3,}", "\n\n", out)

    return out.strip()


def make_dataset_from_stems(stems: List[str]) -> List[Dict[str, str]]:
    items = []
    for stem in stems:
        c_path = DATA_DIR / f"{stem}.c"
        s_path = DATA_DIR / f"{stem}.s"
        if not (c_path.exists() and s_path.exists()):
            continue
        c_txt = load_text(c_path)
        s_txt = load_text(s_path)
        if c_txt and s_txt:
            c_txt = normalize_c_style(c_txt)
            items.append({"src": s_txt, "tgt": c_txt, "stem": stem})
    return items


def load_model_and_tokenizer():
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def batch_iter(items: List[Dict[str, str]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def evaluate_split(
    name: str,
    stems_path: pathlib.Path,
    tokenizer,
    model,
    device,
) -> Tuple[float, float, float]:
    print(f"[eval:{name}] Loading stems from {stems_path} ...")
    stems = read_list(stems_path)
    if not stems:
        raise RuntimeError(f"No stems found in {stems_path}")

    dataset = make_dataset_from_stems(stems)
    if not dataset:
        raise RuntimeError(
            f"No valid (.s, .c) pairs for {name} after filtering. "
            f"Check training_data/ for *.s + *.c pairs."
        )

    print(f"[eval:{name}] Using {len(dataset)} examples")

    total_tokens = 0
    loss_sum = 0.0
    correct_tokens = 0

    for batch_idx, batch in enumerate(batch_iter(dataset, BATCH_EVAL), start=1):
        src_list = [ex["src"] for ex in batch]
        tgt_list = [ex["tgt"] for ex in batch]

        # Tokenize source
        enc = tokenizer(
            src_list,
            max_length=MAX_SRC_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        # Tokenize target as labels
        with tokenizer.as_target_tokenizer():
            dec = tokenizer(
                tgt_list,
                max_length=MAX_TGT_LEN,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

        labels = dec["input_ids"]
        # Mask padding tokens in labels
        labels[labels == tokenizer.pad_token_id] = -100

        enc = {k: v.to(device) for k, v in enc.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(**enc, labels=labels)
            logits = outputs.logits  # [B, T, V]

        # Reshape for token-wise metrics
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)        # [B*T, V]
        labels_flat = labels.view(-1)                    # [B*T]

        # Cross-entropy summed over tokens (ignore padding)
        ce_sum = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=-100,
            reduction="sum",
        )
        loss_sum += ce_sum.item()

        # Count non-pad tokens
        non_pad_mask = labels_flat != -100
        num_tokens = non_pad_mask.sum().item()
        total_tokens += num_tokens

        # Token-level accuracy (still under teacher forcing)
        preds_flat = logits_flat.argmax(dim=-1)
        correct_tokens += (preds_flat[non_pad_mask] == labels_flat[non_pad_mask]).sum().item()

        if batch_idx % 10 == 0 or batch_idx == 1:
            print(
                f"[eval:{name}] Processed {batch_idx * BATCH_EVAL} examples "
                f"(approx)..."
            )

    avg_nll = loss_sum / max(total_tokens, 1)
    perplexity = math.exp(avg_nll)
    token_accuracy = correct_tokens / max(total_tokens, 1)

    print(f"\n=== {name.upper()} METRICS ===")
    print(f"Total tokens          : {total_tokens}")
    print(f"Avg NLL per token     : {avg_nll:.4f}")
    print(f"Perplexity            : {perplexity:.4f}")
    print(f"Token accuracy        : {token_accuracy:.4%}")
    print("=============================\n")

    return avg_nll, perplexity, token_accuracy


def main():
    print("[eval] Loading model and tokenizer...")
    tokenizer, model, device = load_model_and_tokenizer()

    results = {}

    # Validation split
    if VAL_STEMS.exists():
        results["validation"] = evaluate_split(
            "validation", VAL_STEMS, tokenizer, model, device
        )
    else:
        print("[eval] WARNING: val_stems.txt not found; skipping validation.")

    # Test split
    if TEST_STEMS.exists():
        results["test"] = evaluate_split(
            "test", TEST_STEMS, tokenizer, model, device
        )
    else:
        print("[eval] WARNING: test_stems.txt not found; skipping test.")

    print("=== SUMMARY ===")
    for split, (nll, ppl, acc) in results.items():
        print(
            f"{split:10s} | NLL: {nll:.4f} | PPL: {ppl:.4f} | "
            f"token_acc: {acc:.4%}"
        )


if __name__ == "__main__":
    main()

