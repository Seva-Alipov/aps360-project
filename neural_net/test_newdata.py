#!/usr/bin/env python3
import os
import pathlib
import math
import inspect
from typing import Dict, List

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

HERE = pathlib.Path(__file__).resolve().parent
MODEL_DIR = (HERE / "codet5_poc_model").resolve()
DATA_DIR = (HERE / ".." / "test_data").resolve()

MAX_SRC_LEN = int(os.environ.get("MAX_SRC_LEN", "384"))
MAX_TGT_LEN = int(os.environ.get("MAX_TGT_LEN", "256"))
BATCH_EVAL  = int(os.environ.get("BATCH_EVAL", "4"))

def list_pairs() -> List[str]:
    if not DATA_DIR.exists():
        raise RuntimeError(f"Test data directory does not exist: {DATA_DIR}")
    c = {p.stem for p in DATA_DIR.glob("*.c")}
    s = {p.stem for p in DATA_DIR.glob("*.s")}
    stems = sorted(c & s)
    if not stems:
        raise RuntimeError(f"No (.s, .c) pairs found in {DATA_DIR}")
    return stems

def load_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

import re

def normalize_c_style(c_src: str) -> str:
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

def make_dataset_from_stems(stems: List[str]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for stem in stems:
        c_path = DATA_DIR / f"{stem}.c"
        s_path = DATA_DIR / f"{stem}.s"
        if not (c_path.exists() and s_path.exists()):
            continue
        c_txt = load_text(c_path)
        s_txt = load_text(s_path)
        if c_txt and s_txt:
            c_txt = normalize_c_style(c_txt)
            items.append({"src": s_txt, "tgt": c_txt})
    if not items:
        raise RuntimeError(
            f"Found stems but no usable (.s, .c) text pairs in {DATA_DIR}"
        )
    return items

print(f"Loading model and tokenizer from: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, use_safetensors=True)

stems = list_pairs()
print(f"Found {len(stems)} stems in {DATA_DIR}")
items = make_dataset_from_stems(stems)
print(f"Built {len(items)} src/tgt pairs from test_data")

raw_eval = Dataset.from_list(items)

def preprocess(batch):
    model_inputs = tokenizer(
        batch["src"],
        max_length=MAX_SRC_LEN,
        truncation=True,
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["tgt"],
            max_length=MAX_TGT_LEN,
            truncation=True,
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_eval = raw_eval.map(
    preprocess,
    batched=True,
    remove_columns=["src", "tgt"],
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

def build_eval_args() -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())

    kw = {
        "output_dir": str(HERE / "codet5_test_newdata_ckpt"),
        "per_device_eval_batch_size": BATCH_EVAL,
        "logging_steps": 50,
    }
    if "per_device_train_batch_size" in params:
        kw["per_device_train_batch_size"] = BATCH_EVAL

    if "bf16" in params:
        kw["bf16"] = use_bf16
    if "fp16" in params:
        kw["fp16"] = use_fp16

    eval_key = "evaluation_strategy" if "evaluation_strategy" in params else (
        "eval_strategy" if "eval_strategy" in params else None
    )
    if eval_key is not None:
        kw[eval_key] = "no"

    if "label_smoothing_factor" in params:
        kw["label_smoothing_factor"] = 0.1

    print(f"[Eval TrainingArguments compat] using keys: {sorted(kw.keys())}")
    return TrainingArguments(**kw)

args = build_eval_args()

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("\nRunning evaluation on ../test_data ...")
metrics = trainer.evaluate()

loss = metrics.get("eval_loss")
print("\n====== New Data Evaluation ======")
print(f"Number of examples: {len(tokenized_eval)}")
if loss is not None:
    print(f"Eval loss: {loss:.4f}")
    try:
        ppl = math.exp(loss)
        print(f"Perplexity: {ppl:.4f}")
    except OverflowError:
        print("Perplexity: overflow (loss too large)")
else:
    print("Eval loss not found in metrics:", metrics)

# ---- Per-token accuracy (logit argmax vs labels, ignoring -100) ----

print("\nComputing per-token accuracy on ../test_data ...")
pred_output = trainer.predict(tokenized_eval)
preds = pred_output.predictions
labels = pred_output.label_ids

# Some models return a tuple for predictions
if isinstance(preds, tuple):
    preds = preds[0]

# preds: (batch, seq_len, vocab_size) -> argmax over vocab
pred_ids = np.argmax(preds, axis=-1)

# labels: (batch, seq_len), with -100 for ignored positions
mask = labels != -100
total_tokens = mask.sum()
if total_tokens == 0:
    token_acc = float("nan")
    print("No labeled tokens found for accuracy computation.")
else:
    correct = (pred_ids == labels) & mask
    token_acc = correct.sum() / total_tokens
    print(f"Per-token accuracy (ignored label=-100 positions): {token_acc:.4f}")

print("\nRaw metrics dict from Trainer.evaluate():", metrics)

