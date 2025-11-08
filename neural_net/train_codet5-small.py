#!/usr/bin/env python3
import os
import pathlib
import random
import inspect
from typing import Dict, List

import transformers
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "training_data").resolve()

MODEL_NAME = os.environ.get("MODEL_NAME", "Salesforce/codet5-small")
SEED = int(os.environ.get("SEED", "42"))

TRAIN_FRAC = float(os.environ.get("TRAIN_FRAC", "0.8"))
VAL_FRAC   = float(os.environ.get("VAL_FRAC", "0.1"))

MAX_SRC_LEN = int(os.environ.get("MAX_SRC_LEN", "384"))
MAX_TGT_LEN = int(os.environ.get("MAX_TGT_LEN", "256"))

EPOCHS = float(os.environ.get("EPOCHS", "10"))
LR     = float(os.environ.get("LR", "3e-4"))

BATCH_TRAIN = int(os.environ.get("BATCH_TRAIN", "4"))
BATCH_EVAL  = int(os.environ.get("BATCH_EVAL", "4"))
GRAD_ACCUM  = int(os.environ.get("GRAD_ACCUM", "4"))

OUT_DIR_HF = HERE / "codet5_poc_model"
OUT_FILE_PT = HERE / "codet5_poc_weights.pt"
SAMPLES_FILE = HERE / "samples_pred.txt"

TRAIN_STEMS = HERE / "train_stems.txt"
VAL_STEMS   = HERE / "val_stems.txt"
TEST_STEMS  = HERE / "test_stems.txt"

random.seed(SEED)
torch.manual_seed(SEED)

def list_pairs() -> List[str]:
    c = {p.stem for p in DATA_DIR.glob("*.c")}
    s = {p.stem for p in DATA_DIR.glob("*.s")}
    stems = sorted(c & s)
    if not stems:
        raise RuntimeError(f"No (.s, .c) pairs found in {DATA_DIR}")
    return stems

def write_splits(stems: List[str]) -> None:
    random.shuffle(stems)
    n = len(stems)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train = stems[:n_train]
    val   = stems[n_train:n_train + n_val]
    test  = stems[n_train + n_val:]
    TRAIN_STEMS.write_text("\n".join(train) + ("\n" if train else ""), encoding="utf-8")
    VAL_STEMS.write_text("\n".join(val) + ("\n" if val else ""), encoding="utf-8")
    TEST_STEMS.write_text("\n".join(test) + ("\n" if test else ""), encoding="utf-8")

def read_list(path: pathlib.Path) -> List[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

def load_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

import re

def normalize_c_style(c_src: str) -> str:
    """
    Canonicalize brace placement and remove duplicated #include lines
    so the model learns one consistent style.
    """
    # Canonicalize brace position: )\n{ or ){ -> ) {
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

    # Combine headers + body, collapse blank lines
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
            items.append({"src": s_txt, "tgt": c_txt})
    return items

stems_all = list_pairs()
write_splits(stems_all)
train_stems = read_list(TRAIN_STEMS)
val_stems   = read_list(VAL_STEMS)
test_stems  = read_list(TEST_STEMS)

train_items = make_dataset_from_stems(train_stems)
val_items   = make_dataset_from_stems(val_stems)
test_items  = make_dataset_from_stems(test_stems)

if not train_items or not val_items:
    raise RuntimeError("Insufficient data after filtering. Check that you have *.s + *.c pairs.")

raw = DatasetDict(
    train=Dataset.from_list(train_items),
    validation=Dataset.from_list(val_items),
    test=Dataset.from_list(test_items) if test_items else Dataset.from_list([]),
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True)

def preprocess(batch):
    model_inputs = tokenizer(batch["src"], max_length=MAX_SRC_LEN, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["tgt"], max_length=MAX_TGT_LEN, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = raw.map(preprocess, batched=True, remove_columns=["src", "tgt"])
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
use_fp16 = torch.cuda.is_available() and not use_bf16

def build_training_args():
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())

    # Base kwargs common to most versions
    kw = {
        "output_dir": str(HERE / "codet5_poc_ckpt"),
        "num_train_epochs": EPOCHS,
        "per_device_train_batch_size": BATCH_TRAIN,
        "per_device_eval_batch_size": BATCH_EVAL,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LR,
        "warmup_ratio": 0.06,
        "weight_decay": 0.01,
        "logging_steps": 50,
    }
    if "report_to" in params:
        kw["report_to"] = "none"
    if "bf16" in params:
        kw["bf16"] = use_bf16
    if "fp16" in params:
        kw["fp16"] = use_fp16

    # Try to enable epoch eval/save + "keep only best" if this version supports it
    eval_key = "evaluation_strategy" if "evaluation_strategy" in params else ("eval_strategy" if "eval_strategy" in params else None)
    if eval_key is not None:
        kw[eval_key] = "epoch"
        if "save_strategy" in params:
            kw["save_strategy"] = "epoch"
        elif "save_steps" in params:
            # If only step-based saving exists, disable periodic saving to avoid mismatches
            kw["save_steps"] = 0

        # Best-model-at-end + metric if supported
        if "load_best_model_at_end" in params:
            kw["load_best_model_at_end"] = True
        if "metric_for_best_model" in params:
            kw["metric_for_best_model"] = "eval_loss"
        if "greater_is_better" in params:
            kw["greater_is_better"] = False
        if "save_total_limit" in params:
            kw["save_total_limit"] = 1  # ✅ keep only the best checkpoint
    else:
        # No evaluation scheduling in this version → disable best-at-end and periodic saving
        kw.pop("load_best_model_at_end", None)
        if "save_strategy" in params:
            kw["save_strategy"] = "no"
        elif "save_steps" in params:
            kw["save_steps"] = 0

    # Optional regularizer if available
    if "label_smoothing_factor" in params:
        kw["label_smoothing_factor"] = 0.1  # safe default for small data

    print(f"[TrainingArguments compat] using keys: {sorted(kw.keys())}")
    return TrainingArguments(**kw)
# ---- end builder ----

args = build_training_args()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

OUT_DIR_HF.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_DIR_HF, safe_serialization=True)
tokenizer.save_pretrained(OUT_DIR_HF)
torch.save(model.state_dict(), OUT_FILE_PT)

def normalize_for_eval(c_src: str) -> str:
    # same normalization, plus strip trailing whitespace for clean diffs
    return "\n".join(line.rstrip() for line in normalize_c_style(c_src).splitlines())

def generate_one(src_text: str) -> str:
    inputs = tokenizer(src_text, return_tensors="pt", truncation=True, max_length=MAX_SRC_LEN).to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=256, num_beams=4)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

samples = val_items[:5] if len(val_items) >= 1 else train_items[:5]
with open(SAMPLES_FILE, "w", encoding="utf-8") as f:
    for i, ex in enumerate(samples):
        pred = generate_one(ex["src"])
        f.write(f"\n=== SAMPLE {i} ===\n")
        f.write(normalize_for_eval(pred) + "\n")

print("\nDone.")
print(f"Saved HF model to: {OUT_DIR_HF}")
print(f"Saved raw weights to: {OUT_FILE_PT}")
print(f"Wrote samples to: {SAMPLES_FILE}")
print(f"Splits written to: {TRAIN_STEMS.name}, {VAL_STEMS.name}, {TEST_STEMS.name}")

