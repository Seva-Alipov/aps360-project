#!/usr/bin/env python3
import math, re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------- CONFIG -------------
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "codet5_poc_model"
VAL_LIST  = HERE / "val_stems.txt"
DATA_DIR  = HERE.parent / "training_data"

MAX_SRC_LEN = 384
MAX_TGT_LEN = 256
BATCH_SIZE  = 4

# Set this to True iff you normalized C during training
NORMALIZE_C = True
# ----------------------------------

def normalize_c_style(c_src: str) -> str:
    # Canonicalize ") {" placement
    c_src = re.sub(r"\)\s*\n\s*\{", r") {", c_src)
    c_src = re.sub(r"\)\s*\{",      r") {", c_src)
    # De-duplicate #include lines, prefer common ones, cap count
    includes, body = [], []
    for ln in c_src.splitlines():
        if ln.strip().startswith("#include"):
            key = " ".join(ln.strip().split())
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

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def load_val_pairs():
    pairs = []
    if not VAL_LIST.exists():
        raise SystemExit(f"Validation list not found: {VAL_LIST}")
    for stem in VAL_LIST.read_text().splitlines():
        stem = stem.strip()
        if not stem: continue
        asm = DATA_DIR / f"{stem}.s"
        c   = DATA_DIR / f"{stem}.c"
        if asm.exists() and c.exists():
            src = read(asm)                     # keep ASM as-is
            tgt = read(c)
            if NORMALIZE_C:
                tgt = normalize_c_style(tgt)    # match training
            pairs.append((src, tgt))
    if not pairs:
        raise SystemExit("No (.s, .c) validation pairs found.")
    return pairs

def batchify(data, bs):
    for i in range(0, len(data), bs):
        yield data[i:i+bs]

def main():
    if not MODEL_DIR.exists():
        raise SystemExit(f"Model directory not found: {MODEL_DIR}")

    print("[info] Loading model and tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR), local_files_only=True).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[info] Using device: {device}")

    val_pairs = load_val_pairs()
    print(f"[info] Loaded {len(val_pairs)} validation pairs. NORMALIZE_C={NORMALIZE_C}")

    total_loss, n_batches = 0.0, 0
    for batch in batchify(val_pairs, BATCH_SIZE):
        src_texts, tgt_texts = zip(*batch)
        enc = tok(list(src_texts), return_tensors="pt",
                  truncation=True, max_length=MAX_SRC_LEN, padding=True).to(device)
        labels = tok(text_target=list(tgt_texts), return_tensors="pt",
                     truncation=True, max_length=MAX_TGT_LEN, padding=True)["input_ids"].to(device)
        with torch.no_grad():
            out = model(**enc, labels=labels)
        total_loss += out.loss.item() * len(batch)
        n_batches += len(batch)

    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)
    print("\n=== Validation Metrics ===")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity:   {ppl:.2f}")

if __name__ == "__main__":
    main()

