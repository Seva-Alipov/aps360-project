#!/usr/bin/env python3
import math
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- HARD-CODED LOCATIONS --------------------
HERE = Path(__file__).resolve().parent

# Folder containing your fine-tuned model (saved by your training script)
MODEL_DIR = Path.home() / "Documents/aps360-project/neural_net/codet5_poc_model"

# Files expected alongside this script
STEM = "write-entire-file"
ASM_PATH  = HERE / f"{STEM}.s"            # shared encoder input (assembly)
REF_C     = HERE / f"{STEM}.c"            # original C
NN_C      = HERE / f"{STEM}-nn.c"         # your model's C
GHIDRA_C  = HERE / f"{STEM}-ghidra.c"     # ghidra's C

# Sequence lengths (same ballpark as training)
MAX_SRC_LEN = 384
MAX_TGT_LEN = 256
# --------------------------------------------------------------

def normalize_c_style(c_src: str) -> str:
    """
    Canonicalize brace placement and deduplicate #include lines so style
    differences don't dominate loss comparisons.
    """
    # )\n{ and ){  ->  ) {
    c_src = re.sub(r"\)\s*\n\s*\{", r") {", c_src)
    c_src = re.sub(r"\)\s*\{",      r") {", c_src)

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
    ordered = ordered[:6]  # cap to avoid pathological include spam

    out = (("\n".join(ordered) + "\n\n") if ordered else "") + "\n".join(body)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def conditional_loss(model, tok, src_txt: str, tgt_txt: str) -> tuple[float, float]:
    """
    Compute the SAME loss as training: teacher-forced CE of target C
    conditioned on assembly input.
    """
    enc = tok(src_txt, return_tensors="pt", truncation=True, max_length=MAX_SRC_LEN).to(model.device)
    lab = tok(text_target=tgt_txt, return_tensors="pt", truncation=True, max_length=MAX_TGT_LEN)["input_ids"].to(model.device)
    with torch.no_grad():
        out = model(**enc, labels=lab)
    loss = float(out.loss.item())
    ppl = math.exp(loss) if loss < 20 else float("inf")
    return loss, ppl

def main():
    # Sanity checks for required files
    for p in [MODEL_DIR, ASM_PATH, REF_C, NN_C, GHIDRA_C]:
        if not p.exists():
            raise SystemExit(f"Missing required path: {p}")

    # Load tokenizer & model strictly from local disk (no Hub)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR), local_files_only=True).eval()

    # Device selection (HIP/CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Shared encoder input: the assembly (kept as-is)
    src_txt = read_text(ASM_PATH)

    # Targets with normalization ON (so brace/includes style doesn't skew)
    ref_txt = normalize_c_style(read_text(REF_C))
    nn_txt  = normalize_c_style(read_text(NN_C))
    gh_txt  = normalize_c_style(read_text(GHIDRA_C))

    # Training-style conditional loss (C | ASM)
    ref_loss, ref_ppl = conditional_loss(model, tokenizer, src_txt, ref_txt)
    nn_loss,  nn_ppl  = conditional_loss(model, tokenizer, src_txt, nn_txt)
    gh_loss,  gh_ppl  = conditional_loss(model, tokenizer, src_txt, gh_txt)

    print("\n=== Conditional training-style loss given assembly (lower is better) ===")
    print(f"Original : loss={ref_loss:.4f}  ppl={ref_ppl:.2f}")
    print(f"NN output: loss={nn_loss:.4f}  ppl={nn_ppl:.2f}")
    print(f"Ghidra   : loss={gh_loss:.4f}  ppl={gh_ppl:.2f}")

if __name__ == "__main__":
    main()

