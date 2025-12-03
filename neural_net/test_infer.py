#!/usr/bin/env python3
import pathlib, random, re, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = pathlib.Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "training_data").resolve()
MODEL_DIR = HERE / "codet5_poc_model"

# ------- small helpers -------
def clean_asm(text: str, max_lines: int = 200) -> str:
    out = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t: 
            continue
        if t.startswith("."):  # drop assembler directives
            continue
        out.append(t)
        if len(out) >= max_lines:
            break
    return "\n".join(out)

def tidy_c(src: str) -> str:
    # 1) Drop obvious garbage fragments (e.g., lines ending with just '<')
    src = re.sub(r"^<\s*$", "", src, flags=re.M)

    # 2) Normalize brace style so formatting differences don't matter
    src = re.sub(r"\)\s*\n\s*\{", r") {", src)   # )\n{  -> ) {
    src = re.sub(r"\)\s*\{",     r") {", src)

    # 3) Deduplicate #include lines and cap how many we keep
    includes = []
    kept = []
    for ln in src.splitlines():
        if ln.strip().startswith("#include"):
            norm = re.sub(r"\s+", " ", ln.strip())
            if norm not in includes:
                includes.append(norm)
            continue
        kept.append(ln)
    # Keep only a small, sensible set if too many
    # (You can tune this list; it’s just to avoid degenerate spam)
    prefer = ["<stdio.h>", "<stdlib.h>", "<string.h>", "<ctype.h>", "<math.h>"]
    ordered = []
    # keep preferred ones in order if present
    for p in prefer:
        for inc in includes:
            if p in inc and inc not in ordered:
                ordered.append(inc)
    # then any remaining, up to a cap
    for inc in includes:
        if inc not in ordered:
            ordered.append(inc)
    ordered = ordered[:8]  # hard cap

    # 4) Reassemble with includes at the top (if any)
    body = "\n".join(kept).strip()
    header = "\n".join(ordered)
    out = (header + "\n\n" + body).strip() if ordered else body

    # 5) Remove duplicate blank lines
    out = re.sub(r"\n{3,}", "\n\n", out)

    # 6) Clip at the last balanced brace to avoid trailing junk
    bal = 0
    last_good = None
    for i, ch in enumerate(out):
        if ch == "{": bal += 1
        elif ch == "}":
            bal -= 1
            if bal == 0:
                last_good = i + 1
    if last_good is not None:
        out = out[:last_good].rstrip()

    return out

def generate(model, tokenizer, src: str) -> str:
    inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=384).to(model.device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=4,      # prevents short loops
            repetition_penalty=1.15,     # discourages repeats
            length_penalty=0.9,          # slightly favor finishing
            use_cache=True,
        )
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

# ------- run one sample -------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model.eval()

    stems = [ln.strip() for ln in open(HERE / "test_stems.txt", encoding="utf-8") if ln.strip()]
    if not stems:
        raise SystemExit("No stems in test_stems.txt")
    stem = random.choice(stems)  # or set stem = "Odd_word_problem"
    asm = (DATA_DIR / f"{stem}.s").read_text(encoding="utf-8", errors="ignore")
    src = clean_asm(asm)

    pred = generate(model, tokenizer, src)
    pred = tidy_c(pred)

    print(f"\n=== TEST SAMPLE: {stem} ===\n")
    print(pred)

if __name__ == "__main__":
    main()

