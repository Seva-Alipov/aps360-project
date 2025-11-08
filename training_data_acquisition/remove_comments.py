#!/usr/bin/env python3
import re
import pathlib

TRAINING_DATA_DIR = (pathlib.Path(__file__).resolve().parent / ".." / "training_data").resolve()

def remove_comments(code: str) -> str:
    pattern = r"""
        ("(?:\\.|[^"\\])*") |   # double-quoted strings
        ('(?:\\.|[^'\\])*') |   # single-quoted chars
        (//[^\n]*$) |            # single-line comments
        (/\*.*?\*/)|             # multi-line comments
        (\r)                     # carriage returns (optional cleanup)
    """
    def replacer(m):
        if m.group(1) or m.group(2):  # keep strings and chars intact
            return m.group(0)
        return ""
    return re.sub(pattern, replacer, code, flags=re.MULTILINE | re.DOTALL | re.VERBOSE)

def main():
    count = 0
    for c_file in TRAINING_DATA_DIR.glob("*.c"):
        try:
            src = c_file.read_text(encoding="utf-8", errors="ignore")
            cleaned = remove_comments(src)
            if cleaned != src:
                c_file.write_text(cleaned, encoding="utf-8")
                count += 1
        except Exception as e:
            print(f"Error cleaning {c_file.name}: {e}")
    print(f"Removed comments from {count} files in {TRAINING_DATA_DIR}")

if __name__ == "__main__":
    main()

