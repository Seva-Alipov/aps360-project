#!/usr/bin/env python3
import os
import subprocess
import pathlib
import shlex

# ---- Config ----
RISCV_GCC = os.environ.get("RISCV_GCC", "gcc")
EXTRA_FLAGS = ["-std=c99"]  # keep minimal; don't add -Werror
# ----------------

HERE = pathlib.Path(__file__).resolve().parent
TRAINING_DATA_DIR = (HERE / ".." / "training_data").resolve()
LOG_OK = HERE / "compiled_ok.txt"
LOG_FAIL = HERE / "failed_deleted.txt"

def is_safe_child(p: pathlib.Path, parent: pathlib.Path) -> bool:
    try:
        p.relative_to(parent)
        return True
    except ValueError:
        return False

def compile_one(c_path: pathlib.Path) -> tuple[bool, str]:
    asm_path = c_path.with_suffix(".s")
    cmd = [RISCV_GCC, "-O0", "-S", "-o", str(asm_path), str(c_path), *EXTRA_FLAGS]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and asm_path.exists():
            return True, f"{c_path.name} -> {asm_path.name}"
        else:
            # Clean up any partial output
            if asm_path.exists():
                try:
                    asm_path.unlink()
                except Exception:
                    pass
            # Return combined stderr (trim long outputs)
            err = (proc.stderr or "").strip()
            if len(err) > 2000:
                err = err[:2000] + " ... [truncated]"
            return False, err or "unknown compile error"
    except FileNotFoundError:
        return False, f"GCC not found: {RISCV_GCC}. Set RISCV_GCC env var."
    except Exception as e:
        return False, f"unexpected error: {e}"

def main():
    if not TRAINING_DATA_DIR.exists():
        print(f"ERROR: training data dir not found: {TRAINING_DATA_DIR}")
        return

    ok_lines = []
    fail_lines = []

    c_files = sorted(TRAINING_DATA_DIR.glob("*.c"))
    total = len(c_files)
    print(f"Found {total} C files in {TRAINING_DATA_DIR}")

    for idx, c_file in enumerate(c_files, 1):
        if not is_safe_child(c_file, TRAINING_DATA_DIR):
            print(f"[{idx}/{total}] SKIP (outside training_data): {c_file}")
            continue

        print(f"[{idx}/{total}] Compiling: {c_file.name}")
        success, info = compile_one(c_file)

        if success:
            ok_lines.append(f"{c_file} -> {c_file.with_suffix('.s')}")
            print(f"  OK")
        else:
            print(f"  FAIL: {info}")
            # Delete the failing source as requested
            try:
                c_file.unlink()
                fail_lines.append(f"{c_file} (deleted) :: {info.replace(os.linesep, ' ')}")
            except Exception as e:
                fail_lines.append(f"{c_file} (delete FAILED: {e}) :: {info.replace(os.linesep, ' ')}")

    LOG_OK.write_text("\n".join(ok_lines) + ("\n" if ok_lines else ""), encoding="utf-8")
    LOG_FAIL.write_text("\n".join(fail_lines) + ("\n" if fail_lines else ""), encoding="utf-8")

    print("\nDone.")
    print(f"  Compiled OK: {len(ok_lines)}   -> {LOG_OK}")
    print(f"  Failed+Deleted: {len([l for l in fail_lines if '(deleted)' in l])}   -> {LOG_FAIL}")
    if any("GCC not found" in l for l in fail_lines):
        print("Hint: export RISCV_GCC=/path/to/your/riscv64-unknown-elf-gcc")

if __name__ == "__main__":
    main()

