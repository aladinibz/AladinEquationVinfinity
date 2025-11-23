# run_all_plots.py
# ALADIN ∞ ℂ(t) — One command = ALL plots generated

import os
from pathlib import Path

print("ALADIN ∞ ℂ(t) — Generating ALL Final Law plots...\n")

proofs = Path("proofs")
plots = Path("plots")
plots.mkdir(exist_ok=True)

files = sorted([f for f in proofs.glob("*.py") if not f.name.startswith("_")])

total = len(files)
done = 0

for i, f in enumerate(files, 1):
    print(f"[{i}/{total}] {f.name} → ", end="")
    try:
        exec(f.read_text())
        print("DONE")
        done += 1
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*60)
print(f"FINAL LAW COMPLETE — {done}/{total} plots generated")
print("All plots saved in plots/ folder")
print("="*60)
