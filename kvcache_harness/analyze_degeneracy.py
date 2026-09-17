"""Analysis for the bit-width degeneracy sweep: is the non-monotonic
accuracy curve a prompt-level threshold effect, and is the collapse one
basin or noise?
"""
from __future__ import annotations

import json
import re
from collections import Counter
from statistics import mean

PATH = "results/bitwidth_degeneracy/answers.jsonl"

rows = []
with open(PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

bits_list = sorted({r["bits"] for r in rows}, reverse=True)

print("=== per bit-width: accuracy vs DEGENERACY RATE (n=50 prompts x 6 turns) ===")
print(f"{'bits':>5s} {'n':>5s} {'acc':>7s} {'degen_rate':>11s} {'not_wellformed':>15s} {'repetitive':>11s} {'prompts_any_degen':>18s}")
for b in bits_list:
    sub = [r for r in rows if r["bits"] == b]
    if not sub:
        continue
    seeds = sorted({r["seed"] for r in sub})
    per_prompt_degen = []
    for s in seeds:
        answers = [r for r in sub if r["seed"] == s]
        per_prompt_degen.append(any(a["degenerate"] for a in answers))
    print(f"{b:5d} {len(sub):5d} {mean(r['correct'] for r in sub):7.3f} "
          f"{mean(r['degenerate'] for r in sub):11.3f} "
          f"{mean(not r['well_formed'] for r in sub):15.3f} "
          f"{mean(r['repetitive'] for r in sub):11.3f} "
          f"{mean(per_prompt_degen):18.3f}")

print()
print("=== is collapse PROMPT-DEPENDENT? (per-prompt degeneracy fraction, spread across prompts) ===")
for b in bits_list:
    sub = [r for r in rows if r["bits"] == b]
    seeds = sorted({r["seed"] for r in sub})
    fracs = []
    for s in seeds:
        a = [r for r in sub if r["seed"] == s]
        if a:
            fracs.append(mean(x["degenerate"] for x in a))
    if not fracs:
        continue
    allc = sum(1 for x in fracs if x == 1.0)
    none = sum(1 for x in fracs if x == 0.0)
    mixed = len(fracs) - allc - none
    print(f"  bits={b}: prompts fully degenerate={allc:3d}  none degenerate={none:3d}  mixed={mixed:3d}  (of {len(fracs)})")

print()
print("=== SAME BASIN? character profile of degenerate answers, per bit-width ===")
for b in bits_list:
    sub = [r for r in rows if r["bits"] == b and r["degenerate"]]
    if not sub:
        print(f"  bits={b}: no degenerate answers")
        continue
    chars = Counter()
    for r in sub:
        for ch in r["answer"]:
            chars[ch] += 1
    top = "".join(c for c, _ in chars.most_common(8))
    empty = sum(1 for r in sub if not r["answer"].strip())
    starts_sk = sum(1 for r in sub if r["answer"].strip().lower().startswith("sk"))
    print(f"  bits={b}: n_degen={len(sub):4d}  top_chars={top!r}  empty={empty:3d}  starts_'sk'={starts_sk:4d}")

print()
print("=== most common degenerate outputs (verbatim), 5-bit vs 6-bit vs 4-bit ===")
for b in [6, 5, 4]:
    sub = [r["answer"] for r in rows if r["bits"] == b and r["degenerate"]]
    if not sub:
        continue
    print(f"  --- bits={b} ---")
    for txt, n in Counter(sub).most_common(4):
        print(f"     {n:4d}x  {txt!r}")
