"""Solve `n_fields` per (model, target c), and MEASURE the achieved fact cost.

Fact cost is measured in the model's own tokenizer over the TEMPLATED prefix -- the same
convention Paper 2 used for every length, and the same span construction the fragmentation
dumps used. Nominal `c` is a target; the number that enters any analysis is the achieved one.

Also reports the context length `L` and the number of records, so a cell whose filler had to
collapse to zero to fit L is visible rather than silent.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p3.tasks import ledger_c

MODELS = {"M2": "Qwen/Qwen2.5-3B-Instruct", "M3": "meta-llama/Llama-3.2-3B-Instruct"}
TARGETS = (8, 19, 40)
REC = re.compile(r"^R(\d{3}) \| ")
OUT = Path(__file__).resolve().parents[1] / "out" / "c_calibration.json"


def templated_parts(tok, context: str, query: str):
    marker = "␟QUERY␟"
    full = tok.apply_chat_template([{"role": "user", "content": context + "\n\n" + marker}],
                                   tokenize=False, add_generation_prompt=True)
    pre, post = full.split(marker)
    return pre, query + post


def measure(tok, n_fields: int, seeds, target_tokens=2048):
    """Mean achieved fact cost c (tokens of a record line in the templated prefix) and L."""
    cs, Ls, ansl = [], [], []
    for sd, iid in seeds:
        inst = ledger_c.build(sd, iid, n_fields=n_fields,
                              target_tokens=target_tokens, tokenizer=tok)
        pre, _ = templated_parts(tok, inst.context, "")
        enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]
        Ls.append(len(off))
        base = pre.index(inst.context)
        for line in inst.context.split("\n"):
            m = REC.match(line)
            if not m or not (1 <= int(m.group(1)) <= ledger_c.N_RECORDS):
                continue
            a = inst.context.index(line) + base
            b = a + len(line)
            cs.append(sum(1 for (x, y) in off if y > x and x < b and y > a))
        for v in inst.variants:
            ansl.append(len(tok(v.answer, add_special_tokens=False)["input_ids"]))
    return st.fmean(cs), st.fmean(Ls), st.fmean(ansl), min(Ls), max(Ls)


def main() -> int:
    from transformers import AutoTokenizer
    seeds = [(1000 + i, f"cal_{i:05d}") for i in range(8)]
    out = {}
    for tag, M in MODELS.items():
        tok = AutoTokenizer.from_pretrained(M)
        # measure c for a range of n_fields, then pick the closest to each target
        table = {}
        for nf in range(1, 40):
            c, L, ansl, lo, hi = measure(tok, nf, seeds)
            table[nf] = dict(c=c, L=L, answer_tokens=ansl, L_min=lo, L_max=hi)
            if c > max(TARGETS) + 8:
                break
        print(f"\n### {tag}  {M}")
        print(f"{'n_fields':>9s} {'c (tokens)':>11s} {'answer tok':>11s} {'L mean':>8s} "
              f"{'L range':>14s}")
        for nf, r in table.items():
            print(f"{nf:9d} {r['c']:11.3f} {r['answer_tokens']:11.2f} {r['L']:8.1f} "
                  f"{('[%d, %d]' % (r['L_min'], r['L_max'])):>14s}")
        chosen = {}
        for t in TARGETS:
            nf = min(table, key=lambda k: abs(table[k]["c"] - t))
            chosen[str(t)] = dict(n_fields=nf, **table[nf])
            print(f"  target c={t:3d}  ->  n_fields={nf:2d}  achieved c={table[nf]['c']:.3f}  "
                  f"answer {table[nf]['answer_tokens']:.1f} tokens")
        out[tag] = dict(model=M, chosen=chosen,
                        table={str(k): v for k, v in table.items()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
