"""Solve n_fields per (model, k) so the achieved fact cost c lands near 19.

Holding n_fields fixed across k would confound the span count with fact cost, because every
extra part adds a record id and a part label. c is what must be held; n_fields is the knob.
"""
from __future__ import annotations
import json, re, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p3.tasks import multispan

MODELS = {"M2": "Qwen/Qwen2.5-3B-Instruct", "M3": "meta-llama/Llama-3.2-3B-Instruct"}
TARGET_C = 19.0
KS = (1, 2, 4)
REC = re.compile(r"^R(\d{3}) /")
OUT = Path(__file__).resolve().parents[1] / "out" / "k_calibration.json"


def templated_parts(tok, context, query=""):
    marker = "␟QUERY␟"
    full = tok.apply_chat_template([{"role": "user", "content": context + "\n\n" + marker}],
                                   tokenize=False, add_generation_prompt=True)
    pre, post = full.split(marker)
    return pre, query + post


def measure(tok, k, nf, seeds):
    cs, Ls, ans = [], [], []
    for sd, iid in seeds:
        inst = multispan.build(sd, iid, k=k, n_fields=nf, target_tokens=2048, tokenizer=tok)
        pre, _ = templated_parts(tok, inst.context)
        enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]; Ls.append(len(off))
        base = pre.index(inst.context)
        by_rec = {}
        for line in inst.context.split("\n"):
            m = REC.match(line)
            if not m:
                continue
            a = inst.context.index(line) + base; b = a + len(line)
            n = sum(1 for (x, y) in off if y > x and x < b and y > a)
            by_rec[m.group(1)] = by_rec.get(m.group(1), 0) + n
        cs.extend(by_rec.values())
        ans.extend(len(tok(v.answer, add_special_tokens=False)["input_ids"])
                   for v in inst.variants)
    return st.fmean(cs), st.fmean(Ls), st.fmean(ans)


def main():
    from transformers import AutoTokenizer
    seeds = [(2000 + i, "kcal_%05d" % i) for i in range(6)]
    out = {}
    for tag, M in MODELS.items():
        tok = AutoTokenizer.from_pretrained(M)
        out[tag] = {"model": M, "chosen": {}}
        print("\n### %s  %s" % (tag, M))
        for k in KS:
            best = None
            print("  k=%d" % k)
            for nf in range(k, 14):
                c, L, a = measure(tok, k, nf, seeds)
                print("    n_fields=%2d  c=%6.2f  L=%7.1f  answer=%5.1f" % (nf, c, L, a))
                if best is None or abs(c - TARGET_C) < abs(best[1] - TARGET_C):
                    best = (nf, c, L, a)
                if c > TARGET_C + 8:
                    break
            out[tag]["chosen"][str(k)] = dict(n_fields=best[0], c=best[1], L=best[2],
                                              answer_tokens=best[3])
            print("    -> k=%d: n_fields=%d achieved c=%.2f (target %.0f)"
                  % (k, best[0], best[1], TARGET_C))
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
