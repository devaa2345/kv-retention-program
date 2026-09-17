"""Item 3: accuracy of the KeyDiff leak population at slot-fraction EXACTLY ZERO.

23.1% of the correct-without-gold population had the value complete in NO slot at all. That
subgroup's accuracy is the number that decides whether anything is left to explain:

    ~0.42  -> something real remains; the predicate was not the whole story
    ~0.00  -> the leak dissolves into the predicate

Also reported by finer buckets, and for "any value token present at all" (a weaker notion than
"value complete"), because a value split across tokens could be partially present in a slot that
holds none of it completely.

Saves per-row values this time so no further re-run is needed to slice it differently.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import generate_with
from run_grid_aware import aware_parts, spans_over

M = "Qwen/Qwen2.5-3B-Instruct"
C = 512
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
N_SINK, N_WINDOW = 8, 64
OUT = Path("runs/nvidia/diag_keydiff_zero_subgroup.jsonl")

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"
OUT.unlink(missing_ok=True)

rows = []
for i in range(N):
    iid = "grid_%05d" % i
    sd = seed_key_without_seed(
        task="ledger", instance_id=iid, model=M, model_revision=rev, arm="grid", B=1,
        protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")
    inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
    for v in inst.variants:
        pre, post = aware_parts(tok, inst.context, v.query)
        facts, n_ctx = spans_over(inst, tok, pre)
        gold = {t for f in facts if f.fact_id == v.rec_id for t in f.tokens}
        enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]
        base = pre.index(inst.context)
        line = next((ln for ln in inst.context.split("\n")
                     if ln.startswith(v.rec_id + " | ")), "")
        a = inst.context.index(line) + base
        b = a + len(line)
        vtok = {ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > b - 6}

        ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
        cap = methods.Capture()
        p = methods.make_capturing(
            methods.make_floor_constrained(
                methods.build_method("keydiff", ratio), n_ctx, N_SINK, N_WINDOW), cap)
        p.compression_ratio = ratio
        out = generate_with(model, tok, pre, [post], p)[0]
        ok = 1.0 if v.answer in out else 0.0
        hs = cap.heads()
        if not hs:
            continue
        S = len(hs)
        r = dict(iid=iid, rec=v.rec_id, ok=ok, n_slots=S,
                 gfrac=sum(1 for k in hs if gold <= cap.per_head[k]) / S,
                 vfrac=sum(1 for k in hs if vtok and vtok <= cap.per_head[k]) / S,
                 vany=sum(1 for k in hs if vtok & cap.per_head[k]) / S,
                 vpart=statistics.fmean(
                     [len(vtok & cap.per_head[k]) / max(1, len(vtok)) for k in hs]))
        rows.append(r)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")
    if (i + 1) % 25 == 0:
        print("  [%d/%d]" % (i + 1, N), flush=True)

pop = [r for r in rows if r["gfrac"] <= 0.5]
print("\nKeyDiff M2 aware C=512   population (gold slot-fraction <= 0.5): n=%d  acc=%.4f"
      % (len(pop), statistics.fmean(r["ok"] for r in pop)))

z = [r for r in pop if r["vfrac"] == 0.0]
nz = [r for r in pop if r["vfrac"] > 0.0]
print("\n  *** THE DECIDING NUMBER ***")
print("    value complete in EXACTLY ZERO slots: n=%d (%.1f%% of population)  acc=%.4f"
      % (len(z), 100.0 * len(z) / len(pop), statistics.fmean(r["ok"] for r in z) if z else float("nan")))
print("    value complete in >= 1 slot:          n=%d (%.1f%%)                acc=%.4f"
      % (len(nz), 100.0 * len(nz) / len(pop),
         statistics.fmean(r["ok"] for r in nz) if nz else float("nan")))

print("\n  by value-complete slot fraction bucket:")
buckets = [(0.0, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
for lo, hi in buckets:
    v = [r for r in pop if (r["vfrac"] == 0.0 if hi == 0.0 else lo < r["vfrac"] <= hi)]
    if v:
        print("    %-14s n=%-5d acc=%.4f"
              % (("exactly 0" if hi == 0.0 else "(%.2f, %.2f]" % (lo, hi)), len(v),
                 statistics.fmean(r["ok"] for r in v)))

print("\n  among the exactly-zero subgroup, is the value PARTIALLY present?")
if z:
    print("    mean fraction of slots holding >=1 value token: %.4f"
          % statistics.fmean(r["vany"] for r in z))
    print("    mean share of value tokens present per slot:    %.4f"
          % statistics.fmean(r["vpart"] for r in z))
    zz = [r for r in z if r["vany"] == 0.0]
    print("    subgroup with NO value token in ANY slot: n=%d  acc=%s"
          % (len(zz), ("%.4f" % statistics.fmean(r["ok"] for r in zz)) if zz else "n/a"))
print("\n  wrote %s" % OUT)
