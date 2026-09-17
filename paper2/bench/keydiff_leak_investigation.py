"""Where does KeyDiff's aware accuracy come from when the gold record is NOT retained?

Established so far: under the aware protocol at C=512, KeyDiff answers 43.6% of instances
correctly on which the queried record was not fully retained (5.5% agnostic). `floor_pos` is
0.0000 in both protocols, so it is not a generic property of the harness. The id+value
hypothesis is DEAD -- redefining completeness as (id, value) moved KeyDiff's completeness by
+0.0006 to +0.0014, i.e. nothing.

Two candidates, tested here on M2 aware, C=512, n=200:

(a) THE VALUE, NOT THE RECORD. The answer is scored by substring match. If the queried record's
    6-digit VALUE tokens survive -- even with its id and the rest of the line evicted -- the
    model may still emit them. This is not the id+value hypothesis: that required BOTH id and
    value; this requires only the value, with no binding information at all. Sub-case: the model
    emits SEVERAL 6-digit numbers (a dump of whatever values survived) and the substring scorer
    counts a hit if the gold value is among them. That would make the 43.6% a scoring artefact,
    not retrieval. Both are measured: value-retention, and how many distinct 6-digit numbers the
    output contains.

(b) THE TEXT CARRIES IT. If the value appears anywhere in the prefix outside the queried
    record's own line -- exemplars, preamble, filler, the query, the chat template -- then it
    can be answered without the record at all. Checked by searching the full prefix for the
    answer string outside the gold span, and by checking the query and template directly.

If neither accounts for it, that is the finding and it is left open.
"""
from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import generate_with
from run_grid_aware import aware_parts, spans_over

M = "Qwen/Qwen2.5-3B-Instruct"
C = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
N_SINK, N_WINDOW = 8, 64
ARMS = ["keydiff", "snapkv", "floor_pos"]
SIX = re.compile(r"\d{6}")

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"


def value_span_tokens(pre, inst, rec_id):
    """Token indices of just the 6-digit VALUE of `rec_id` over the aware prefix."""
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    base = pre.index(inst.context)
    for line in inst.context.split("\n"):
        if line.startswith(rec_id + " | "):
            a = inst.context.index(line) + base
            b = a + len(line)
            va, vb = b - 6, b
            return {ti for ti, (x, y) in enumerate(off) if y > x and x < vb and y > va}
    return set()


res = {a: Counter() for a in ARMS}
acc_by = {a: {"val_ret": [], "val_not": []} for a in ARMS}
nnum = {a: [] for a in ARMS}
text_leak = Counter()
dup_value = Counter()
outs_sample = []

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
        vtok = value_span_tokens(pre, inst, v.rec_id)
        ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)

        # ---- (b) does the TEXT carry the answer outside the gold line? ----------
        gold_line = next((ln for ln in inst.context.split("\n")
                          if ln.startswith(v.rec_id + " | ")), "")
        outside = pre.replace(gold_line, "", 1)
        text_leak["answer_in_prefix_outside_gold_line"] += int(v.answer in outside)
        text_leak["answer_in_query"] += int(v.answer in v.query)
        text_leak["answer_in_template_tail"] += int(v.answer in post)
        text_leak["total"] += 1
        # duplicate values across records in the same instance
        dup_value["dup_value_in_context"] += int(inst.context.count(v.answer) > 1)

        for a in ARMS:
            cap = methods.Capture()
            if a == "floor_pos":
                p, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                       n_window=N_WINDOW, facts=facts, seed=sd)
            else:
                p = methods.make_capturing(
                    methods.make_floor_constrained(
                        methods.build_method(a, ratio), n_ctx, N_SINK, N_WINDOW), cap)
                p.compression_ratio = ratio
            out = generate_with(model, tok, pre, [post], p)[0]
            ok = 1.0 if v.answer in out else 0.0

            if a == "floor_pos":
                kept = set(ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts).kept)
                gold_ret = gold <= kept
                val_ret = bool(vtok) and vtok <= kept
            else:
                hs = cap.heads()
                gold_ret = statistics.fmean(
                    [1.0 if gold <= cap.per_head[k] else 0.0 for k in hs]) > 0.5 if hs else False
                val_ret = statistics.fmean(
                    [1.0 if (vtok and vtok <= cap.per_head[k]) else 0.0
                     for k in hs]) > 0.5 if hs else False

            res[a]["n"] += 1
            res[a]["correct"] += int(ok)
            if not gold_ret:                       # the population of interest
                res[a]["no_gold"] += 1
                res[a]["no_gold_correct"] += int(ok)
                acc_by[a]["val_ret" if val_ret else "val_not"].append(ok)
                nnum[a].append(len(set(SIX.findall(out))))
                if ok and a == "keydiff" and len(outs_sample) < 8:
                    outs_sample.append((v.answer, val_ret, out[:120]))

    if (i + 1) % 25 == 0:
        print("  [%d/%d]" % (i + 1, N), flush=True)

print("\n%s  aware  C=%d  N=%d instances x %d variants" % (M, C, N, len(inst.variants)))

print("\n(b) DOES THE TEXT CARRY THE ANSWER?  (n=%d prefixes)" % text_leak["total"])
for k in ("answer_in_prefix_outside_gold_line", "answer_in_query", "answer_in_template_tail"):
    print("    %-38s %d  (%.1f%%)"
          % (k, text_leak[k], 100.0 * text_leak[k] / max(1, text_leak["total"])))
print("    %-38s %d  (%.1f%%)"
      % ("duplicate value elsewhere in context", dup_value["dup_value_in_context"],
         100.0 * dup_value["dup_value_in_context"] / max(1, text_leak["total"])))

print("\n(a) CONDITIONAL ON GOLD NOT RETAINED: does the VALUE alone explain it?")
print("    %-10s %8s %10s %12s %12s %14s %14s"
      % ("arm", "n_no_gold", "acc", "n val_ret", "acc|val_ret", "n val_NOT", "acc|val_NOT"))
for a in ARMS:
    r = acc_by[a]["val_ret"]
    nr = acc_by[a]["val_not"]
    ar = statistics.fmean(r) if r else float("nan")
    an = statistics.fmean(nr) if nr else float("nan")
    ng = res[a]["no_gold"]
    print("    %-10s %8d %10.4f %12d %12.4f %14d %14.4f"
          % (a, ng, res[a]["no_gold_correct"] / max(1, ng), len(r), ar, len(nr), an))

print("\n    distinct 6-digit numbers emitted per answer (gold-not-retained population):")
for a in ARMS:
    v = nnum[a]
    if not v:
        continue
    print("      %-10s mean %.2f   max %d   %% with >1: %.1f%%"
          % (a, statistics.fmean(v), max(v), 100.0 * sum(1 for x in v if x > 1) / len(v)))

if outs_sample:
    print("\n    sample KeyDiff correct-without-gold outputs:")
    for ans, vr, o in outs_sample:
        print("      answer=%s value_retained=%s  out=%r" % (ans, vr, o))
