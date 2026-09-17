"""Follow-up 2 — a DIAGNOSTIC arm: contiguous retention matched on gold-token count.

*** POST-HOC. NOT IN THE FROZEN PREREG (b3f5fb3c…). A diagnostic, not a ladder arm, not a
    method, and not eligible for any G_m. ***

Question: is contiguity the operative variable, or token count? At C=512 on M2 the methods
retain MORE gold tokens than `floor_pos` (21.3–22.3 vs 19.5) and still score far lower. That is
suggestive but confounded: the arms differ in both count and contiguity.

This isolates it. `floor_matched` is a CONTIGUOUS block, grown until it holds at least as many
gold tokens as the method it is matched to. If a contiguous policy holding ~21 gold tokens beats
a fragmented policy holding ~22, contiguity is operative and token count is not.

**It deliberately breaks budget parity**, which is why it can never be a ladder arm: to match on
gold tokens the block must be sized freely, so its total `B'` differs from `B`. The realised
`B'` is reported for every cell so the cost of the match is visible rather than hidden. Read it
as "a contiguous policy needs B' tokens to hold the same gold as this method holds in B, and
here is what each then scores".
"""
from __future__ import annotations
import json, re, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts

M = "Qwen/Qwen2.5-3B-Instruct"
GRID = Path("runs/nvidia/grid_M2_ledger_agnostic.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
C = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]
N_SINK, N_WINDOW = 8, 64
REC = re.compile(r"^R(\d{3}) \| ")

acc = defaultdict(dict)
with GRID.open(encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        acc[(r["C"], r["key"]["arm"])][r["key"]["instance_id"]] = r["score"]

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"


def all_record_spans(inst, pre):
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]; base = pre.index(inst.context)
    out = {}
    for line in inst.context.split("\n"):
        m = REC.match(line)
        if not m or not (1 <= int(m.group(1)) <= ledger.N_RECORDS):
            continue
        a = inst.context.index(line) + base; b = a + len(line)
        out[m.group(0)] = {ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a}
    return out


@torch.inference_mode()
def gold_tokens_of(name, ratio, pre, n_ctx, recs, qids):
    cap = methods.Capture()
    p = methods.make_capturing(
        methods.make_floor_constrained(methods.build_method(name, ratio),
                                       n_ctx, N_SINK, N_WINDOW), cap)
    p.compression_ratio = ratio
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with p(model):
        model(**ids, use_cache=True)
    per = [sum(len(recs[q] & cap.per_head[k]) for q in qids if q in recs) for k in cap.heads()]
    return statistics.fmean(per)


def contiguous_keep(n_ctx, size):
    """floor_pos-shaped: sinks + the most recent `size` tokens of the compressible region."""
    floor, region = ladder._floors(n_ctx, N_SINK, N_WINDOW)
    return floor | set(region[-size:] if size < len(region) else region)


def match_size(n_ctx, recs, qids, target):
    """Smallest contiguous block whose gold-token count reaches `target`."""
    _, region = ladder._floors(n_ctx, N_SINK, N_WINDOW)
    lo, hi = 1, len(region)
    while lo < hi:
        mid = (lo + hi) // 2
        k = contiguous_keep(n_ctx, mid)
        g = sum(len(recs[q] & k) for q in qids if q in recs)
        if g >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


print(f"POST-HOC DIAGNOSTIC — not in the frozen prereg. {M}  C={C}  N={N}")
print(f"{'matched to':16s} {'method gold':>12s} {'method acc':>11s} | "
      f"{'contig gold':>12s} {'contig Bp':>10s} {'contig acc':>11s} | {'delta':>7s}")

for name in ARMS:
    mg, ma, cg, cb, ca = [], [], [], [], []
    for i in range(N):
        iid = f"grid_{i:05d}"
        sd = seed_key_without_seed(
            task="ledger", instance_id=iid, model=M, model_revision=rev, arm="grid", B=1,
            protocol="agnostic", device="nvidia", backend="cuda-12.8",
            torch_version=torch.__version__, transformers_version="5.2.0",
            kvpress_version="0.5.4", dtype="bfloat16")
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)
        posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
        recs = all_record_spans(inst, pre)
        qids = [v.rec_id + " | " for v in inst.variants]
        ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)

        tgt = gold_tokens_of(name, ratio, pre, n_ctx, recs, qids)
        size = match_size(n_ctx, recs, qids, tgt)
        keep = contiguous_keep(n_ctx, size)
        p = press.OracleCausalPress().set_keep(sorted(keep), n_ctx)
        outs = generate_with(model, tok, pre, posts, p)

        mg.append(tgt); ma.append(acc[(C, name)].get(iid, float("nan")))
        cg.append(sum(len(recs[q] & keep) for q in qids if q in recs))
        cb.append(len(keep)); ca.append(ledger.score_instance(outs, inst))

    ma_ = [x for x in ma if x == x]
    a, b = statistics.fmean(ma_), statistics.fmean(ca)
    print(f"{name:16s} {statistics.fmean(mg):12.2f} {a:11.4f} | "
          f"{statistics.fmean(cg):12.2f} {statistics.fmean(cb):10.1f} {b:11.4f} | {b - a:+7.4f}")
