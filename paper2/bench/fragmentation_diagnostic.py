"""Fragmentation diagnostic — is the sub-floor deficit a real property or a harness fault?

The deficit survives the floor-parity fix and scales monotonically with budget (-0.03 at C=32
to -0.24 at C=512 on M2). A uniform harness offset would not have that shape. Hypothesis:

    methods retain a comparable or larger COUNT of gold tokens but fragment them across many
    records, while floor_pos retains fewer records INTACT.

A record line is only worth anything retained whole, so fragmenting is worth less than nothing.

Measured per arm, per budget, on identical instances (capture-only; no generation, so this does
not contend meaningfully with a running grid):

  gold_tok_kept     tokens of the H queried records retained
  queried_complete  P(the queried record is retained COMPLETE)   <- decides answerability
  queried_partial   P(retained but INCOMPLETE)                   <- the fragmentation signature
  queried_none      P(no tokens of it retained)
  recs_touched      distinct records (of all N) with >= 1 token retained
  recs_complete     distinct records retained whole
  frag_ratio        recs_touched / max(1, recs_complete)         <- higher = more fragmented

Per-head accounting throughout: head-wise presses allocate differently per head, and a global
union would hide exactly the behaviour under test.
"""
from __future__ import annotations
import re
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, templated_parts

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
BUDGETS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "32,64,128,256,512").split(",")]
ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]
N_SINK, N_WINDOW = 8, 64

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"
REC = re.compile(r"^R(\d{3}) \| ")


def all_record_spans(inst, tok, pre):
    """Token index sets for ALL N record lines (not just the H queried ones)."""
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    base = pre.index(inst.context)
    out = {}
    for line in inst.context.split("\n"):
        m = REC.match(line)
        if not m or not (1 <= int(m.group(1)) <= ledger.N_RECORDS):
            continue
        a = inst.context.index(line) + base
        b = a + len(line)
        out[m.group(0)] = {ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a}
    return out, len(off)


@torch.inference_mode()
def capture(name, ratio, pre, n_ctx):
    cap = methods.Capture()
    base = methods.make_floor_constrained(methods.build_method(name, ratio),
                                          n_ctx, N_SINK, N_WINDOW)
    p = methods.make_capturing(base, cap)
    p.compression_ratio = ratio
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with p(model):
        model(**ids, use_cache=True)
    return cap


insts = []
for i in range(N):
    sd = seed_key_without_seed(
        task="ledger", instance_id=f"grid_{i:05d}", model=M, model_revision=rev, arm="grid",
        B=1, protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")
    insts.append((ledger.build(sd, f"grid_{i:05d}", target_tokens=2048, tokenizer=tok), sd))

print(f"{M}  N={N}  (per-head accounting; capture only, no generation)")
print(f"{'C':>5} {'arm':15s} {'gold_tok':>9s} {'complete':>9s} {'partial':>8s} {'none':>7s} "
      f"{'touched':>8s} {'recs_cpl':>9s} {'frag':>6s}")

for C in BUDGETS:
    agg = {}
    for inst, sd in insts:
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)
        recs, _ = all_record_spans(inst, tok, pre)
        ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
        qids = [v.rec_id + " | " for v in inst.variants]

        def tally(keepsets):
            gt, cpl, par, non, tch, rcp = [], [], [], [], [], []
            for kk in keepsets:
                gt.append(sum(len(recs[q] & kk) for q in qids if q in recs))
                c = p_ = z = 0
                for q in qids:
                    s = recs.get(q, set())
                    n_in = len(s & kk)
                    if n_in == len(s) and s: c += 1
                    elif n_in > 0: p_ += 1
                    else: z += 1
                h = len(qids)
                cpl.append(c / h); par.append(p_ / h); non.append(z / h)
                tch.append(sum(1 for s in recs.values() if s & kk))
                rcp.append(sum(1 for s in recs.values() if s and s <= kk))
            return [statistics.fmean(x) for x in (gt, cpl, par, non, tch, rcp)]

        fp = set(ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts).kept)
        agg.setdefault("floor_pos", []).append(tally([fp]))
        for name in ARMS:
            cap = capture(name, ratio, pre, n_ctx)
            agg.setdefault(name, []).append(
                tally([cap.per_head[k] for k in cap.heads()]))

    for name in ["floor_pos"] + ARMS:
        v = agg[name]
        g, c, p_, z, t, rc = (statistics.fmean(r[i] for r in v) for i in range(6))
        print(f"{C:5d} {name:15s} {g:9.2f} {c:9.3f} {p_:8.3f} {z:7.3f} {t:8.2f} {rc:9.2f} "
              f"{t / max(1e-9, rc):6.1f}")
    print()
