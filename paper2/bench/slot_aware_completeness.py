"""Items 1 and 2: re-do the completeness measures slot-aware, and see what survives.

Every completeness number in the fragmentation tables is aggregated over (layer, KV-head) slots
for method arms -- 72 on M2, 224 on M3 -- but EXACT for `floor_pos`, which has one global
keep-set. Three measures are recorded per record so the comparison can be made on equal terms:

    MEAN  = mean over slots of "complete in this slot"   <- what the tables currently use
    ANY   = complete in AT LEAST ONE slot                <- functional availability
    MAJ   = complete in > 50% of slots                   <- the leak test's predicate

For `floor_pos` all three coincide (one slot), which is exactly the asymmetry under test.

ANY is the measure motivated by evidence rather than convenience: the slot-fraction test showed
the value complete in only ~10% of slots among CORRECT answers (r = +0.50 with correctness), so a
record present in a minority of slots is frequently enough to answer. MEAN charges an arm for the
slots where the record is absent; ANY asks whether the model could reach it at all. Both are
reported; neither is asserted to be the right one.

Writes one row per (instance, C, arm) so items 1 and 2 are both pure re-analysis afterwards.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, templated_parts

N_SINK, N_WINDOW = 8, 64
ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]
REC = re.compile(r"^R(\d{3}) \| ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budgets", default="16,32,64,128,256,512")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "x"
    out = Path("runs/nvidia/slotaware_%s.jsonl" % args.tag)
    out.unlink(missing_ok=True)
    budgets = [int(b) for b in args.budgets.split(",")]

    for i in range(args.n):
        iid = "grid_%05d" % i
        sd = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev, arm="grid",
            B=1, protocol="agnostic", device="nvidia", backend="cuda-12.8",
            torch_version=torch.__version__, transformers_version="5.2.0",
            kvpress_version="0.5.4", dtype="bfloat16")
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)

        enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
        off = enc["offset_mapping"]
        base = pre.index(inst.context)
        lines, idvals = {}, {}
        for line in inst.context.split("\n"):
            m = REC.match(line)
            if not m or not (1 <= int(m.group(1)) <= ledger.N_RECORDS):
                continue
            a = inst.context.index(line) + base
            b = a + len(line)
            lines[m.group(0)] = {ti for ti, (x, y) in enumerate(off)
                                 if y > x and x < b and y > a}
            idvals[m.group(0)] = {ti for ti, (x, y) in enumerate(off)
                                  if y > x and ((x < a + 4 and y > a) or (x < b and y > b - 6))}
        qids = [v.rec_id + " | " for v in inst.variants]

        for C in budgets:
            ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
            sets = {"floor_pos": [set(
                ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts).kept)]}
            for arm in ARMS:
                cap = methods.Capture()
                p = methods.make_capturing(
                    methods.make_floor_constrained(
                        methods.build_method(arm, ratio), n_ctx, N_SINK, N_WINDOW), cap)
                p.compression_ratio = ratio
                ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
                with torch.inference_mode(), p(model):
                    model(**ids, use_cache=True)
                sets[arm] = [cap.per_head[k] for k in cap.heads()]

            for arm, ks in sets.items():
                if not ks:
                    continue
                S = len(ks)
                row = dict(model=args.model, tag=args.tag, instance_id=iid, C=C, arm=arm,
                           n_slots=S)
                for unit, spans in (("line", lines), ("idval", idvals)):
                    qm, qa, qj = [], [], []
                    for q in qids:
                        t = spans.get(q)
                        if not t:
                            continue
                        nc = sum(1 for kk in ks if t <= kk)
                        qm.append(nc / S)
                        qa.append(1.0 if nc >= 1 else 0.0)
                        qj.append(1.0 if (nc / S) > 0.5 else 0.0)
                    row["q_mean_" + unit] = statistics.fmean(qm) if qm else 0.0
                    row["q_any_" + unit] = statistics.fmean(qa) if qa else 0.0
                    row["q_maj_" + unit] = statistics.fmean(qj) if qj else 0.0
                    # all-records versions, for the fragmentation ratio
                    rm = ra = 0.0
                    for t in spans.values():
                        if not t:
                            continue
                        nc = sum(1 for kk in ks if t <= kk)
                        rm += nc / S
                        ra += 1.0 if nc >= 1 else 0.0
                    row["recs_mean_" + unit] = rm
                    row["recs_any_" + unit] = ra
                row["recs_touched_mean"] = statistics.fmean(
                    [sum(1 for t in lines.values() if t & kk) for kk in ks])
                row["recs_touched_any"] = float(
                    sum(1 for t in lines.values() if any(t & kk for kk in ks)))
                row["gold_tok_mean"] = statistics.fmean(
                    [sum(len(lines[q] & kk) for q in qids if q in lines) for kk in ks])
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        if (i + 1) % 20 == 0:
            print("  [%d/%d]" % (i + 1, args.n), flush=True)

    print("  wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
