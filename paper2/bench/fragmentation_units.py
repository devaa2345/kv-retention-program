"""Item 2: fragmentation with the completeness unit redefined as (id, value).

The old unit was the WHOLE record line `R011 | Surname | Dept | 280294`. But answering needs
only the **id** and the **6-digit value** -- the surname and department are dispensable. A policy
that keeps the high-entropy id and value while dropping the low-entropy middle was being scored
as "record not retained" when it is in fact functionally complete. That mis-scoring is what the
aware leak test surfaced (KeyDiff answered 43.6% of "non-retained" instances under aware).

Both units are computed on the same capture and printed side by side, so the size of the
correction is visible rather than asserted.

  LINE   = every token of the record line retained          (old unit)
  IDVAL  = every token of the id AND of the value retained  (new unit)
"""
from __future__ import annotations
import re, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, templated_parts

M = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
BUDGETS = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "16,32,64,128,256,512").split(",")]
ARMS = ["snapkv", "expected_attn", "keydiff", "adakv_snapkv"]
N_SINK, N_WINDOW = 8, 64

tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(
    M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
rev = getattr(model.config, "_commit_hash", None) or "x"
REC = re.compile(r"^R(\d{3}) \| ")


def record_units(inst, tok, pre):
    """Per record: (whole-line tokens, id-tokens | value-tokens)."""
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    base = pre.index(inst.context)
    line_u, idval_u = {}, {}
    for line in inst.context.split("\n"):
        m = REC.match(line)
        if not m or not (1 <= int(m.group(1)) <= ledger.N_RECORDS):
            continue
        a = inst.context.index(line) + base
        b = a + len(line)
        full = {ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a}
        # id  = "R011" at the head of the line;  value = the trailing 6 digits
        ia, ib = a, a + 4
        va, vb = b - 6, b
        idv = {ti for ti, (x, y) in enumerate(off)
               if y > x and ((x < ib and y > ia) or (x < vb and y > va))}
        line_u[m.group(0)] = full
        idval_u[m.group(0)] = idv
    return line_u, idval_u, len(off)


@torch.inference_mode()
def capture(name, ratio, pre, n_ctx):
    cap = methods.Capture()
    p = methods.make_capturing(
        methods.make_floor_constrained(methods.build_method(name, ratio),
                                       n_ctx, N_SINK, N_WINDOW), cap)
    p.compression_ratio = ratio
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with p(model):
        model(**ids, use_cache=True)
    return cap


GRID = {"Qwen/Qwen2.5-3B-Instruct": "runs/nvidia/grid_M2_ledger_agnostic.jsonl",
        "meta-llama/Llama-3.2-3B-Instruct": "runs/nvidia/grid_M3_ledger_agnostic.jsonl"}


def grid_accuracy(path):
    """mean accuracy per (C, arm) from the canonical agnostic grid on disk."""
    import json
    from collections import defaultdict
    acc = defaultdict(list)
    pp = Path(path)
    if not pp.exists():
        return {}
    for line in pp.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        acc[(r["C"], r["key"]["arm"])].append(r["score"])
    return {k: statistics.fmean(v) for k, v in acc.items()}


import json as _json
DUMP = Path("runs/nvidia") / ("frag_perinstance_%s.jsonl" %
                              ("M2" if "Qwen" in M else "M3"))
if N >= 50:                      # only the full runs write a dump
    DUMP.unlink(missing_ok=True)
    _dump = DUMP.open("a", encoding="utf-8")
else:
    _dump = None

ACC = grid_accuracy(GRID.get(M, ""))
print(f"  accuracy joined from {GRID.get(M)}: {len(ACC)} (C, arm) cells\n")

insts = []
for i in range(N):
    iid = f"grid_{i:05d}"
    sd = seed_key_without_seed(
        task="ledger", instance_id=iid, model=M, model_revision=rev, arm="grid", B=1,
        protocol="agnostic", device="nvidia", backend="cuda-12.8",
        torch_version=torch.__version__, transformers_version="5.2.0",
        kvpress_version="0.5.4", dtype="bfloat16")
    insts.append((ledger.build(sd, iid, target_tokens=2048, tokenizer=tok), sd))

print(f"{M}  N={N}  (per-head accounting; capture only, no generation)")
print(f"  unit sizes are printed once below, then per-cell results\n")
print(f"{'C':>5} {'arm':15s} | {'gold_tok':>9s} {'acc':>7s} | {'qCPL_LINE':>10s} "
      f"{'qCPL_IDVAL':>11s} {'delta':>8s} | {'rCPL_LINE':>10s} {'rCPL_IDVAL':>11s} "
      f"| {'touched':>8s} {'frag_LINE':>10s} {'frag_IDVAL':>11s}")

shown = False
for C in BUDGETS:
    agg = {}
    for inst, sd in insts:
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)
        lines, idvals, _ = record_units(inst, tok, pre)
        if not shown:
            k0 = next(iter(lines))
            print(f"  [unit sizes] record {k0!r}: line = {len(lines[k0])} tokens, "
                  f"(id,value) = {len(idvals[k0])} tokens\n")
            shown = True
        ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
        qids = [v.rec_id + " | " for v in inst.variants]

        def tally(keepsets):
            qL, qV, rL, rV, tch, gt, gv = [], [], [], [], [], [], []
            for kk in keepsets:
                h = len(qids)
                qL.append(sum(1 for q in qids if lines.get(q) and lines[q] <= kk) / h)
                qV.append(sum(1 for q in qids if idvals.get(q) and idvals[q] <= kk) / h)
                rL.append(sum(1 for s in lines.values() if s and s <= kk))
                rV.append(sum(1 for s in idvals.values() if s and s <= kk))
                tch.append(sum(1 for s in lines.values() if s & kk))
                # gold TOKENS retained -- the matched-count comparison depends on this
                gt.append(sum(len(lines[q] & kk) for q in qids if q in lines))
                gv.append(sum(len(idvals[q] & kk) for q in qids if q in idvals))
            return [statistics.fmean(x) for x in (qL, qV, rL, rV, tch, gt, gv)]

        def head_stats(keepsets):
            """Head-level analogue of the token-level record measure.

            Token level asks: of the records this arm TOUCHES, how many does it complete?
            Head level asks the same question across KV heads for the QUERIED record: of the
            heads that hold any of it, how many hold all of it? If the two are one mechanism,
            they should track each other and should not both survive when the other is
            controlled for.
            """
            H = max(1, len(keepsets))
            ht, hcL, hcV = [], [], []
            for q in qids:
                ln, iv = lines.get(q), idvals.get(q)
                if not ln:
                    continue
                ht.append(sum(1 for kk in keepsets if ln & kk))
                hcL.append(sum(1 for kk in keepsets if ln <= kk))
                hcV.append(sum(1 for kk in keepsets if iv and iv <= kk))
            f = statistics.fmean
            return dict(n_heads=H,
                        heads_touched=f(ht) if ht else 0.0,
                        heads_complete_line=f(hcL) if hcL else 0.0,
                        heads_complete_idval=f(hcV) if hcV else 0.0,
                        head_complete_frac_line=(f(hcL) / H) if hcL else 0.0,
                        head_complete_frac_idval=(f(hcV) / H) if hcV else 0.0,
                        head_frag_line=(f(ht) / max(1e-9, f(hcL))) if hcL else float("inf"),
                        head_frag_idval=(f(ht) / max(1e-9, f(hcV))) if hcV else float("inf"))

        def emit(name, keepsets):
            t = tally(keepsets)
            if _dump is not None:
                row = dict(model=M, instance_id=inst.instance_id, C=C, arm=name,
                           qcpl_line=t[0], qcpl_idval=t[1], recs_complete_line=t[2],
                           recs_complete_idval=t[3], recs_touched=t[4],
                           gold_tok_line=t[5], gold_tok_idval=t[6],
                           frag_line=t[4] / max(1e-9, t[2]),
                           frag_idval=t[4] / max(1e-9, t[3]))
                row.update(head_stats(keepsets))
                _dump.write(_json.dumps(row) + chr(10))
            return t

        fp = set(ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts).kept)
        agg.setdefault("floor_pos", []).append(emit("floor_pos", [fp]))
        for name in ARMS:
            cap = capture(name, ratio, pre, n_ctx)
            agg.setdefault(name, []).append(
                emit(name, [cap.per_head[k] for k in cap.heads()]))

    means = {}
    for name in ["floor_pos"] + ARMS:
        v = agg[name]
        qL, qV, rL, rV, t, gt, gv = (statistics.fmean(r[i] for r in v) for i in range(7))
        a = ACC.get((C, name))
        means[name] = dict(qL=qL, qV=qV, rL=rL, rV=rV, t=t, gt=gt, gv=gv, acc=a)
        print(f"{C:5d} {name:15s} | {gt:9.2f} {('%.4f' % a) if a is not None else '   n/a':>7s} "
              f"| {qL:10.4f} {qV:11.4f} {qV-qL:+8.4f} "
              f"| {rL:10.2f} {rV:11.2f} | {t:8.2f} "
              f"{t/max(1e-9,rL):10.2f} {t/max(1e-9,rV):11.2f}")

    # ---- the matched-gold-token comparison, under BOTH units ----------------
    # The paper's strongest single fact: an arm holding the SAME number of gold tokens as
    # floor_pos scores far below it. If IDVAL completeness explains the accuracy gap, the
    # method's qCPL_IDVAL should rise to meet floor_pos's; if it does not, the fact survives
    # the redefinition and fragmentation is about WHICH tokens, not how many.
    f = means["floor_pos"]
    print(f"\n  MATCHED-GOLD-TOKEN COMPARISON @ C={C}   "
          f"(floor_pos holds {f['gt']:.2f} gold tokens, acc "
          f"{('%.4f' % f['acc']) if f['acc'] is not None else 'n/a'})")
    print(f"    {'method':15s} {'gold_tok':>9s} {'d_gold':>8s} {'acc':>7s} {'acc ratio':>10s} "
          f"{'qCPL_LINE':>10s} {'qCPL_IDVAL':>11s} {'IDVAL closes gap?':>19s}")
    for name in ARMS:
        m = means[name]
        dg = m["gt"] - f["gt"]
        ratio = (m["acc"] / f["acc"]) if (m["acc"] is not None and f["acc"]) else float("nan")
        # does the IDVAL unit bring the method's completeness up to floor_pos's?
        gapL = f["qL"] - m["qL"]
        gapV = f["qV"] - m["qV"]
        if abs(gapL) < 1e-9:
            verdict = "n/a (no gap)"
        elif gapV < 0.25 * gapL:
            verdict = "YES -- gap mostly closes"
        elif gapV < 0.75 * gapL:
            verdict = "partly"
        else:
            verdict = "NO -- gap survives"
        flag = "  <-- MATCHED" if abs(dg) <= 0.5 else ""
        print(f"    {name:15s} {m['gt']:9.2f} {dg:+8.2f} "
              f"{('%.4f' % m['acc']) if m['acc'] is not None else '   n/a':>7s} "
              f"{ratio:10.2f} {m['qL']:10.4f} {m['qV']:11.4f} {verdict:>19s}{flag}")
    print()

if _dump is not None:
    _dump.close()
    print("  wrote per-instance dump: %s" % DUMP)
