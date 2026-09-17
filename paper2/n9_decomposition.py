"""N9 — the decomposition. PINNED ENV ONLY. Frozen prereg b3f5fb3c… §8.

Triggered by the main grid: a large residual sits where the design did not expect it (methods at
or below `floor_pos` in 51 of 54 cells), which is Outcome B/D territory.

Four constructed ceilings at matched budget, each relaxing exactly one constraint:

    A(floor_pos)                     position only
       | + Delta_selection           which tokens, globally, one set for all heads
    A(oracle_causal_global)
       | + Delta_head                per-head token sets, same total budget
    A(oracle_causal_perhead)
       | + Delta_temporal            plus knowing WHICH candidate is queried
    A(oracle_prescient)

Each Delta is paired per instance with a bootstrap CI, and the three sum to the total headroom
by construction.

**M3 is prioritised.** Per frozen PREREG §3.9(a), Delta_head is measured on M3/M4 only: M1 and
M2 have 2 KV heads, so there is almost nothing for head-wise allocation to express and the
contrast is structurally near-degenerate there. M2 is run for the other two deltas and its
Delta_head is reported as near-degenerate rather than as a measurement.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import press, stats
from harness.keys import RecordKey, assert_owned_by, seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts

N_SINK, N_WINDOW = 8, 64
PREREG = "b3f5fb3c1e949c94e0785ba7a9843cbcc2548103215100fe0ec1d42b47ba886e"
ENV = dict(backend="cuda-12.8", transformers_version="5.2.0", kvpress_version="0.5.4",
           dtype="bfloat16")
ARMS = ("floor_pos", "oracle_causal", "oracle_causal_perhead", "oracle_prescient")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    n_kv = int(getattr(model.config, "num_key_value_heads",
                       model.config.num_attention_heads))
    print(f"  {args.model}: {n_kv} KV heads")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try: done.add(json.loads(line)["key_digest"])
            except Exception: pass
    print(f"  resuming: {len(done)} records present")

    budgets = [int(x) for x in args.budgets.split(",")]
    t0 = time.time(); written = failed = 0
    scores = {(C, a): {} for C in budgets for a in ARMS}

    for i in range(args.n):
        iid = f"grid_{i:05d}"
        sd = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev,
            arm="grid", B=1, protocol="agnostic", device="nvidia",
            torch_version=torch.__version__, seed=0, **ENV)
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)
        posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]

        for C in budgets:
            B = C + N_SINK + N_WINDOW
            for arm in ARMS:
                k = RecordKey(task="ledger", instance_id=iid, model=args.model,
                              model_revision=rev, arm=arm, B=B, protocol="agnostic_n9",
                              device="nvidia", torch_version=torch.__version__,
                              seed=sd, batch_size=1, **ENV)
                assert_owned_by(k, "nvidia")
                if k.digest() in done:
                    continue
                try:
                    if arm == "oracle_prescient":
                        sc = []
                        for v, post in zip(inst.variants, posts):
                            gf = next(f for f in facts if f.fact_id == v.rec_id)
                            p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                                   n_window=N_WINDOW, facts=facts, gold=gf,
                                                   seed=sd, n_heads=n_kv)
                            o = generate_with(model, tok, pre, [post], p)
                            sc.append(1.0 if v.answer in o[0] else 0.0)
                        score = sum(sc) / len(sc)
                    else:
                        p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                               n_window=N_WINDOW, facts=facts, seed=sd,
                                               n_heads=n_kv)
                        score = ledger.score_instance(
                            generate_with(model, tok, pre, posts, p), inst)
                    scores[(C, arm)][iid] = score
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key_digest": k.digest(), "key": k.as_dict(),
                                            "score": score, "C": C, "n_kv_heads": n_kv}) + "\n")
                    done.add(k.digest()); written += 1
                except Exception as e:
                    failed += 1
                    with out.with_suffix(".failures.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key_digest": k.digest(), "key": k.as_dict(),
                                            "error": f"{type(e).__name__}: {e}"[:400]}) + "\n")
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{args.n}] written {written} failed {failed} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- deltas -----------------------------------------------------------
    # reload everything so a resumed run still analyses the full set
    allsc = {(C, a): {} for C in budgets for a in ARMS}
    for line in out.open(encoding="utf-8"):
        r = json.loads(line)
        allsc[(r["C"], r["key"]["arm"])][r["key"]["instance_id"]] = r["score"]

    rep = {"model": args.model, "n_kv_heads": n_kv, "n": args.n, "prereg_sha256": PREREG,
           "delta_head_measured": bool(n_kv >= 4), "cells": {}}
    print(f"\n{'C':>5} {'floor':>8} {'causal':>8} {'perhead':>8} {'presc':>8} | "
          f"{'D_select':>9} {'D_head':>9} {'D_temporal':>11}")
    for C in budgets:
        ids = sorted(set.intersection(*[set(allsc[(C, a)]) for a in ARMS]))
        if not ids:
            continue
        v = {a: [allsc[(C, a)][i] for i in ids] for a in ARMS}
        m = {a: sum(v[a]) / len(ids) for a in ARMS}
        ds = stats.paired_contrast(v["oracle_causal"], v["floor_pos"], seed=C, n_boot=20000)
        dh = stats.paired_contrast(v["oracle_causal_perhead"], v["oracle_causal"], seed=C,
                                   n_boot=20000)
        dt = stats.paired_contrast(v["oracle_prescient"], v["oracle_causal_perhead"], seed=C,
                                   n_boot=20000)
        rep["cells"][f"C{C}"] = {
            "C": C, "n": len(ids), "means": {a: round(m[a], 4) for a in ARMS},
            "delta_selection": {"mean": round(ds.mean_diff, 4),
                                "ci": [round(ds.ci_low, 4), round(ds.ci_high, 4)]},
            "delta_head": {"mean": round(dh.mean_diff, 4),
                           "ci": [round(dh.ci_low, 4), round(dh.ci_high, 4)],
                           "structurally_degenerate": bool(n_kv < 4)},
            "delta_temporal": {"mean": round(dt.mean_diff, 4),
                               "ci": [round(dt.ci_low, 4), round(dt.ci_high, 4)]},
            "sums_to_headroom": round(ds.mean_diff + dh.mean_diff + dt.mean_diff, 4),
            "headroom": round(m["oracle_prescient"] - m["floor_pos"], 4)}
        print(f"{C:5d} {m['floor_pos']:8.4f} {m['oracle_causal']:8.4f} "
              f"{m['oracle_causal_perhead']:8.4f} {m['oracle_prescient']:8.4f} | "
              f"{ds.mean_diff:+9.4f} {dh.mean_diff:+9.4f} {dt.mean_diff:+11.4f}")

    expected = args.n * len(budgets) * len(ARMS)
    rep["coverage"] = {"expected": expected, "have": len(done),
                       "holes": expected - len(done), "failed": failed}
    print(f"\n  COVERAGE: expected {expected} have {len(done)} "
          f"holes {expected - len(done)} failed {failed}")
    Path(str(out) + ".analysis.json").write_text(json.dumps(rep, indent=2) + "\n",
                                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
