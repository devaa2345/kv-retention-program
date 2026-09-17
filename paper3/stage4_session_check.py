"""Which stored M3 oracle generation does a FRESH session reproduce?

WHY. At Stage 4, on M3 at c ~ 19 and C >= 128, oracle_prescient (anchors package) and
oracle_causal (plane package) disagree in some generations although their keep-sets were shown
identical on CPU. stage4_determinism.py then found M3 byte-identical within one session: same-path
reruns and the cross-path comparison all agreed, 48/48. The mismatching instances therefore
reproduce exactly inside a single process, so the disagreement lies BETWEEN the two stored
generation sessions. The causal rows for instances <= 50 were written before the power loss and
the prescient rows after it.

This regenerates, in one fresh session, both oracle arms on the instances whose stored generations
disagree. Each fresh generation is compared byte-for-byte with BOTH stored versions:

    fresh == stored causal  and  != stored prescient   -> the prescient session is the odd one
    fresh == stored prescient and != stored causal     -> the causal session is the odd one
    fresh matches neither                              -> cross-session nondeterminism
    fresh matches both                                  -> impossible while the two stored disagree

Paths are the ones Stage 4 used: oracle_causal through press.build_arm with one prefill and cloned
caches; oracle_prescient through press.build_arm with gold, one prefill per query. Fresh
generations are written to out/stage4_session_check_M3.json.

Refuses to run while any stage4_run.py or stage4_determinism.py process exists.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
NL = chr(10)


def guard():
    for name in ("stage4_run.py", "stage4_determinism.py"):
        r = subprocess.run(["pgrep", "-f", name], capture_output=True, text=True)
        if r.stdout.strip():
            raise SystemExit("ABORT: %s is running; Machine N is single-process." % name)


def jl(p):
    out = []
    for line in p.open(encoding="utf-8"):
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-instances", type=int, default=12)
    ap.add_argument("--budgets", default="512,256")
    args = ap.parse_args()
    guard()

    budgets = [int(x) for x in args.budgets.split(",")]
    pres = {(r["C"], r["key"]["instance_id"]): r for r in jl(RUNS / "stage4_anchors_M3.jsonl")
            if r["arm"] == "oracle_prescient"}
    cau = {(r["C"], r["key"]["instance_id"]): r for r in jl(RUNS / "stage4_plane_M3.jsonl")
           if r["arm"] == "oracle_causal" and 17 < r["c"] < 21}
    targets = []
    for C in budgets:
        ids = sorted(i for (CC, i), p in pres.items()
                     if CC == C and (C, i) in cau and p["gen"] != cau[(C, i)]["gen"])
        targets += [(C, i) for i in ids[:args.max_instances]]
    print("targets (C, instance): %d" % len(targets), flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from harness import press
    from p3 import runner
    from p3.tasks import ledger_c

    M = "meta-llama/Llama-3.2-3B-Instruct"
    CAL = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))
    nf = CAL["M3"]["chosen"]["19"]["n_fields"]
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(
        M, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

    tally = {"fresh==causal only": 0, "fresh==prescient only": 0, "neither": 0, "both": 0}
    by_arm = {"causal": [0, 0], "prescient": [0, 0]}      # [matches stored same-arm, total]
    rows = []
    for C, iid in targets:
        rc, rp = cau[(C, iid)], pres[(C, iid)]
        inst = ledger_c.build(rc["key"]["seed"], iid, n_fields=nf, target_tokens=2048,
                              tokenizer=tok)
        pre, _ = runner.templated_parts(tok, inst.context, "")
        facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
        assert n_ctx == rc["n_ctx"] == rp["n_ctx"], (iid, n_ctx)
        posts = [runner.templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
        mns = [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + runner.MAX_NEW_SLACK
               for v in inst.variants]

        p, _ = press.build_arm("oracle_causal", n_ctx=n_ctx, C=C, n_sink=8, n_window=64,
                               facts=facts, seed=rc["key"]["seed"])
        fresh_c, _ = runner.generate_with(model, tok, pre, posts, p, mns)
        fresh_p = []
        for v, post, mn in zip(inst.variants, posts, mns):
            gf = next(f for f in facts if f.fact_id == v.rec_id)
            pp, _ = press.build_arm("oracle_prescient", n_ctx=n_ctx, C=C, n_sink=8, n_window=64,
                                    facts=facts, gold=gf, seed=rp["key"]["seed"])
            fresh_p.append(runner.generate_with(model, tok, pre, [post], pp, [mn])[0][0])

        for v in range(len(posts)):
            sc, sp = rc["gen"][v], rp["gen"][v]
            by_arm["causal"][0] += fresh_c[v] == sc
            by_arm["causal"][1] += 1
            by_arm["prescient"][0] += fresh_p[v] == sp
            by_arm["prescient"][1] += 1
            if sc == sp:
                continue
            f = fresh_c[v]
            if f == sc and f == sp:
                tally["both"] += 1
            elif f == sc:
                tally["fresh==causal only"] += 1
            elif f == sp:
                tally["fresh==prescient only"] += 1
            else:
                tally["neither"] += 1
        rows.append(dict(C=C, instance_id=iid, fresh_causal=fresh_c, fresh_prescient=fresh_p,
                         stored_causal=rc["gen"], stored_prescient=rp["gen"]))
        print("  C=%d %s  fresh causal == stored causal %d/4   fresh prescient == stored "
              "prescient %d/4   fresh causal == fresh prescient %d/4"
              % (C, iid, sum(a == b for a, b in zip(fresh_c, rc["gen"])),
                 sum(a == b for a, b in zip(fresh_p, rp["gen"])),
                 sum(a == b for a, b in zip(fresh_c, fresh_p))), flush=True)

    print(NL + "Fresh session vs the two stored sessions, on generations where they disagree:")
    for k, v in tally.items():
        print("  %-24s %d" % (k, v))
    print("Fresh arm reproduces its own stored arm, over all generations checked:")
    for arm, (m, t) in by_arm.items():
        print("  %-10s %d / %d" % (arm, m, t))
    json.dump(dict(tally=tally, by_arm=by_arm, rows=rows),
              open(HERE / "out" / "stage4_session_check_M3.json", "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
