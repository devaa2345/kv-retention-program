"""Generation determinism, M2 vs M3 — run ONLY after Stage 4's GPU packages have finished.

WHY. At Stage 4, oracle_prescient and oracle_causal were checked on the c~19 instances at
C in {128, 256, 512}, where all four candidates fit and the two oracles must hold the SAME
keep-set. On M2 they agree 100/100 in score and 100/100 byte-for-byte in generation. On M3,
on the matched instances, 6/58 scores and up to 14/58 generations differ -- while rebuilding the
keep-sets on CPU confirms they are IDENTICAL for every differing instance. So identical kept tokens
produced different text on Llama but not on Qwen. Stage 2's byte-identity evidence (176 duplicate
keys agreeing) was drawn entirely from M2, so M3's determinism was never actually verified.

The two oracle paths differ in exactly one mechanical way:
    causal     ONE prefill under the press, cache cloned, then all H=4 queries decoded
    prescient  a SEPARATE prefill per query, then that one query decoded

This separates two explanations, which have different consequences for the paper:
    (A) run-to-run nondeterminism -- the SAME path, run twice, differs. Then every M3 arm carries
        a noise floor and small between-arm differences on M3 are not resolvable.
    (B) path dependence only -- each path is self-consistent, but prefill-once-and-clone differs
        from prefill-per-query. Then only the two oracle arms are affected, and I(C) on M3 needs
        both oracles run through one path.

MEASURED, per model, on a fixed set of c~19 instances at C=512 with the causal keep-set:
    same_path_causal     causal path run twice               -> byte-identical generations?
    same_path_presc      prescient path run twice            -> byte-identical generations?
    cross_path           causal vs prescient, same keep-set  -> byte-identical generations?
    prefill_logit_diff   max |logit| difference, two independent prefills of the same input

Refuses to run while any stage4_run.py process exists (Machine N is single-process).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAGS = {"Qwen/Qwen2.5-3B-Instruct": "M2", "meta-llama/Llama-3.2-3B-Instruct": "M3"}
NL = chr(10)


def guard():
    r = subprocess.run(["pgrep", "-f", "stage4_run.py"], capture_output=True, text=True)
    if r.stdout.strip():
        raise SystemExit("ABORT: a stage4_run.py process is running; Machine N is single-process.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--C", type=int, default=512)
    args = ap.parse_args()
    guard()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from harness import ladder, press
    from p3 import keys3, runner
    from p3.tasks import ledger_c

    tag = TAGS[args.model]
    CAL = json.loads((HERE / "out" / "c_calibration.json").read_text(encoding="utf-8"))
    nf = CAL[tag]["chosen"]["19"]["n_fields"]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    ENV = dict(backend="cuda-12.8", transformers_version="5.2.0", kvpress_version="0.5.4",
               dtype="bfloat16", device="nvidia", batch_size=1, protocol="agnostic")
    C = args.C

    @torch.inference_mode()
    def prefill_logits(pre, p):
        ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
        with p(model):
            out = model(**ids, use_cache=True)
        return out.logits[0, -1].float()

    counts = dict(same_path_causal=0, same_path_presc=0, cross_path=0, total=0)
    logit_diffs = []
    for i in range(args.n):
        iid = "s4_%05d" % i
        sd = keys3.instance_seed(task="ledger_c", instance_id=iid, model=args.model,
                                 model_revision=rev, n_fields=nf, layout=None, n_records=1, **ENV)
        inst = ledger_c.build(sd, iid, n_fields=nf, target_tokens=2048, tokenizer=tok)
        pre, _ = runner.templated_parts(tok, inst.context, "")
        facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
        posts = [runner.templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
        mns = [len(tok(v.answer, add_special_tokens=False)["input_ids"]) + runner.MAX_NEW_SLACK
               for v in inst.variants]
        kept = ladder.oracle_causal(n_ctx, C, facts, n_sink=8, n_window=64).kept

        def causal_path():
            p = press.OracleCausalPress().set_keep(kept, n_ctx)
            return runner.generate_with(model, tok, pre, posts, p, mns)[0]

        def presc_path():
            outs = []
            for post, mn in zip(posts, mns):
                p = press.OracleCausalPress().set_keep(kept, n_ctx)   # SAME keep-set, per-query prefill
                outs.append(runner.generate_with(model, tok, pre, [post], p, [mn])[0][0])
            return outs

        c1, c2 = causal_path(), causal_path()
        p1, p2 = presc_path(), presc_path()
        for v in range(len(posts)):
            counts["total"] += 1
            counts["same_path_causal"] += c1[v] == c2[v]
            counts["same_path_presc"] += p1[v] == p2[v]
            counts["cross_path"] += c1[v] == p1[v]
        l1 = prefill_logits(pre, press.OracleCausalPress().set_keep(kept, n_ctx))
        l2 = prefill_logits(pre, press.OracleCausalPress().set_keep(kept, n_ctx))
        logit_diffs.append(float((l1 - l2).abs().max()))
        print("  %s %s  causal-rerun %d/4  presc-rerun %d/4  cross %d/4  max|dlogit| %.3e"
              % (tag, iid, sum(a == b for a, b in zip(c1, c2)),
                 sum(a == b for a, b in zip(p1, p2)), sum(a == b for a, b in zip(c1, p1)),
                 logit_diffs[-1]), flush=True)

    t = counts["total"]
    print(NL + "%s  C=%d  %d instances x 4 variants = %d generations" % (tag, C, args.n, t))
    print("  same path, causal rerun     byte-identical %d/%d" % (counts["same_path_causal"], t))
    print("  same path, prescient rerun  byte-identical %d/%d" % (counts["same_path_presc"], t))
    print("  cross path, same keep-set   byte-identical %d/%d" % (counts["cross_path"], t))
    print("  prefill logits, two runs    max |diff| %.3e" % max(logit_diffs))
    verdict = ("(A) RUN-TO-RUN NONDETERMINISM"
               if counts["same_path_causal"] < t or counts["same_path_presc"] < t
               else ("(B) PATH DEPENDENCE ONLY" if counts["cross_path"] < t
                     else "DETERMINISTIC and path-independent at this sample"))
    print("  -> %s" % verdict)
    json.dump(dict(model=tag, C=C, counts=counts, max_logit_diff=max(logit_diffs),
                   verdict=verdict),
              open(HERE / "out" / ("stage4_determinism_%s.json" % tag), "w", encoding="utf-8"),
              indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
