"""N5–N8 — the main grid. PINNED ENV ONLY. Writes runs/nvidia/ only (repo rule 1).

Frozen prereg b3f5fb3c…. One JSONL row per record, append-only, resumable: a row already
present for a dedup key is skipped, so an interrupted package restarts where it stopped without
regenerating anything. The dedup key is `harness/keys.py` and nothing else (repo rule 3).

Order is M2 complete across all its admitted budgets, then M3 — an interruption leaves one model
finished rather than two half-done.

Coverage is asserted after each package and holes are logged, never refilled mid-run.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import RecordKey, assert_owned_by, seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts

N_SINK, N_WINDOW = 8, 64
PREREG = "b3f5fb3c1e949c94e0785ba7a9843cbcc2548103215100fe0ec1d42b47ba886e"
LADDER_ARMS = ("full_cache", "null", "random", "floor_pos", "oracle_causal", "oracle_prescient")

ENV = dict(backend="cuda-12.8", transformers_version="5.2.0", kvpress_version="0.5.4",
           dtype="bfloat16")


def key_for(task, iid, model, rev, arm, B, protocol, seed):
    return RecordKey(task=task, instance_id=iid, model=model, model_revision=rev, arm=arm,
                     B=B, protocol=protocol, device="nvidia",
                     torch_version=torch.__version__, seed=seed, batch_size=1, **ENV)


def load_done(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["key_digest"])
                except Exception:
                    continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--task", default="ledger")
    ap.add_argument("--protocol", default="agnostic")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    print(f"  resuming: {len(done)} records already present in {out.name}")

    budgets = [int(x) for x in args.budgets.split(",")]
    arms = args.arms.split(",")
    t0 = time.time()
    written = skipped = failed = 0

    for i in range(args.n):
        iid = f"grid_{i:05d}"
        sd = seed_key_without_seed(
            task=args.task, instance_id=iid, model=args.model, model_revision=rev,
            arm="grid", B=1, protocol=args.protocol, device="nvidia",
            torch_version=torch.__version__, seed=0, **ENV)
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)
        posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]

        for C in budgets:
            B = C + N_SINK + N_WINDOW
            for arm in arms:
                k = key_for(args.task, iid, args.model, rev, arm, B, args.protocol, sd)
                assert_owned_by(k, "nvidia")
                if k.digest() in done:
                    skipped += 1
                    continue
                try:
                    if arm in LADDER_ARMS:
                        if arm == "oracle_prescient":
                            sc = []
                            for v, post in zip(inst.variants, posts):
                                gf = next(f for f in facts if f.fact_id == v.rec_id)
                                p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                                       n_window=N_WINDOW, facts=facts,
                                                       gold=gf, seed=sd)
                                o = generate_with(model, tok, pre, [post], p)
                                sc.append(1.0 if v.answer in o[0] else 0.0)
                            score = sum(sc) / len(sc)
                            outs = None
                        else:
                            p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                                   n_window=N_WINDOW, facts=facts, seed=sd)
                            outs = generate_with(model, tok, pre, posts, p)
                            score = ledger.score_instance(outs, inst)
                    else:
                        ratio = press._ratio_for(B, n_ctx)
                        p = methods.build_method(arm, ratio)
                        # Mandatory floors apply to EVERY arm (frozen PREREG §4.1).
                        p = methods.make_floor_constrained(p, n_ctx, N_SINK, N_WINDOW)
                        p.compression_ratio = ratio
                        outs = generate_with(model, tok, pre, posts, p)
                        score = ledger.score_instance(outs, inst)

                    row = {"key_digest": k.digest(), "key": k.as_dict(),
                           "score": score, "n_ctx": n_ctx, "C": C,
                           "per_variant": [1.0 if v.answer in o else 0.0
                                           for v, o in zip(inst.variants, outs)]
                           if outs else sc}
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    done.add(k.digest())
                    written += 1
                except Exception as e:
                    failed += 1
                    fp = out.with_suffix(".failures.jsonl")
                    with fp.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key_digest": k.digest(), "key": k.as_dict(),
                                            "error": f"{type(e).__name__}: {e}"[:400]}) + "\n")

        if (i + 1) % 10 == 0:
            el = time.time() - t0
            rate = written / el if el else 0
            print(f"  [{i+1}/{args.n}] written {written} skipped {skipped} failed {failed} "
                  f"({rate:.2f} rec/s, {el/60:.1f} min)", flush=True)

    # coverage assertion
    expected = args.n * len(budgets) * len(arms)
    have = len(load_done(out))
    print(f"\n  COVERAGE: expected {expected}, have {have}, holes {expected - have}, "
          f"failed {failed}")
    cov = {"model": args.model, "expected": expected, "have": have,
           "holes": expected - have, "failed": failed, "prereg_sha256": PREREG,
           "budgets": budgets, "arms": arms, "n": args.n}
    Path(str(out) + ".coverage.json").write_text(json.dumps(cov, indent=2) + "\n",
                                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
