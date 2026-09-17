"""N8 — the query-AWARE calibration sub-grid. PINNED ENV ONLY. Writes runs/nvidia/ only.

Frozen prereg b3f5fb3c… §5.2. This is now load-bearing rather than optional: the headline
agnostic result is that methods lose to a recency floor, and the first objection is that SnapKV
is a query-aware method being run without its query. This answers it.

**Protocol difference.** Agnostic prefills the context alone, compresses, then appends the
question — so a press scoring by "attention from the last window" sees only filler. Aware
prefills context AND question together, so the press scores with the question in view. That is
the protocol the literature evaluates these methods under.

Cost consequence: the aware cache depends on the question, so it cannot be shared across the H
variants. Aware costs H prefills per instance-arm where agnostic costs one.

Reference deltas to check the harness against (PREREG §5.2): SnapKV's aware-arm gain ≈ +0.20,
KeyDiff's ≈ +0.01, with ordering SnapKV >> AdaKV > TOVA > ExpectedAttention > KeyDiff. If our
deltas reproduce that ordering the harness reproduces the field's own result; if not, that is a
harness problem and is reported as one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press
from harness.keys import RecordKey, assert_owned_by, seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import generate_with, templated_parts

N_SINK, N_WINDOW = 8, 64
MAX_NEW = 32
PREREG = "b3f5fb3c1e949c94e0785ba7a9843cbcc2548103215100fe0ec1d42b47ba886e"
LADDER_ARMS = ("full_cache", "null", "random", "floor_pos", "oracle_causal", "oracle_prescient")
ENV = dict(backend="cuda-12.8", transformers_version="5.2.0", kvpress_version="0.5.4",
           dtype="bfloat16")


def aware_parts(tok, context: str, query: str) -> tuple[str, str]:
    """(prefix carrying BOTH context and question, generation-prompt suffix)."""
    marker = "␟GEN␟"
    full = tok.apply_chat_template(
        [{"role": "user", "content": context + "\n\n" + query}],
        tokenize=False, add_generation_prompt=True)
    # split at the assistant header: everything before it is prefilled and compressed
    idx = full.rfind("<|im_start|>assistant")
    if idx < 0:
        idx = full.rfind("<|start_header_id|>assistant")
    if idx < 0:
        return full, ""
    return full[:idx], full[idx:]


def spans_over(inst, tok, pre):
    """Candidate token indices over the AWARE prefix (context + question)."""
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    base = pre.index(inst.context)
    by_id = {}
    for sp in inst.candidates:
        a, b = sp.start + base, sp.end + base
        by_id.setdefault(sp.rec_id, []).append(
            tuple(ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a))
    return [ladder.FactSpans(r, tuple(s)) for r, s in by_id.items()], len(off)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try: done.add(json.loads(line)["key_digest"])
            except Exception: pass
    print(f"  resuming: {len(done)} records present")

    budgets = [int(x) for x in args.budgets.split(",")]
    arms = args.arms.split(",")
    t0 = time.time(); written = skipped = failed = 0

    for i in range(args.n):
        iid = f"grid_{i:05d}"
        sd = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev,
            arm="grid", B=1, protocol="agnostic", device="nvidia",
            torch_version=torch.__version__, seed=0, **ENV)
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)

        for C in budgets:
            B = C + N_SINK + N_WINDOW
            for arm in arms:
                k = RecordKey(task="ledger", instance_id=iid, model=args.model,
                              model_revision=rev, arm=arm, B=B, protocol="aware",
                              device="nvidia", torch_version=torch.__version__,
                              seed=sd, batch_size=1, **ENV)
                assert_owned_by(k, "nvidia")
                if k.digest() in done:
                    skipped += 1; continue
                try:
                    sc = []
                    for v in inst.variants:
                        pre, post = aware_parts(tok, inst.context, v.query)
                        facts, n_ctx = spans_over(inst, tok, pre)
                        gf = next(f for f in facts if f.fact_id == v.rec_id)
                        if arm == "full_cache":
                            p = None
                        elif arm in LADDER_ARMS:
                            p, _ = press.build_arm(
                                arm, n_ctx=n_ctx, C=C, n_sink=N_SINK, n_window=N_WINDOW,
                                facts=facts, gold=gf, seed=sd)
                        else:
                            ratio = press._ratio_for(B, n_ctx)
                            p = methods.make_floor_constrained(
                                methods.build_method(arm, ratio), n_ctx, N_SINK, N_WINDOW)
                            p.compression_ratio = ratio
                        o = generate_with(model, tok, pre, [post], p)
                        sc.append(1.0 if v.answer in o[0] else 0.0)
                    score = sum(sc) / len(sc)
                    row = {"key_digest": k.digest(), "key": k.as_dict(), "score": score,
                           "C": C, "per_variant": sc}
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    done.add(k.digest()); written += 1
                except Exception as e:
                    failed += 1
                    with out.with_suffix(".failures.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"key_digest": k.digest(), "key": k.as_dict(),
                                            "error": f"{type(e).__name__}: {e}"[:400]}) + "\n")
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{args.n}] written {written} failed {failed} "
                  f"({written/max(el,1e-9):.2f} rec/s, {el/60:.1f} min)", flush=True)

    expected = args.n * len(budgets) * len(arms)
    print(f"\n  COVERAGE: expected {expected}, have {len(done)}, "
          f"holes {expected - len(done)}, failed {failed}")
    Path(str(out) + ".coverage.json").write_text(json.dumps(
        {"model": args.model, "protocol": "aware", "expected": expected, "have": len(done),
         "holes": expected - len(done), "failed": failed, "budgets": budgets, "arms": arms,
         "n": args.n, "prereg_sha256": PREREG}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
