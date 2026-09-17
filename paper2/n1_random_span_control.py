"""N1 — Stage 3 random-span control. THE LAST HARD GATE. PINNED ENV ONLY.

Runs against the VALIDATED Stage 5 kvpress ladder, under frozen prereg
`b3f5fb3c1e949c94e0785ba7a9843cbcc2548103215100fe0ec1d42b47ba886e`.

v1 §7, Stage 3:

    "Protect a random contiguous span of exactly `k_gold` length, at the same budget. Does
     protection behave as an *importance* signal or merely as *reserved capacity*?
     Gate: if random-span ~= gold-span protection, every `G_m` is measured against a
     misdescribed reference and the whole design is rebuilt before proceeding."

Why it matters: Paper 2's entire result is a ratio measured against reference arms that reserve
capacity for chosen tokens. If reserving *any* span were as good as reserving the *right* span,
then `floor_pos`, `oracle_causal` and `oracle_prescient` would not mean what the prereg says,
and every `G_m` would be normalised against a misdescribed denominator.

Three arms, identical budget `B`, identical instances, identical decoding:

    floor_pos        no protection                                  the registered denominator
    protect_gold     floor_pos + the queried record pinned          protection with the RIGHT tokens
    protect_random   floor_pos + a random contiguous span of        protection with ARBITRARY tokens
                     exactly k_gold tokens pinned

The random span is drawn from the compressible region, CRC32-seeded (repo rule 5), and is
resampled if it overlaps any candidate record — an accidental hit would measure luck rather
than capacity.

**Instance ids are `s5_XXXXX`, deliberately the same set Stage 5 validated on**, so the ladder
arms reported alongside are on the identical instances and the two tables are directly
comparable rather than merely adjacent.

PASS = protection is an importance signal: `protect_gold` − `protect_random` exceeds the
       pre-registered margin with a paired bootstrap CI excluding zero, at every budget.
FAIL = reserved capacity explains the effect. The design is rebuilt.

The margin is 0.15, fixed in advance — the same threshold PREREG §5 uses to decide a headroom
is wide enough to normalise against. It is not chosen after seeing the numbers.
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, press, stats
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts

N_SINK, N_WINDOW = 8, 64
MARGIN = 0.15


def random_span(n_ctx: int, length: int, forbidden: set[int], seed: int) -> set[int]:
    """A contiguous span of `length` tokens from the compressible region, never touching gold."""
    lo = N_SINK
    hi = n_ctx - N_WINDOW - length
    if hi <= lo:
        raise ValueError("no room for a random span in the compressible region")
    rng = random.Random(zlib.crc32(f"randspan|{seed}|{n_ctx}|{length}".encode()) & 0xFFFFFFFF)
    for _ in range(500):
        s = rng.randrange(lo, hi)
        span = set(range(s, s + length))
        if not (span & forbidden):
            return span
    raise RuntimeError("could not place a non-overlapping random span")


def pinned_keep(n_ctx: int, C: int, span: set[int]) -> list[int]:
    """floor_pos, with `span` pinned. Retains exactly B, same filler order as every other arm."""
    floor, region = ladder._floors(n_ctx, N_SINK, N_WINDOW)
    keep = floor | set(span)
    keep = ladder._pad_from_floor(keep, region, C - len(span & set(region)))
    return sorted(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--budgets", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    budgets = [int(x) for x in args.budgets.split(",")]

    insts = []
    for i in range(args.n):
        iid = f"s5_{i:05d}"                       # SAME instance set Stage 5 validated on
        seed = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev,
            arm="stage5", B=1, protocol="agnostic", device="nvidia", backend="cuda-12.8",
            torch_version=torch.__version__, transformers_version="5.2.0",
            kvpress_version="0.5.4", dtype="bfloat16")
        insts.append((ledger.build(seed, iid, target_tokens=2048, tokenizer=tok), seed))

    res: dict = {"model": args.model, "model_revision": rev, "n": args.n,
                 "margin": MARGIN, "prereg_sha256": "b3f5fb3c1e949c94e0785ba7a9843cbcc"
                                                    "2548103215100fe0ec1d42b47ba886e",
                 "instance_set": "s5_00000.. (identical to Stage 5)", "budgets": {}}

    print(f"{args.model}  n={args.n}")
    print(f"  {'C':>5} {'floor':>8} {'rand-span':>10} {'gold-span':>10} "
          f"{'gold-rand':>10} {'95% CI':>20} {'differ':>7}  gate")

    for C in budgets:
        per = {"floor_pos": [], "protect_random": [], "protect_gold": []}
        for inst, seed in insts:
            pre, _ = templated_parts(tok, inst.context, "")
            facts, n_ctx = facts_and_ctx(inst, tok, pre)
            posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
            all_cand = {t for f in facts for t in f.tokens}

            # arm 1 — no protection (one compressed cache, all H queries reuse it)
            p, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                   n_window=N_WINDOW, facts=facts, seed=seed)
            per["floor_pos"].append(
                ledger.score_instance(generate_with(model, tok, pre, posts, p), inst))

            # arms 2 and 3 — the pinned span differs per variant, so one cache per query
            gsc, rsc = [], []
            for j, (v, post) in enumerate(zip(inst.variants, posts)):
                gf = next(f for f in facts if f.fact_id == v.rec_id)
                k_gold = len(gf.tokens)

                pg = press.OracleCausalPress().set_keep(
                    pinned_keep(n_ctx, C, set(gf.tokens)), n_ctx)
                gsc.append(1.0 if v.answer in
                           generate_with(model, tok, pre, [post], pg)[0] else 0.0)

                span = random_span(n_ctx, k_gold, all_cand, seed + j)
                pr = press.OracleCausalPress().set_keep(pinned_keep(n_ctx, C, span), n_ctx)
                rsc.append(1.0 if v.answer in
                           generate_with(model, tok, pre, [post], pr)[0] else 0.0)
            per["protect_gold"].append(sum(gsc) / len(gsc))
            per["protect_random"].append(sum(rsc) / len(rsc))

        g = stats.paired_contrast(per["protect_gold"], per["protect_random"],
                                  name=f"gold-random@C{C}", seed=C, n_boot=20000)
        r = stats.paired_contrast(per["protect_random"], per["floor_pos"],
                                  name=f"random-floor@C{C}", seed=C, n_boot=20000)
        m = {k: sum(v) / len(v) for k, v in per.items()}
        passed = bool(g.mean_diff > MARGIN and g.ci_low > 0)
        res["budgets"][f"C{C}"] = {
            "C": C, "B": C + N_SINK + N_WINDOW,
            "means": {k: round(v, 4) for k, v in m.items()},
            "gold_minus_random": {"mean": round(g.mean_diff, 4),
                                  "ci": [round(g.ci_low, 4), round(g.ci_high, 4)],
                                  "n_differ": g.n_differ},
            "random_minus_floor": {"mean": round(r.mean_diff, 4),
                                   "ci": [round(r.ci_low, 4), round(r.ci_high, 4)],
                                   "n_differ": r.n_differ},
            "pass": passed}
        print(f"  {C:5d} {m['floor_pos']:8.4f} {m['protect_random']:10.4f} "
              f"{m['protect_gold']:10.4f} {g.mean_diff:+10.4f} "
              f"[{g.ci_low:+.4f},{g.ci_high:+.4f}] {g.n_differ:7d}  "
              f"{'PASS' if passed else '** FAIL **'}")

    res["gate"] = {
        "pass": all(v["pass"] for v in res["budgets"].values()),
        "criterion": f"protect_gold - protect_random > {MARGIN} with a paired bootstrap CI "
                     "excluding zero, at every admitted budget",
        "if_fail": "protection is reserved capacity, not an importance signal; every G_m would "
                   "be normalised against a misdescribed reference and the design is rebuilt",
    }
    print(f"\n  N1 GATE: {'PASS' if res['gate']['pass'] else '** FAIL **'}")

    tag = args.model.split("/")[-1].replace(".", "_")
    p = Path(__file__).resolve().parent / "gates" / "nvidia" / f"n1_random_span_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {p}")
    return 0 if res["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
