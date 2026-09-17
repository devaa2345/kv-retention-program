"""N4a — per-method admission gates G2 and G3. PINNED ENV ONLY.

Frozen prereg b3f5fb3c…, §7. One method at a time; a method that fails either gate does not
enter the grid and is recorded in the unadmitted table with the failing gate. A failure never
stops the run.

  G2  permutation ablation
      Permuting the method's own score vector must destroy performance:
        (a) accuracy(permuted) must fall to the `random` arm's level, and
        (b) the retained sets of the original and permuted press must be uncorrelated —
            Spearman rho on their per-head indicator vectors ~ 0.
      A method statistically indistinguishable from its own permutation is measuring nothing.

  G3  oracle-overlap monotonicity
      Retained-set overlap with `oracle_causal` must be positive and increase with C.
      **Accounted per (layer, head)**, never as a global count: AdaKV and LU-KV spend different
      budgets on different heads by design, and a head-agnostic union would hide exactly the
      behaviour under test.
      Overlap is measured INSIDE THE COMPRESSIBLE REGION only. Both arms retain the sink and the
      recency window by construction, so including the floors would inflate every overlap toward
      1 and make the gate pass trivially.

Both gates generate tokens (§3.9(c)). Neither uses cache length or prefill logits, because
head-wise presses do not shrink the cache (AdaKV masks instead) and prefill logits are identical
at every ratio — the two documented false-negative traps.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, methods, press, stats
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts

N_SINK, N_WINDOW = 8, 64
PREREG = "b3f5fb3c1e949c94e0785ba7a9843cbcc2548103215100fe0ec1d42b47ba886e"


def region_of(n_ctx: int) -> set[int]:
    return set(range(N_SINK, n_ctx - N_WINDOW))


def run_method(model, tok, pre, posts, inst, name, C, n_ctx, seed, permuted=False):
    """Generate under a method press; return (score, capture)."""
    B = C + N_SINK + N_WINDOW
    ratio = press._ratio_for(B, n_ctx)
    p = methods.build_method(name, ratio)
    if permuted:
        p = methods.make_permuted(p, seed)
    # Mandatory floors apply to EVERY arm (frozen PREREG §4.1); applied AFTER the permutation
    # so the ablation permutes the method's own ranking, not the floors.
    p = methods.make_floor_constrained(p, n_ctx, N_SINK, N_WINDOW)
    cap = methods.Capture()
    p = methods.make_capturing(p, cap)
    p.compression_ratio = ratio
    outs = generate_with(model, tok, pre, posts, p)
    return ledger.score_instance(outs, inst), cap


def indicator(keep: set[int], region: list[int]) -> np.ndarray:
    s = set(keep)
    return np.fromiter((1.0 if t in s else 0.0 for t in region), dtype=float, count=len(region))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--methods", default=",".join(methods.TIER_A))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    budgets = [int(x) for x in args.budgets.split(",")]

    insts = []
    for i in range(args.n):
        iid = f"g_{i:05d}"
        sd = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev,
            arm="n4a", B=1, protocol="agnostic", device="nvidia", backend="cuda-12.8",
            torch_version=torch.__version__, transformers_version="5.2.0",
            kvpress_version="0.5.4", dtype="bfloat16")
        insts.append((ledger.build(sd, iid, target_tokens=2048, tokenizer=tok), sd))

    res: dict = {"model": args.model, "model_revision": rev, "n": args.n,
                 "prereg_sha256": PREREG, "budgets": budgets, "methods": {}}

    # random-arm reference per budget (G2's comparison target)
    rnd_ref = {}
    for C in budgets:
        sc = []
        for inst, sd in insts:
            pre, _ = templated_parts(tok, inst.context, "")
            facts, n_ctx = facts_and_ctx(inst, tok, pre)
            posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
            p, _ = press.build_arm("random", n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                   n_window=N_WINDOW, facts=facts, seed=sd)
            sc.append(ledger.score_instance(generate_with(model, tok, pre, posts, p), inst))
        rnd_ref[C] = sc
        print(f"  [ref] random arm @C={C}: {statistics.fmean(sc):.4f}")

    for name in args.methods.split(","):
        print(f"\n=== {name} ===")
        m: dict = {"budgets": {}, "errors": []}
        try:
            for C in budgets:
                orig, perm, ovl = [], [], []
                rhos = []
                for inst, sd in insts:
                    pre, _ = templated_parts(tok, inst.context, "")
                    facts, n_ctx = facts_and_ctx(inst, tok, pre)
                    posts = [templated_parts(tok, inst.context, v.query)[1]
                             for v in inst.variants]
                    region = sorted(region_of(n_ctx))

                    s_o, cap_o = run_method(model, tok, pre, posts, inst, name, C, n_ctx, sd)
                    s_p, cap_p = run_method(model, tok, pre, posts, inst, name, C, n_ctx, sd,
                                            permuted=True)
                    orig.append(s_o)
                    perm.append(s_p)

                    # G2(b): per-head Spearman between original and permuted retained sets
                    for key in cap_o.heads():
                        if key not in cap_p.per_head:
                            continue
                        a = indicator(cap_o.per_head[key], region)
                        b = indicator(cap_p.per_head[key], region)
                        if a.std() == 0 or b.std() == 0:
                            continue
                        rhos.append(float(spearmanr(a, b).statistic))

                    # G3: per-head overlap with oracle_causal, inside the region only
                    oc = ladder.oracle_causal(n_ctx, C, facts, n_sink=N_SINK,
                                              n_window=N_WINDOW)
                    ok = set(oc.kept) & set(region)
                    if ok:
                        for key in cap_o.heads():
                            ovl.append(len(cap_o.per_head[key] & ok) / len(ok))

                g2 = stats.paired_contrast(orig, perm, name=f"{name}-perm@C{C}", seed=C,
                                           n_boot=20000)
                m["budgets"][f"C{C}"] = {
                    "C": C,
                    "acc_original": round(statistics.fmean(orig), 4),
                    "acc_permuted": round(statistics.fmean(perm), 4),
                    "acc_random_arm": round(statistics.fmean(rnd_ref[C]), 4),
                    "orig_minus_perm": {"mean": round(g2.mean_diff, 4),
                                        "ci": [round(g2.ci_low, 4), round(g2.ci_high, 4)]},
                    "rho_retained_orig_vs_perm": round(statistics.fmean(rhos), 4) if rhos else None,
                    "oracle_overlap_per_head": round(statistics.fmean(ovl), 4) if ovl else None,
                }
                r = m["budgets"][f"C{C}"]
                print(f"  C={C:4d} orig {r['acc_original']:.4f} perm {r['acc_permuted']:.4f} "
                      f"rand {r['acc_random_arm']:.4f}  o-p {r['orig_minus_perm']['mean']:+.4f} "
                      f"{r['orig_minus_perm']['ci']}  rho {r['rho_retained_orig_vs_perm']}  "
                      f"ovl {r['oracle_overlap_per_head']}")

            # ---- gate decisions -------------------------------------------------
            bs = list(m["budgets"].values())
            # G2 has three parts, following the prereg wording literally:
            #   (a) the permutation must be a VALID ablation -- permuted accuracy falls to the
            #       random arm's level, at every budget;
            #   (b) rho between the original and permuted retained sets ~ 0, at every budget;
            #   (c) the original must be distinguishable from its own permutation SOMEWHERE in
            #       the admitted grid. "Statistically indistinguishable from its own
            #       permutation" means indistinguishable everywhere; a method that is inert at
            #       a tight budget but informative at a loose one is not measuring nothing, and
            #       requiring (c) at every budget would fail it for the wrong reason. Per-budget
            #       detail is recorded either way, so an inert cell stays visible.
            g2_ablation = all(b["acc_permuted"] <= b["acc_random_arm"] + 0.02 for b in bs)
            g2_rho = all((b["rho_retained_orig_vs_perm"] is None
                          or abs(b["rho_retained_orig_vs_perm"]) < 0.10) for b in bs)
            g2_pass = any(b["orig_minus_perm"]["ci"][0] > 0 for b in bs)
            g2_budgets_informative = [b["C"] for b in bs if b["orig_minus_perm"]["ci"][0] > 0]
            # G3: overlap positive everywhere and non-decreasing in C
            ovs = [b["oracle_overlap_per_head"] for b in bs]
            g3_pos = all(o is not None and o > 0 for o in ovs)
            g3_mono = all(ovs[i] <= ovs[i + 1] + 1e-9 for i in range(len(ovs) - 1)) if g3_pos else False
            m["G2"] = {"pass": bool(g2_pass and g2_rho and g2_ablation),
                       "beats_own_permutation_somewhere": bool(g2_pass),
                       "informative_at_budgets": g2_budgets_informative,
                       "permuted_falls_to_random_arm": bool(g2_ablation),
                       "rho_near_zero": bool(g2_rho)}
            m["G3"] = {"pass": bool(g3_pos and g3_mono), "overlap_positive": bool(g3_pos),
                       "monotone_in_C": bool(g3_mono), "overlaps": ovs}
            m["admitted"] = bool(m["G2"]["pass"] and m["G3"]["pass"])
        except Exception as e:
            m["errors"].append(f"{type(e).__name__}: {e}"[:300])
            m["admitted"] = False
            m["G2"] = {"pass": False}; m["G3"] = {"pass": False}
            print(f"  ERROR: {type(e).__name__}: {e}"[:200])
        res["methods"][name] = m
        print(f"  -> G2 {m['G2']['pass']}  G3 {m['G3']['pass']}  "
              f"ADMITTED={m['admitted']}")

    res["admitted"] = [k for k, v in res["methods"].items() if v["admitted"]]
    res["unadmitted"] = [k for k, v in res["methods"].items() if not v["admitted"]]
    print(f"\n  ADMITTED  : {res['admitted']}")
    print(f"  UNADMITTED: {res['unadmitted']}")

    tag = args.model.split("/")[-1].replace(".", "_")
    p = Path(__file__).resolve().parent / "gates" / "nvidia" / f"n4a_admission_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
