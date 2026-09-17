"""Ablation: do distractors need to share the credentials' surface shape?

Section 2 says distractors are "of the same surface shape but non-credential labels". I read
that as sharing the full `sk-<14 hex>` value form ([GAP-V]). This ablation rewrites distractor
values to a visibly different form (`ref_<10 hex>`) and re-measures.

It tests TWO things at once, which is why it is worth running:

  1. Task difficulty / ceiling  -- arm 6. Do confusable distractor values depress the ceiling?
  2. What structural protection actually is -- arms 2 and 4. Structural protection matches
     lines by surface pattern, so when distractors STOP matching that pattern it silently
     degenerates into an oracle: it would protect only the 6 credential lines instead of
     competing across all 26. If arm 2 jumps toward arm 5 under this ablation, that is direct
     evidence the shared surface shape is what makes structural protection a non-trivial
     mechanism rather than a disguised oracle.

Does NOT modify kvre/task.py -- the alternate context is derived by post-processing, so the
main suite's task generator is untouched.
"""
import sys, re, json, hashlib, statistics; sys.path.insert(0, '/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.task import build_prompt, Prompt
from kvre.arms import make_cfg
from kvre.cache_engine import QuantAudit
from kvre.analysis import paired_bootstrap

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
ARMS = [2, 4, 6]
BUDGET = 257


def alt_prompt(seed: int) -> Prompt:
    """Same prompt, with distractor values rewritten to a non-credential surface form."""
    p = build_prompt(seed)
    credv = [c.value for c in p.credentials]          # list, not set; membership only
    def repl(m):
        v = m.group(0)
        if v in credv:
            return v
        return "ref_" + hashlib.sha256(v.encode()).hexdigest()[:10]
    ctx = re.sub(r"sk-[0-9a-f]{14}", repl, p.context)
    assert all(c.value in ctx for c in p.credentials), "credential value damaged by rewrite"
    assert ctx.count("sk-") == len(p.credentials), "distractor rewrite incomplete"
    return Prompt(seed=p.seed, context=ctx, credentials=p.credentials,
                  turn_order=p.turn_order)


def main():
    model, tok = load(); eng = Engine(model, tok)
    seeds = list(range(N))
    res = {"shared_shape": {}, "distinct_shape": {}}
    for arm in ARMS:
        for label, builder in (("shared_shape", build_prompt), ("distinct_shape", alt_prompt)):
            accs = []
            for s in seeds:
                r = eng.run_prompt(builder(s), make_cfg(arm, BUDGET), seed=s,
                                   audit=QuantAudit())
                accs.append(r["accuracy"])
            res[label][arm] = accs
            print(f"  arm{arm} {label:<15} mean={statistics.mean(accs):.4f} n={len(accs)}",
                  flush=True)

    print("\n=== paired effect of making distractors visually distinct ===")
    out = {}
    for arm in ARMS:
        a, b = res["distinct_shape"][arm], res["shared_shape"][arm]
        r = paired_bootstrap(a, b, f"arm{arm} distinct-vs-shared", n_boot=4000)
        out[f"arm{arm}"] = {"shared": r.mean_b, "distinct": r.mean_a, "diff": r.mean_diff,
                            "ci": list(r.mean_ci), "n_differ": r.n_differ, "p": r.p_perm}
        print(f"  arm{arm}: shared={r.mean_b:.4f} distinct={r.mean_a:.4f} "
              f"diff={r.mean_diff:+.4f} CI[{r.mean_ci[0]:+.4f},{r.mean_ci[1]:+.4f}] "
              f"ndiffer={r.n_differ} p={r.p_perm:.4f}", flush=True)
    json.dump(out, open("results/ablation_distractor.json", "w"), indent=2)
    print("\nwrote results/ablation_distractor.json")
    print("Interpretation: a large arm-2 gain indicates structural protection degenerates to an")
    print("oracle when distractors stop matching the protected line pattern; an arm-6 gain")
    print("indicates confusable distractor values were depressing the ceiling.")


if __name__ == "__main__":
    main()
