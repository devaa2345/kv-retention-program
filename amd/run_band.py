"""Section 7 band check: all-FULL minus all-QUANT at fixed retention, budget 257, 4-bit.

The spec justifies quant_bits=4 empirically: "the band (all-FULL minus all-QUANT at fixed
retention) is non-degenerate there, measured at 0.278 for budget 257". This measures it.

Retention is held identical (protection ON, tiered arm); only full_fraction varies between
1.0 (every retained position FULL) and 0.0 (every retained position QUANT).
"""
import sys, json, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.task import build_prompt
from kvre.arms import make_cfg
from kvre.cache_engine import QuantAudit
from kvre.analysis import paired_bootstrap

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
BITS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
model, tok = load(); eng = Engine(model, tok)
res = {}
for ff, name in ((1.0, "all_FULL"), (0.0, "all_QUANT")):
    accs = []
    for s in range(N):
        cfg = make_cfg(4, 257, quant_bits=BITS, full_fraction=ff)
        accs.append(eng.run_prompt(build_prompt(s), cfg, seed=s, audit=QuantAudit())["accuracy"])
    res[name] = accs
    print(f"  full_fraction={ff:.1f} ({name}): mean={statistics.mean(accs):.4f} n={N}", flush=True)

r = paired_bootstrap(res["all_FULL"], res["all_QUANT"], "band", n_boot=10000)
print(f"\nBAND (all-FULL - all-QUANT) at budget 257, {BITS}-bit:")
print(f"  {r.mean_diff:+.4f}   95% CI [{r.mean_ci[0]:+.4f}, {r.mean_ci[1]:+.4f}]   "
      f"n_differ={r.n_differ}   p={r.p_perm:.4f}")
print(f"  spec reference: 0.278")
json.dump({"bits": BITS, "budget": 257, "n": N,
           "all_FULL": r.mean_a, "all_QUANT": r.mean_b, "band": r.mean_diff,
           "ci": list(r.mean_ci), "n_differ": r.n_differ, "p": r.p_perm,
           "reference": 0.278}, open(f"results/band_{BITS}bit.json", "w"), indent=2)
print(f"wrote results/band_{BITS}bit.json")
