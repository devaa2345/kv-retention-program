"""Phase 2's five promotion signals at 8-bit — residual test for [GAP-W].

Spec §7 fixes Phase 2 at 4-bit, so this is NOT a spec-conforming run. It exists to test whether
the ~0.05-0.08 level shift between my Phase 2 values and the reference shares a cause with the
Phase 1 interaction disagreement, i.e. whether the reference's tiering was materially less
destructive than 4-bit.
"""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells
from kvre.arms import PROMOTION_SIGNALS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
model, tok = load(); eng = Engine(model, tok)
cells=[dict(arm=4,budget=257,seed=s,iso_condition="iso_token",quant_bits=8,
            promotion=sig,context_target=1029)
       for sig in PROMOTION_SIGNALS for s in range(N)]
print(f"phase2 8-bit: {len(cells)} cells -> results/bits8", flush=True)
t0=time.time()
info=run_cells(eng, cells, "results/bits8", "phase2_8bit", verbose_every=100)
print(json.dumps(info,indent=2)); print("total %.1f min"%((time.time()-t0)/60))
