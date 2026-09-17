"""Phase 1 tiered arms at 8-bit — tests the unmarked gap [GAP-W].

Spec section 7 fixes quant_bits=4 for PHASE 2 only; Phase 1's bit-width is never stated, while
section 3 calls quant_byte_cost=0.25 "the 8-bit scheme as originally registered". Arms 1 and 2
are permanent-eviction (n_quant=0) and therefore bit-width invariant, so only arms 3 and 4 are
re-run; the interaction is formed against the existing arm 1/2 rows.
"""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
model, tok = load(); eng = Engine(model, tok)
cells = [dict(arm=a, budget=b, seed=s, iso_condition="iso_token", quant_bits=8,
              promotion="attention", context_target=1029)
         for b in [154, 257, 514] for a in [3, 4] for s in range(N)]
print(f"phase1 8-bit: {len(cells)} cells -> results/bits8", flush=True)
t0=time.time()
info = run_cells(eng, cells, "results/bits8", "phase1_8bit", verbose_every=100)
print(json.dumps(info, indent=2)); print("total %.1f min"%((time.time()-t0)/60))
