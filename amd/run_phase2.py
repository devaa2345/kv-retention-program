"""Phase 2: 4-bit, budget 257, n=150, five promotion signals.
Retention held fixed and attention-ranked; protection ON; tiered eviction ON (spec section 7).
"""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells
from kvre.arms import PROMOTION_SIGNALS

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEEDS = list(range(N)); QBITS = 4; BUDGET = 257
OUT = f"results/bits{QBITS}"
model, tok = load(); eng = Engine(model, tok)
cells = []
for sig in PROMOTION_SIGNALS:
    for s in SEEDS:
        cells.append(dict(arm=4, budget=BUDGET, seed=s, iso_condition="iso_token",
                          quant_bits=QBITS, promotion=sig, context_target=1029))
print(f"phase2: {len(cells)} cells -> {OUT}", flush=True)
t0=time.time()
info = run_cells(eng, cells, OUT, "phase2_promotion", verbose_every=100)
print(json.dumps(info, indent=2)); print("total %.1f min"%((time.time()-t0)/60))
