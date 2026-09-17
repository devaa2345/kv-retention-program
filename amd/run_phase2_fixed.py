"""Phase 2 re-run with the corrected promotion signals (P4/P5 decontaminated).

Written to results/p2fix/ so it does not collide with the original run's cell keys; both are
retained so the effect of the contamination is itself measurable.
"""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells
from kvre.arms import PROMOTION_SIGNALS
N=int(sys.argv[1]) if len(sys.argv)>1 else 150
S0=int(sys.argv[2]) if len(sys.argv)>2 else 0
S1=int(sys.argv[3]) if len(sys.argv)>3 else N
model,tok=load(); eng=Engine(model,tok)
cells=[dict(arm=4,budget=257,seed=s,iso_condition="iso_token",quant_bits=4,
            promotion=sig,context_target=1029)
       for sig in PROMOTION_SIGNALS for s in range(S0,S1)]
print(f"phase2 FIXED signals: {len(cells)} cells -> results/p2fix", flush=True)
t0=time.time()
print(json.dumps(run_cells(eng, cells, "results/p2fix", "phase2_fixed", verbose_every=150), indent=2))
print("total %.1f min"%((time.time()-t0)/60))
