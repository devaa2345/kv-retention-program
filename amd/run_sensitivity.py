"""[SPEC-GAP 3] quant_byte_cost sensitivity, budget 257, arms 3 and 4.

§3 registers 0.25 but notes it ignores per-group scales and zero-points and that a real int8
scheme with bf16 scales is nearer 0.28-0.31, and instructs that the sensitivity be run. Changing
the constant changes the iso-memory token grant: 411 at 0.25, 402 at 0.28, 392 at 0.31.
"""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells
from kvre.cache_engine import iso_memory_tokens
COST=float(sys.argv[1]); N=int(sys.argv[2]) if len(sys.argv)>2 else 150
model,tok=load(); eng=Engine(model,tok)
cells=[dict(arm=a,budget=257,seed=s,iso_condition=f"iso_memory_c{COST}",quant_bits=4,
            promotion="attention",context_target=1029,quant_byte_cost=COST)
       for a in [3,4] for s in range(N)]
print(f"sensitivity cost={COST} -> T={iso_memory_tokens(257,0.5,COST)}, {len(cells)} cells", flush=True)
t0=time.time()
print(json.dumps(run_cells(eng,cells,"results/bits4",f"sens_c{COST}",verbose_every=100),indent=2))
print("total %.1f min"%((time.time()-t0)/60))
