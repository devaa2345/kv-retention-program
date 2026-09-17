"""Phase 1 iso-token at 8-bit on the second model, matched task, budgets 154/257/514, n=150."""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells
MODEL="Qwen/Qwen2.5-3B-Instruct"
N=int(sys.argv[1]) if len(sys.argv)>1 else 150
S0=int(sys.argv[2]) if len(sys.argv)>2 else 0
S1=int(sys.argv[3]) if len(sys.argv)>3 else N
model,tok=load(MODEL); eng=Engine(model,tok); eng.model_id=MODEL
cells=[dict(arm=a,budget=b,seed=s,iso_condition="iso_token",quant_bits=8,
            promotion="attention",context_target=1029,collect_dormancy=(a==1))
       for b in [154,257,514] for a in [1,2,3,4,5,6] for s in range(S0,S1)]
print(f"3B phase1 8-bit: {len(cells)} cells -> results3b/bits8", flush=True)
t0=time.time()
print(json.dumps(run_cells(eng,cells,"results3b/bits8","phase1_8bit",verbose_every=150),indent=2))
print("total %.1f min"%((time.time()-t0)/60))
