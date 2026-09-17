"""Bit-width sweep on the second model: 8/7/6/5/4/3, n=50, budget 257, matched task."""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells
MODEL="Qwen/Qwen2.5-3B-Instruct"
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
WIDTHS=[int(x) for x in sys.argv[2].split(",")] if len(sys.argv)>2 else [8,7,6,5,4,3]
model,tok=load(MODEL); eng=Engine(model,tok); eng.model_id=MODEL
t0=time.time(); infos=[]
for w in WIDTHS:
    cells=[dict(arm=4,budget=257,seed=s,iso_condition="iso_token",quant_bits=w,
                promotion="attention",context_target=1029) for s in range(N)]
    print(f"3B sweep {w}-bit: {len(cells)} cells -> results3b/bits{w}", flush=True)
    infos.append(run_cells(eng,cells,f"results3b/bits{w}",f"sweep_bits{w}",verbose_every=25))
print(json.dumps(infos,indent=2)); print("total %.1f min"%((time.time()-t0)/60))
