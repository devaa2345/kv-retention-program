"""Replication of the 3B collapse/recovery widths on 50 unseen prompts (seeds 50-99)."""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells
MODEL="Qwen/Qwen2.5-3B-Instruct"
model,tok=load(MODEL); eng=Engine(model,tok); eng.model_id=MODEL
t0=time.time()
for w in [5,4,3]:
    cells=[dict(arm=4,budget=257,seed=s,iso_condition="iso_token",quant_bits=w,
                promotion="attention",context_target=1029) for s in range(50,100)]
    print(f"fresh {w}-bit: {len(cells)} cells", flush=True)
    run_cells(eng,cells,f"results3b/bits{w}",f"fresh_bits{w}",verbose_every=25)
print("total %.1f min"%((time.time()-t0)/60))
