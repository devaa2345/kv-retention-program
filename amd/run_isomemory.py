"""iso-memory condition (spec §6) + [SPEC-GAP 3] quant_byte_cost sensitivity.

Tiered arms get more raw positions, funded by the cold tier's lower byte cost, so total bytes
match the permanent arms. T = budget / (f + (1-f)*cost). With f=0.5, cost=0.25 -> T = 1.6*budget.
Permanent arms (1,2) are unchanged by the condition, so their iso_token rows are reused.

Sensitivity: §3 notes 0.25 ignores per-group scales and zero-points and that a real int8 scheme
is nearer 0.28-0.31, and instructs that the sensitivity be run. Done here at budget 257.
"""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells
from kvre.cache_engine import iso_memory_tokens

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
S0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
S1 = int(sys.argv[3]) if len(sys.argv) > 3 else N
model, tok = load(); eng = Engine(model, tok)
cells=[]
for b in [154,257,514]:                       # primary iso-memory at the registered 0.25
    for a in [3,4]:
        for s in range(S0,S1):
            cells.append(dict(arm=a,budget=b,seed=s,iso_condition="iso_memory",quant_bits=4,
                              promotion="attention",context_target=1029,quant_byte_cost=0.25))
for cost in [0.28,0.31]:                      # [SPEC-GAP 3] sensitivity, budget 257
    for a in [3,4]:
        for s in range(S0,S1):
            cells.append(dict(arm=a,budget=257,seed=s,iso_condition=f"iso_memory_c{cost}",
                              quant_bits=4,promotion="attention",context_target=1029,
                              quant_byte_cost=cost))
print("iso-memory token counts:", {b: iso_memory_tokens(b) for b in [154,257,514]}, flush=True)
print("sensitivity T @257:", {c: iso_memory_tokens(257,0.5,c) for c in [0.25,0.28,0.31]}, flush=True)
print(f"{len(cells)} cells -> results/bits4", flush=True)
t0=time.time()
print(json.dumps(run_cells(eng, cells, "results/bits4", "isomemory", verbose_every=150), indent=2))
print("total %.1f min"%((time.time()-t0)/60))
