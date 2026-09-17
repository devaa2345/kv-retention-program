"""Bit-width sweep: n=50, widths 8/7/6/5/4/3, budget 257, tiered+protected arm."""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEEDS = list(range(N)); WIDTHS = [8,7,6,5,4,3]
model, tok = load(); eng = Engine(model, tok)
t0=time.time(); infos=[]
for w in WIDTHS:
    out = f"results/bits{w}"          # spec section 6: different bit-widths, different dirs
    cells = [dict(arm=4, budget=257, seed=s, iso_condition="iso_token", quant_bits=w,
                  promotion="attention", context_target=1029) for s in SEEDS]
    print(f"sweep {w}-bit: {len(cells)} cells -> {out}", flush=True)
    infos.append(run_cells(eng, cells, out, f"sweep_bits{w}", verbose_every=25))
print(json.dumps(infos, indent=2)); print("total %.1f min"%((time.time()-t0)/60))
