"""Context length at fixed retention ratio ~13%: 2x2 factorial (arms 1-4) at 2048 and 4096.

[GAP-J]: recency_window stays fixed at 64 at every context length. Documented, not neutral.
Filler paragraph counts are calibrated per target length by calibrate_context.py.
"""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
CFG = json.load(open("results/context_calibration.json"))
model, tok = load(); eng = Engine(model, tok)
t0=time.time(); infos=[]
for entry in CFG["lengths"]:
    target, nf, budget = entry["target"], entry["n_filler"], entry["budget"]
    cells = [dict(arm=a, budget=budget, seed=s, iso_condition="iso_token", quant_bits=4,
                  promotion="attention", context_target=target, n_filler_paragraphs=nf,
                  recency_window=64, collect_dormancy=(a == 1))
             for a in [1,2,3,4] for s in range(N)]
    print(f"context {target}: budget={budget} n_filler={nf} cells={len(cells)}", flush=True)
    infos.append(run_cells(eng, cells, "results/bits4", f"context{target}", verbose_every=100))
print(json.dumps(infos, indent=2)); print("total %.1f min"%((time.time()-t0)/60))
