"""Phase 1: iso-token at budgets 154/257/514, n=150, arms 1-6."""
import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.runner import run_cells

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
BUDGETS = [154, 257, 514]
SEEDS = list(range(N))
QBITS = 4
OUT = f"results/bits{QBITS}"          # different bit-widths -> different directories

model, tok = load(); eng = Engine(model, tok)
cells = []
for b in BUDGETS:
    for arm in [1, 2, 3, 4, 5, 6]:
        for s in SEEDS:
            cells.append(dict(arm=arm, budget=b, seed=s, iso_condition="iso_token",
                              quant_bits=QBITS, promotion="attention", context_target=1029,
                              collect_dormancy=(arm == 1)))
print(f"phase1: {len(cells)} cells -> {OUT}", flush=True)
t0 = time.time()
info = run_cells(eng, cells, OUT, "phase1_iso_token", verbose_every=100)
print(json.dumps(info, indent=2))
print("total %.1f min" % ((time.time()-t0)/60))
