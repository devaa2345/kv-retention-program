"""3B post-hoc: level occupancy and reconstruction error on tensors intercepted during LIVE
generation (not prefill), for the sweep table."""
import sys, json, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.task import build_prompt
from kvre.arms import make_cfg
from kvre.cache_engine import QuantAudit, relative_error
import kvre.cache_engine as ce
MODEL="/home/kxrx26/llama32_3b_instruct"
model,tok=load(MODEL); eng=Engine(model,tok); eng.model_id=MODEL
out={"model":MODEL}

print("=== level occupancy (3B) ===",flush=True)
hist={}
for bits in [8,7,6,5,4,3]:
    au=QuantAudit()
    for s in range(4):
        Engine(model,tok).run_prompt(build_prompt(s), make_cfg(4,257,quant_bits=bits), seed=s, audit=au)
    h=dict(sorted(au.level_histogram.items())); tot=sum(h.values()) or 1
    occ=sum(k*v for k,v in h.items())/tot
    hist[bits]={"limit":2**bits,"max_levels":au.max_levels_seen,"mean_levels_used":occ,
                "occupancy_frac":occ/(2**bits),"violations":len(au.violations)}
    print(f"  {bits}-bit: max={au.max_levels_seen:>4}/{2**bits:<4} mean_used={occ:7.2f} "
          f"occupancy={occ/(2**bits):.3f} viol={len(au.violations)}",flush=True)
out["level_occupancy"]=hist

print("\n=== monotone error on INTERCEPTED generation tensors (3B) ===",flush=True)
captured=[]; orig=ce.quantize_dequantize
def capturing(x,bits,return_codes=False,arith=None):
    if len(captured)<24: captured.append(x.detach().clone())
    return orig(x,bits,return_codes=return_codes,arith=arith)
ce.quantize_dequantize=capturing
try:
    for s in range(3):
        Engine(model,tok).run_prompt(build_prompt(s), make_cfg(4,257,quant_bits=4), seed=s, audit=QuantAudit())
        if captured: break
finally:
    ce.quantize_dequantize=orig
tbl={}
for bits in [8,7,6,5,4,3,2]:
    errs=[relative_error(t,orig(t,bits)) for t in captured]
    tbl[bits]=100.0*sum(errs)/len(errs)
    print(f"  {bits}-bit: {tbl[bits]:.4f}%",flush=True)
w=sorted(tbl,reverse=True)
probs=[f"{a}->{b}" for a,b in zip(w,w[1:]) if not tbl[b]>tbl[a]]
out["intercepted_monotonicity"]={"captured":len(captured),"error_pct_by_bits":tbl,
                                 "monotone":not probs,"problems":probs}
print("  monotone:",not probs,f"(captured {len(captured)} real tensors)")
json.dump(out,open("results_llama/posthoc.json","w"),indent=2)
print("\nwrote results_llama/posthoc.json")
