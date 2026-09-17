import sys, json, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre import gates
model, tok = load(); eng = Engine(model, tok)
CAL = list(range(9000, 9020))          # calibration seeds, disjoint from measurement seeds 0-149
t0=time.time()
res = {}
res["G1"] = gates.gate_G1(tok, CAL)
print("G1", res["G1"]["pass"], res["G1"]["median_len"], res["G1"]["range"], flush=True)
res["G2"] = gates.gate_G2(eng, CAL)
print("G2", res["G2"]["pass"], round(res["G2"]["full_cache_mean_acc"],4), flush=True)
res["G3"] = gates.gate_G3(eng, CAL[:8], [154,257,514])
print("G3", res["G3"]["pass"], res["G3"]["void_refused"], res["G3"]["full_cache_by_budget"], flush=True)
res["G4"] = gates.gate_G4(eng, tok)
print("G4", res["G4"]["pass"], flush=True)
res["G5"] = gates.gate_G5(eng, CAL)
print("G5", res["G5"]["pass"], res["G5"]["frac_weak"], res["G5"]["frac_strict_8step"], flush=True)
json.dump(res, open("results/calibration_gates.json","w"), indent=2, default=str)
print("\nelapsed %.1fs"%(time.time()-t0))
print("ALL PASS:", all(res[g]["pass"] for g in res))
for g in res:
    if not res[g]["pass"]: print("FAILED", g, res[g].get("problems"))
