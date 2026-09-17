"""Recalibrate every gate on the second model at the matched-task configuration.

Matched task = identical to the 1.5B run: N=6, 20 distractors, filler 7, pad 26, budgets
154/257/514. The competence gate cannot be equalised on this model (see calib3b_step1/step2:
never below ~0.94 across 8 configurations spanning N=6-16 and contexts 1030-8155), so the
ceiling difference is reported as a measured covariate rather than tuned away.
"""
import sys, json, statistics, time; sys.path.insert(0,'/home/kxrx26/research test')
import torch
from kvre.model import load, kv_tensors_from_cache
from kvre.engine import Engine
from kvre import gates
from kvre.task import build_prompt, question_for
from kvre.arms import make_cfg
from kvre.cache_engine import error_table, assert_monotone_error

MODEL = sys.argv[1] if len(sys.argv)>1 else "Qwen/Qwen2.5-3B-Instruct"
CAL   = list(range(9000, 9020))
model, tok = load(MODEL); eng = Engine(model, tok); eng.model_id = MODEL
res = {"model": MODEL, "config": "matched-task: N=6, 20 distractors, filler 7, pad 26"}
t0=time.time()

res["G1"] = gates.gate_G1(tok, CAL)
print("G1", res["G1"]["pass"], res["G1"]["median_len"], res["G1"]["range"], flush=True)

res["G2"] = gates.gate_G2(eng, CAL)
print("G2", res["G2"]["pass"], round(res["G2"]["full_cache_mean_acc"],4),
      "(1.5B was 0.7167 on the same seeds)", flush=True)

res["G3"] = gates.gate_G3(eng, CAL[:8], [154,257,514])
print("G3", res["G3"]["pass"], res["G3"]["void_refused"], res["G3"]["full_cache_by_budget"], flush=True)

# G4: quantizer integrity on THIS model's real cached K/V
p = build_prompt(9000)
text = tok.apply_chat_template([{"role":"user","content":p.context+"\n\n"+question_for(p,0)}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
with torch.no_grad(): out = model(**ids, use_cache=True)
tbl = error_table(kv_tensors_from_cache(out.past_key_values), [8,4,2])
mono = assert_monotone_error(tbl)
res["G4"] = gates.gate_G4(eng, tok)
res["G4"]["error_table_3b"] = tbl
print("G4", res["G4"]["pass"], "| 3B error table:",
      {b:{k:round(v,2) for k,v in tbl[b].items()} for b in tbl}, flush=True)

res["G5"] = gates.gate_G5(eng, CAL)
print("G5", res["G5"]["pass"], res["G5"]["frac_weak"], res["G5"]["frac_strict_8step"], flush=True)

json.dump(res, open("results3b/calibration_gates.json","w"), indent=2, default=str)
print("\nelapsed %.1f min"%((time.time()-t0)/60))
print("ALL PASS:", all(res[g]["pass"] for g in ("G1","G2","G3","G4","G5")))
for g in ("G1","G2","G3","G4","G5"):
    if not res[g]["pass"]: print("  FAILED", g, res[g].get("problems"))
