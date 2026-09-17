"""Llama-3.2-3B-Instruct: validity check + bit-width sweep (8/7/6/5/4/3, n=50).

Cross-ARCHITECTURE test. The two Qwen models share a tokenizer and a 2 KV heads x 128 head_dim
KV geometry, so their quantizer groups are 256 elements. Llama-3.2-3B uses 8 KV heads x 128,
i.e. 1024-element groups -- four times larger. If the non-monotone collapse survives that, it is
not an artefact of one quantizer grouping.
"""
import sys, json, time, statistics; sys.path.insert(0,'/home/kxrx26/research test')
import torch
from kvre.model import load, kv_tensors_from_cache
from kvre.engine import Engine
from kvre.runner import run_cells
from kvre.task import build_prompt, question_for
from kvre.arms import make_cfg
from kvre.cache_engine import error_table, assert_monotone_error

MODEL=sys.argv[2] if len(sys.argv)>2 else "/home/kxrx26/llama32_3b_instruct"
N=int(sys.argv[1]) if len(sys.argv)>1 else 50
model,tok=load(MODEL); eng=Engine(model,tok); eng.model_id=MODEL
hd = model.config.hidden_size//model.config.num_attention_heads
print(f"{MODEL}: {model.config.num_hidden_layers} layers, {model.config.num_key_value_heads} kv heads, "
      f"head_dim {hd}, hidden {model.config.hidden_size}", flush=True)
print(f"  quantizer group size = {model.config.num_key_value_heads*hd} elements "
      f"(Qwen models: 256)", flush=True)

# --- validity: context length, competence, quantizer error on THIS model's KV ---
CAL=list(range(9000,9012))
L=[len(tok(tok.apply_chat_template([{"role":"user","content":build_prompt(s).context+"\n\n"+
    question_for(build_prompt(s),0)}], tokenize=False, add_generation_prompt=True))["input_ids"])
   for s in CAL]
print(f"  context median {statistics.median(L):.0f} range [{min(L)},{max(L)}]", flush=True)
acc=[eng.run_prompt(build_prompt(s), make_cfg(6,10**9), seed=s)["accuracy"] for s in CAL]
print(f"  full_cache_ref = {statistics.mean(acc):.4f}  (1.5B 0.7167, Qwen-3B 0.9833)", flush=True)

p=build_prompt(9000)
ids=tok(tok.apply_chat_template([{"role":"user","content":p.context+"\n\n"+question_for(p,0)}],
        tokenize=False, add_generation_prompt=True), return_tensors="pt",
        add_special_tokens=False).to("cuda")
with torch.no_grad(): o=model(**ids, use_cache=True)
tbl=error_table(kv_tensors_from_cache(o.past_key_values),[8,4,2])
print("  quantizer error:", {b:{k:round(v,2) for k,v in tbl[b].items()} for b in tbl}, flush=True)
print("  monotone:", not assert_monotone_error(tbl), flush=True)
json.dump({"model":"meta-llama/Llama-3.2-3B-Instruct","path":MODEL,"layers":model.config.num_hidden_layers,
           "kv_heads":model.config.num_key_value_heads,"head_dim":hd,
           "group_size":model.config.num_key_value_heads*hd,
           "ctx_median":statistics.median(L),"full_cache_ref":statistics.mean(acc),
           "error_table":tbl}, open("results_llama/validity.json","w"), indent=2)

# --- sweep ---
t0=time.time()
for w in [8,7,6,5,4,3]:
    cells=[dict(arm=4,budget=257,seed=s,iso_condition="iso_token",quant_bits=w,
                promotion="attention",context_target=1029) for s in range(N)]
    print(f"llama sweep {w}-bit: {len(cells)} cells", flush=True)
    run_cells(eng,cells,f"results_llama/bits{w}",f"sweep_bits{w}",verbose_every=25)
print("sweep total %.1f min"%((time.time()-t0)/60))
