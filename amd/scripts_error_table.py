import sys, torch; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load, kv_tensors_from_cache
from kvre.task import build_prompt, question_for
from kvre.cache_engine import error_table, assert_monotone_error

model, tok = load()
print("loaded:", model.config.num_hidden_layers, "layers,",
      model.config.num_key_value_heads, "kv heads, head_dim",
      getattr(model.config,'head_dim', model.config.hidden_size//model.config.num_attention_heads))

p = build_prompt(0)
text = tok.apply_chat_template(
    [{"role":"user","content":p.context+"\n\n"+question_for(p,0)}],
    tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**ids, use_cache=True, output_attentions=True)
print("prefill tokens:", ids['input_ids'].shape[-1])
print("attentions returned:", out.attentions is not None,
      "| n_layers:", len(out.attentions), "| shape:", tuple(out.attentions[0].shape))

kvs = kv_tensors_from_cache(out.past_key_values)
print("cache layers:", len(kvs), "| K shape:", tuple(kvs[0][0].shape), "| dtype:", kvs[0][0].dtype)

tbl = error_table(kvs, [8,4,2])
print("\n=== Section 3 error table (real cached K/V, bf16 path) ===")
print(f"{'bits':>5} {'keys':>10} {'values':>10}   {'ref keys':>9} {'ref vals':>9}")
ref = {8:(1.17,0.88), 4:(20.06,14.99), 2:(101.11,74.68)}
for b in [8,4,2]:
    print(f"{b:>5} {tbl[b]['keys']:>9.2f}% {tbl[b]['values']:>9.2f}%   "
          f"{ref[b][0]:>8.2f}% {ref[b][1]:>8.2f}%")
print("\nmonotonicity problems:", assert_monotone_error(tbl) or "NONE")
