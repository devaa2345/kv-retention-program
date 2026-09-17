import sys, torch; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load, kv_tensors_from_cache
from kvre.task import build_prompt, question_for
from kvre.cache_engine import relative_error
model, tok = load()
p = build_prompt(0)
text = tok.apply_chat_template([{"role":"user","content":p.context+"\n\n"+question_for(p,0)}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**ids, use_cache=True)
kvs = kv_tensors_from_cache(out.past_key_values)

def qdq(x, bits, mode):
    b,h,s,d = x.shape
    g = x.permute(0,2,1,3).reshape(b,s,h*d)
    if mode in ("fp32_all","fp32_mid"):
        gw = g.float()
    else:
        gw = g
    lo = gw.min(-1,keepdim=True).values; hi = gw.max(-1,keepdim=True).values
    qmax = float(2**bits-1)
    scale = (hi-lo)/torch.tensor(qmax, dtype=gw.dtype, device=x.device)
    scale = torch.where(scale==0, torch.ones_like(scale), scale)
    codes = torch.clamp(torch.round((gw-lo)/scale), 0.0, qmax)
    deq = codes*scale+lo
    if mode == "fp32_mid":            # fp32 math, stored back to bf16 (the engine holds bf16)
        deq = deq.to(torch.bfloat16)
    if mode == "fp32_all":
        deq = deq.to(x.dtype)         # measured against bf16 orig anyway
    return deq.reshape(b,s,h,d).permute(0,2,1,3)

def perhead(x, bits):                 # [GAP-G] alternative grouping: one group per head
    b,h,s,d = x.shape
    g = x.permute(0,2,1,3).reshape(b,s,h,d)
    lo = g.min(-1,keepdim=True).values; hi = g.max(-1,keepdim=True).values
    qmax=float(2**bits-1)
    scale=(hi-lo)/torch.tensor(qmax,dtype=g.dtype,device=x.device)
    scale=torch.where(scale==0,torch.ones_like(scale),scale)
    deq=torch.clamp(torch.round((g-lo)/scale),0.0,qmax)*scale+lo
    return deq.permute(0,2,1,3)

ref = {8:(1.17,0.88), 4:(20.06,14.99), 2:(101.11,74.68)}
print(f"{'variant':<22}{'bits':>5}{'keys':>9}{'values':>9}   {'refK':>7}{'refV':>7}")
for mode in ["bf16_all","fp32_mid","fp32_all","perhead_bf16"]:
    for bits in [8,4,2]:
        ke,ve=[],[]
        for k,v in kvs:
            f = perhead if mode=="perhead_bf16" else (lambda t,b_: qdq(t,b_,mode))
            ke.append(relative_error(k, f(k,bits))); ve.append(relative_error(v, f(v,bits)))
        print(f"{mode:<22}{bits:>5}{100*sum(ke)/len(ke):>8.2f}%{100*sum(ve)/len(ve):>8.2f}%   "
              f"{ref[bits][0]:>6.2f}%{ref[bits][1]:>6.2f}%")
