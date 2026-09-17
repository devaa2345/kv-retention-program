"""Recalibrate the task for a second model. Nothing from the 1029-token / N=6 configuration is
assumed to transfer.

A larger model copies 14 hex characters more reliably, so at fixed N it has more headroom and a
scale comparison would be confounded by unequal room. This tunes N (credential count) so the
competence gate G2 lands near the 1.5B's measured 0.7989, then derives budgets from the same
retention ratios, then re-runs every gate.
"""
import sys, json, statistics, time; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.task import build_prompt, question_for
from kvre.arms import make_cfg
from kvre.policy import PolicyConfig
from kvre.cache_engine import QuantAudit, error_table, assert_monotone_error
from kvre.model import kv_tensors_from_cache

MODEL   = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
TARGET  = 0.7989          # the 1.5B's measured full_cache_ref; match the *gate*, not the task
CAL     = list(range(9000, 9020))
# 1.5B reference: budgets 154/257/514 against a realised full-cache length of ~1250
RATIOS  = [154/1250, 257/1250, 514/1250]

model, tok = load(MODEL); eng = Engine(model, tok); eng.model_id = MODEL
print(f"model {MODEL}: {model.config.num_hidden_layers} layers, "
      f"{model.config.num_key_value_heads} kv heads, hidden {model.config.hidden_size}", flush=True)

def ctx_len(n_cred, nf=7, seeds=range(9000,9006)):
    L=[]
    for s in seeds:
        p=build_prompt(s, n_filler_paragraphs=nf, n_credentials=n_cred)
        L.append(len(tok(tok.apply_chat_template(
            [{"role":"user","content":p.context+"\n\n"+question_for(p,0)}],
            tokenize=False, add_generation_prompt=True))["input_ids"]))
    return statistics.median(L)

def full_cache_acc(n_cred, nf, seeds):
    a=[]
    for s in seeds:
        p=build_prompt(s, n_filler_paragraphs=nf, n_credentials=n_cred)
        a.append(eng.run_prompt(p, make_cfg(6, 10**9), seed=s)["accuracy"])
    return statistics.mean(a)

print("\n=== step 1: sweep N to hit the competence gate ===", flush=True)
print(f"{'N':>3} {'ctx':>6} {'full_cache_acc':>15}   target {TARGET:.4f}")
trace={}
best=None
for n_cred in [6, 9, 12, 16]:
    c=ctx_len(n_cred)
    a=full_cache_acc(n_cred, 7, CAL[:12])
    trace[n_cred]={"ctx":c,"acc":a}
    print(f"{n_cred:>3} {c:>6.0f} {a:>15.4f}", flush=True)
    if best is None or abs(a-TARGET) < abs(trace[best]["acc"]-TARGET): best=n_cred
print(f"\nclosest N = {best} (acc {trace[best]['acc']:.4f}, ctx {trace[best]['ctx']:.0f})")
json.dump({"model":MODEL,"target":TARGET,"trace":trace,"chosen_N":best},
          open("results/calib3b_step1.json","w"), indent=2)
