"""Step 2: N does not move the 3B's competence gate. Try context length / distractor density,
which stress retrieval rather than copy fidelity.
"""
import sys, json, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.task import build_prompt, question_for
from kvre.arms import make_cfg
MODEL="Qwen/Qwen2.5-3B-Instruct"; TARGET=0.7989; CAL=list(range(9000,9012))
model,tok=load(MODEL); eng=Engine(model,tok); eng.model_id=MODEL
def ctx(nc,nd,nf,seeds=range(9000,9005)):
    return statistics.median([len(tok(tok.apply_chat_template(
        [{"role":"user","content":build_prompt(s,n_filler_paragraphs=nf,n_credentials=nc,
          n_distractors=nd).context+"\n\n"+question_for(build_prompt(s,n_filler_paragraphs=nf,
          n_credentials=nc,n_distractors=nd),0)}], tokenize=False, add_generation_prompt=True)
        )["input_ids"]) for s in seeds])
def acc(nc,nd,nf,seeds):
    return statistics.mean([eng.run_prompt(build_prompt(s,n_filler_paragraphs=nf,n_credentials=nc,
            n_distractors=nd), make_cfg(6,10**9), seed=s)["accuracy"] for s in seeds])
print(f"{'N':>3} {'distr':>6} {'filler':>7} {'ctx':>6} {'full_cache':>11}   target {TARGET:.4f}", flush=True)
trace=[]
for nc,nd,nf in [(6,60,20),(6,120,40),(6,200,60),(10,200,60)]:
    c=ctx(nc,nd,nf); a=acc(nc,nd,nf,CAL)
    trace.append({"N":nc,"distractors":nd,"filler":nf,"ctx":c,"acc":a})
    print(f"{nc:>3} {nd:>6} {nf:>7} {c:>6.0f} {a:>11.4f}", flush=True)
json.dump(trace, open("results/calib3b_step2.json","w"), indent=2)
