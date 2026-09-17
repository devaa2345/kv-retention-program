"""Calibrate the context-length conditions and RE-RUN the gates at each length.

The brief is explicit that the 1029-token calibration must not be assumed to transfer.
Retention ratio is held near 13% of the *realised full-cache length* (prefill + all questions
and answers), which is the denominator under which Phase 1's budget 154 equals ~12.4% at the
1029-token condition. Documented in FINDINGS.md.
"""
import sys, json; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.task import build_prompt, question_for
from kvre.arms import make_cfg
from kvre import gates

TARGETS = [2048, 4096]
RATIO = 0.13
model, tok = load(); eng = Engine(model, tok)
CAL = list(range(9000, 9012))

def ctx_len(nf, seeds=range(9000,9006)):
    L=[]
    for s in seeds:
        p = build_prompt(s, n_filler_paragraphs=nf)
        L.append(len(tok(tok.apply_chat_template(
            [{"role":"user","content":p.context+"\n\n"+question_for(p,0)}],
            tokenize=False, add_generation_prompt=True))["input_ids"]))
    return sum(L)/len(L)

out = {"ratio": RATIO, "lengths": []}
for target in TARGETS:
    # search filler-paragraph count for this target
    best, bestd = 7, 1e9
    for nf in range(5, 90):
        d = abs(ctx_len(nf) - target)
        if d < bestd: best, bestd, = nf, d
        if ctx_len(nf) > target * 1.15: break
    realised = ctx_len(best)
    # realised full-cache length after all 6 turns, from the uncapped reference arm
    r = eng.run_prompt(build_prompt(9000, n_filler_paragraphs=best), make_cfg(6, 10**9),
                       seed=9000)
    full_len = r["effective_tokens"]
    budget = int(round(RATIO * full_len))
    entry = {"target": target, "n_filler": best, "realised_ctx": realised,
             "full_cache_len": full_len, "budget": budget,
             "retention_ratio": budget / full_len,
             "competitive_room": budget - 64 - 1}
    print(json.dumps(entry), flush=True)

    # ---- gates re-run at THIS context length ----
    g = {}
    g["G1"] = gates.gate_G1(tok, CAL, target=target, tol=0.05)
    # G1 uses build_prompt defaults; override for this length
    import statistics
    L=[len(tok(tok.apply_chat_template([{"role":"user","content":build_prompt(s,n_filler_paragraphs=best).context+"\n\n"+question_for(build_prompt(s,n_filler_paragraphs=best),0)}],tokenize=False,add_generation_prompt=True))["input_ids"]) for s in CAL]
    g["G1"] = {"gate":"G1","pass": abs(statistics.median(L)-target)/target <= 0.05,
               "median_len": statistics.median(L), "range":[min(L),max(L)]}
    accs=[eng.run_prompt(build_prompt(s,n_filler_paragraphs=best), make_cfg(6,10**9), seed=s)["accuracy"] for s in CAL]
    g["G2"] = {"gate":"G2","pass": statistics.mean(accs)>=0.70, "full_cache_mean_acc": statistics.mean(accs)}
    cfg = make_cfg(1, budget)
    g["G3"] = {"gate":"G3","pass": cfg.competitive_room()>0, "competitive_room": cfg.competitive_room()}
    w=0
    for s in CAL:
        rr=eng.run_prompt(build_prompt(s,n_filler_paragraphs=best), make_cfg(1,budget), seed=s, collect_dormancy=True)
        w += 1 if rr["dormancy_events"]>=1 else 0
    g["G5"] = {"gate":"G5","pass": w/len(CAL)>=0.80, "frac_weak": w/len(CAL)}
    entry["gates"] = g
    entry["gates_pass"] = all(g[k]["pass"] for k in g)
    print(f"  gates@{target}: " + ", ".join(f"{k}={g[k]['pass']}" for k in g), flush=True)
    out["lengths"].append(entry)

json.dump(out, open("results/context_calibration.json","w"), indent=2)
print("\nwrote results/context_calibration.json")
print("ALL CONTEXT GATES PASS:", all(e["gates_pass"] for e in out["lengths"]))
