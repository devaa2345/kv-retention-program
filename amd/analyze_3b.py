"""Second-model analysis: Phase 1 at 8-bit, sweep, fresh-seed replication, cross-model table."""
import sys, json, glob, statistics, os, re; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.analysis import paired_bootstrap, bh_correct

def load(pat):
    seen={}
    K=('arm','nominal_budget','seed','quant_bits','promotion','context_target','iso_condition')
    for p in sorted(glob.glob(pat)):
        if 'failures' in p: continue
        for l in open(p):
            if not l.strip(): continue
            try: r=json.loads(l)
            except: continue
            seen.setdefault(tuple(r.get(x) for x in K), r)
    return list(seen.values())

rows3b=load('results3b/*/*.jsonl')
out={"model":"Qwen/Qwen2.5-3B-Instruct","n_rows":len(rows3b)}
def sel(rs,**kw): return sorted([r for r in rs if all(r.get(k)==v for k,v in kw.items())],key=lambda r:r['seed'])
def cell(b,a): return {r['seed']:r['accuracy'] for r in sel(rows3b,nominal_budget=b,arm=a,quant_bits=8,promotion='attention',context_target=1029)}

print("=== 3B Phase 1, iso-token, 8-bit ===")
print(f"{'budget':>7} " + " ".join(f"{'arm'+str(a):>9}" for a in range(1,7)) + f" {'n':>5}")
p1={}
for b in (154,257,514):
    cs=[cell(b,a) for a in range(1,7)]
    if not all(cs): continue
    ns=min(len(c) for c in cs)
    print(f"{b:>7} " + " ".join(f"{statistics.mean(c.values()):>9.4f}" for c in cs) + f" {ns:>5}")
    p1[b]={f"arm{a}":statistics.mean(cs[a-1].values()) for a in range(1,7)}
    p1[b]["n"]=ns
out["phase1"]=p1

print("\n=== 3B Phase 1 contrasts and interaction ===")
print(f"{'budget':>7} {'protection':>11} {'tiering':>9} {'interaction':>12} {'95% CI':>22} {'ndiff':>6}")
inter={}
for b in (154,257,514):
    d={a:cell(b,a) for a in range(1,7)}
    common=sorted(set(d[1])&set(d[2])&set(d[3])&set(d[4]))
    if not common: continue
    prot=[(d[2][s]+d[4][s])/2-(d[1][s]+d[3][s])/2 for s in common]
    tier=[(d[3][s]+d[4][s])/2-(d[1][s]+d[2][s])/2 for s in common]
    iv=[(d[4][s]-d[3][s])-(d[2][s]-d[1][s]) for s in common]
    r=paired_bootstrap(iv,[0.0]*len(iv),'',n_boot=10000)
    print(f"{b:>7} {statistics.mean(prot):>+11.4f} {statistics.mean(tier):>+9.4f} "
          f"{r.mean_diff:>+12.4f} [{r.mean_ci[0]:>+7.4f},{r.mean_ci[1]:>+7.4f}] {r.n_differ:>6}")
    inter[b]={"protection":statistics.mean(prot),"tiering":statistics.mean(tier),
              "interaction":r.mean_diff,"ci":list(r.mean_ci),"n_differ":r.n_differ,"n":len(common)}
out["interaction"]=inter

print("\n=== 3B fresh-seed replication of the collapse widths (seeds 50-99) ===")
print(f"{'bits':>5} {'orig 0-49':>10} {'fresh 50-99':>12} {'diff':>8} {'n_fresh':>8}")
rep={}
for w in (5,4,3):
    o=[r['accuracy'] for r in rows3b if r['quant_bits']==w and r['arm']==4 and r['nominal_budget']==257 and r['seed']<50]
    f=[r['accuracy'] for r in rows3b if r['quant_bits']==w and r['arm']==4 and r['nominal_budget']==257 and 50<=r['seed']<100]
    if not f: continue
    print(f"{w:>5} {statistics.mean(o):>10.4f} {statistics.mean(f):>12.4f} "
          f"{statistics.mean(f)-statistics.mean(o):>+8.4f} {len(f):>8}")
    rep[w]={"orig":statistics.mean(o),"fresh":statistics.mean(f),"n_fresh":len(f)}
out["replication"]=rep
json.dump(out,open("results3b/analysis.json","w"),indent=2)
print("\nwrote results3b/analysis.json")
