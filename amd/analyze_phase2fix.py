"""Phase 2: original (contaminated P4/P5) vs decontaminated signals."""
import sys, json, glob, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.analysis import paired_bootstrap, bh_correct
from kvre.arms import PROMO_IDS

def load(pat, tag):
    d={}
    for p in glob.glob(pat):
        if 'failures' in p: continue
        for l in open(p):
            try: r=json.loads(l)
            except: continue
            if (r['arm']==4 and r['nominal_budget']==257 and r['quant_bits']==4
                    and r['context_target']==1029 and r['iso_condition']=='iso_token'):
                d.setdefault((r['promotion'],r['seed']),r['accuracy'])
    return d
orig=load('results/bits4/*.jsonl','orig'); fix=load('results/p2fix/*.jsonl','fix')
REF={"attention":0.203,"epiphany":0.216,"random":0.154,"roundrobin":0.184,"oracle":0.188}
print(f"{'id':>3} {'signal':<12} {'original':>9} {'fixed':>9} {'delta':>8} {'ref':>7} {'n_orig':>7} {'n_fix':>6}")
out={}
for sig in ["attention","epiphany","random","roundrobin","oracle"]:
    o=[v for (s,_),v in orig.items() if s==sig]; f=[v for (s,_),v in fix.items() if s==sig]
    if not o: continue
    mo=statistics.mean(o); mf=statistics.mean(f) if f else float('nan')
    print(f"{PROMO_IDS[sig]:>3} {sig:<12} {mo:>9.4f} {mf:>9.4f} {mf-mo:>+8.4f} "
          f"{REF[sig]:>7.3f} {len(o):>7} {len(f):>6}")
    out[PROMO_IDS[sig]]={"signal":sig,"original":mo,"fixed":mf if f else None,
                         "reference":REF[sig],"n_orig":len(o),"n_fix":len(f)}
# contrasts vs P3 under the fixed signals
if fix:
    print("\n=== fixed-signal contrasts vs P3 (random control), BH-corrected ===")
    res=[];meta=[]
    for sig in ["attention","epiphany","roundrobin","oracle"]:
        common=sorted({s for (g,s) in fix if g==sig} & {s for (g,s) in fix if g=="random"})
        if not common: continue
        r=paired_bootstrap([fix[(sig,s)] for s in common],[fix[("random",s)] for s in common],
                           f"{PROMO_IDS[sig]} vs P3", n_boot=10000)
        res.append(r); meta.append(sig)
    if res:
        padj=bh_correct([r.p_perm for r in res])
        for i,r in enumerate(res):
            print(f"  {PROMO_IDS[meta[i]]} vs P3: {r.mean_diff:+.4f} "
                  f"[{r.mean_ci[0]:+.4f},{r.mean_ci[1]:+.4f}] n_differ={r.n_differ} "
                  f"p_BH={padj[i]:.4f} equiv={'yes' if r.equivalent_at_005 else 'no'}")
            out[f"contrast_{PROMO_IDS[meta[i]]}_vs_P3"]={"diff":r.mean_diff,"ci":list(r.mean_ci),
                                                          "n_differ":r.n_differ,"p_bh":padj[i]}
json.dump(out, open('results/phase2_fixed_analysis.json','w'), indent=2)
print("\nwrote results/phase2_fixed_analysis.json")
