"""Re-derive Phase 1 effects from raw JSONL (verification pass, 2026-09-20).
Two estimands, both reported:
  DiD    = (arm4-arm3) - (arm2-arm1)   pre-registered interaction (PREREG s3)
  SIMPLE = arm4 - arm2                 'recoverability effect given protection'
Paired per-seed, mean, 10k prompt-resampled bootstrap, numpy default_rng(0). No GPU, no new runs."""
import json, collections, numpy as np
R='D:/INNOCREW/Blockage/results/'
def load(p): return [json.loads(l) for l in open(R+p) if l.strip()]
def tab(rows):
    t=collections.defaultdict(dict)
    for r in rows: t[(r['arm'],r['budget'],r['iso_condition'])][r['seed']]=r['frac_retrieved']
    return t
def boot(d,n=10000):
    rng=np.random.default_rng(0); d=np.asarray(d); m=d[rng.integers(0,len(d),(n,len(d)))].mean(1)
    return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5),int((d!=0).sum()),len(d)
A=['1_no_protect_permanent','2_protect_permanent','3_recoverable_no_protection','4_recoverable_protected']
def est(T,B,iso,kind):
    a1,a2,a3,a4=[T[(a,B,iso if a[0] in '34' else 'iso_token')] for a in A]
    if kind=='SIMPLE': s=sorted(set(a2)&set(a4)); return np.array([a4[k]-a2[k] for k in s])
    s=sorted(set(a1)&set(a2)&set(a3)&set(a4)); return np.array([(a4[k]-a3[k])-(a2[k]-a1[k]) for k in s])
def show(label,T,iso,kinds=('DiD','SIMPLE')):
    for B in (154,257,514):
        for k in kinds:
            if k=='DiD' and any(a not in {x[0] for x in T} for a in A): continue
            print('%-26s B=%3d %-6s %-10s %+.4f [%+.4f, %+.4f]  differ %d/%d'%((label,B,k,iso)+boot(est(T,B,iso,k))))
p1=tab(load('phase1/raw_results.jsonl')); show('8bit sum',p1,'iso_token'); show('8bit sum',p1,'iso_memory')
pm=tab(load('phase1_meanscore/raw_results.jsonl')); show('8bit mean',pm,'iso_token')
T4=dict(p1); T4.update(tab(load('phase1_4bit/raw_results.jsonl'))); show('4bit sum',T4,'iso_token')
q=tab([dict(r,iso_condition='iso_token') for r in load('q4_phase1_4bit_mean/raw.jsonl')])
a2,a4=q[(A[1],514,'iso_token')],q[(A[3],514,'iso_token')]
print('%-26s B=514 SIMPLE only (arms 1,3 not run) %+.4f [%+.4f, %+.4f]  differ %d/%d'%(('4bit mean',)+boot([a4[k]-a2[k] for k in sorted(a2)])))
# 4-bit recoverability effect with protection off (arm3 - arm1), used in ANALYSIS.md s10.2
for B in (154,257,514):
    a1=T4[(A[0],B,'iso_token')];a3=T4[(A[2],B,'iso_token')];s=sorted(a1)
    print('4bit sum B=%3d arm3-arm1 %+.4f [%+.4f, %+.4f]  differ %d/%d'%((B,)+boot([a3[k]-a1[k] for k in s])))
