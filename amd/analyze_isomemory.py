"""iso-memory condition (§6) and the [SPEC-GAP 3] quant_byte_cost sensitivity."""
import sys, json, glob, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.analysis import paired_bootstrap
from kvre.cache_engine import iso_memory_tokens

rows={}
for p in glob.glob('results/bits*/*.jsonl'):
    if 'failures' in p: continue
    for l in open(p):
        try: r=json.loads(l)
        except: continue
        if r['promotion']=='attention' and r['context_target']==1029 and r['quant_bits']==4:
            rows.setdefault((r['iso_condition'],r['nominal_budget'],r['arm'],r['seed']),r)

def cell(iso,b,a):
    src = 'iso_token' if a in (1,2) else iso        # permanent arms unaffected by the condition
    return {s:r['accuracy'] for (i,bu,ar,s),r in rows.items() if i==src and bu==b and ar==a}

out={}
print("=== iso-token vs iso-memory (tiered arms funded to equal bytes) ===")
print("T = budget / (f + (1-f)*cost); f=0.5, cost=0.25 -> T = 1.6*budget")
print(f"{'budget':>7} {'T':>5} {'arm3 tok':>9} {'arm3 mem':>9} {'arm4 tok':>9} {'arm4 mem':>9} "
      f"{'a4 mem-tok':>11} {'95% CI':>22} {'p':>7}")
for b in [154,257,514]:
    T=iso_memory_tokens(b)
    t3,m3,t4,m4 = cell('iso_token',b,3),cell('iso_memory',b,3),cell('iso_token',b,4),cell('iso_memory',b,4)
    if not (m3 and m4): continue
    common=sorted(set(t4)&set(m4))
    r=paired_bootstrap([m4[s] for s in common],[t4[s] for s in common],f"b{b}",n_boot=10000)
    print(f"{b:>7} {T:>5} {statistics.mean(t3.values()):>9.4f} {statistics.mean(m3.values()):>9.4f} "
          f"{statistics.mean(t4.values()):>9.4f} {statistics.mean(m4.values()):>9.4f} "
          f"{r.mean_diff:>+11.4f} [{r.mean_ci[0]:>+7.4f},{r.mean_ci[1]:>+7.4f}] {r.p_perm:>7.4f}")
    out[f"b{b}"]={"T":T,"arm3_token":statistics.mean(t3.values()),"arm3_memory":statistics.mean(m3.values()),
                  "arm4_token":statistics.mean(t4.values()),"arm4_memory":statistics.mean(m4.values()),
                  "arm4_diff":r.mean_diff,"ci":list(r.mean_ci),"p":r.p_perm,"n":len(common)}

print("\n=== interaction under each iso-condition (BH within condition, per §8) ===")
for iso in ['iso_token','iso_memory']:
    for b in [154,257,514]:
        d={a:cell(iso,b,a) for a in [1,2,3,4]}
        common=sorted(set(d[1])&set(d[2])&set(d[3])&set(d[4]))
        if not common: continue
        iv=[(d[4][s]-d[3][s])-(d[2][s]-d[1][s]) for s in common]
        r=paired_bootstrap(iv,[0.0]*len(iv),'',n_boot=10000)
        print(f"  {iso:<11} b{b:<4} interaction {r.mean_diff:>+8.4f} "
              f"[{r.mean_ci[0]:>+7.4f},{r.mean_ci[1]:>+7.4f}]  n_differ={r.n_differ}")
        out[f"interaction_{iso}_b{b}"]={"mean":r.mean_diff,"ci":list(r.mean_ci),"n_differ":r.n_differ}

print("\n=== [SPEC-GAP 3] quant_byte_cost sensitivity, budget 257 ===")
print(f"{'cost':>6} {'T':>5} {'arm3':>8} {'arm4':>8}   (0.25 is the registered value; §3 notes a real")
print(f"{'':>6} {'':>5} {'':>8} {'':>8}    int8 scheme with bf16 scales is nearer 0.28-0.31)")
for c in [0.25,0.28,0.31]:
    iso = 'iso_memory' if c==0.25 else f'iso_memory_c{c}'
    a3,a4 = cell(iso,257,3), cell(iso,257,4)
    if not a4: continue
    T=iso_memory_tokens(257,0.5,c)
    print(f"{c:>6.2f} {T:>5} {statistics.mean(a3.values()) if a3 else float('nan'):>8.4f} "
          f"{statistics.mean(a4.values()):>8.4f}")
    out[f"sensitivity_c{c}"]={"T":T,"arm3":statistics.mean(a3.values()) if a3 else None,
                              "arm4":statistics.mean(a4.values()),"n":len(a4)}
json.dump(out, open('results/isomemory_analysis.json','w'), indent=2)
print("\nwrote results/isomemory_analysis.json")
