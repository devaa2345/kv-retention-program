"""Phase 1 as a three-factor design: protection x tiering x bit-width.

Complete n=150 data exists at both 4-bit and 8-bit for the tiered arms. Arms 1 and 2 are
permanent-eviction (n_quant=0) and therefore bit-width invariant, so they serve as the shared
reference level of the bit-width factor rather than being re-run.
"""
import sys, json, glob, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.analysis import paired_bootstrap, bh_correct

rows={}
for p in glob.glob('results/bits*/*.jsonl'):
    if 'failures' in p: continue
    for l in open(p):
        try: r=json.loads(l)
        except: continue
        if (r['promotion']=='attention' and r['context_target']==1029
                and r['iso_condition']=='iso_token'):
            rows.setdefault((r['quant_bits'],r['nominal_budget'],r['arm'],r['seed']),r)

def cell(bits,b,a):
    bb = 4 if a in (1,2) else bits          # arms 1/2 bit-invariant
    return {s:r['accuracy'] for (q,bu,ar,s),r in rows.items() if q==bb and bu==b and ar==a}

print("=== cell means: protection x tiering x bit-width ===")
print(f"{'budget':>7} {'bits':>5} {'a1 none/perm':>13} {'a2 str/perm':>12} "
      f"{'a3 none/tier':>13} {'a4 str/tier':>12}")
out={}
for b in [154,257,514]:
    for bits in [8,4]:
        cs=[cell(bits,b,a) for a in [1,2,3,4]]
        if not all(cs): continue
        m=[statistics.mean(c.values()) for c in cs]
        print(f"{b:>7} {bits:>5} {m[0]:>13.4f} {m[1]:>12.4f} {m[2]:>13.4f} {m[3]:>12.4f}")

print("\n=== main effects and interaction, by bit-width ===")
print(f"{'budget':>7} {'bits':>5} {'protection':>11} {'tiering':>10} {'interaction':>12} "
      f"{'95% CI':>22} {'n differ':>9}")
res=[];meta=[]
for b in [154,257,514]:
    for bits in [8,4]:
        d={a:cell(bits,b,a) for a in [1,2,3,4]}
        common=sorted(set(d[1])&set(d[2])&set(d[3])&set(d[4]))
        if not common: continue
        prot=[( d[2][s]+d[4][s] )/2-( d[1][s]+d[3][s] )/2 for s in common]
        tier=[( d[3][s]+d[4][s] )/2-( d[1][s]+d[2][s] )/2 for s in common]
        inter=[(d[4][s]-d[3][s])-(d[2][s]-d[1][s]) for s in common]
        r=paired_bootstrap(inter,[0.0]*len(inter),f"b{b} {bits}bit",n_boot=10000)
        res.append(r); meta.append({"budget":b,"bits":bits})
        print(f"{b:>7} {bits:>5} {statistics.mean(prot):>+11.4f} {statistics.mean(tier):>+10.4f} "
              f"{r.mean_diff:>+12.4f} [{r.mean_ci[0]:>+7.4f},{r.mean_ci[1]:>+7.4f}] {r.n_differ:>9}")
        out[f"b{b}_{bits}bit"]={"protection":statistics.mean(prot),"tiering":statistics.mean(tier),
                                "interaction":r.mean_diff,"ci":list(r.mean_ci),
                                "n_differ":r.n_differ,"n":len(common)}

print("\n=== the bit-width factor itself: does tiering damage depend on width? ===")
print(f"{'budget':>7} {'tiering@8bit':>13} {'tiering@4bit':>13} {'difference':>11} {'95% CI':>22}")
for b in [154,257,514]:
    d8={a:cell(8,b,a) for a in [1,2,3,4]}; d4={a:cell(4,b,a) for a in [1,2,3,4]}
    common=sorted(set(d8[3])&set(d8[4])&set(d4[3])&set(d4[4])&set(d8[1])&set(d8[2]))
    if not common: continue
    t8=[(d8[3][s]+d8[4][s])/2-(d8[1][s]+d8[2][s])/2 for s in common]
    t4=[(d4[3][s]+d4[4][s])/2-(d4[1][s]+d4[2][s])/2 for s in common]
    r=paired_bootstrap(t4,t8,f"width b{b}",n_boot=10000)
    print(f"{b:>7} {statistics.mean(t8):>+13.4f} {statistics.mean(t4):>+13.4f} "
          f"{r.mean_diff:>+11.4f} [{r.mean_ci[0]:>+7.4f},{r.mean_ci[1]:>+7.4f}]")
    out[f"widtheffect_b{b}"]={"tier_8bit":statistics.mean(t8),"tier_4bit":statistics.mean(t4),
                              "diff":r.mean_diff,"ci":list(r.mean_ci)}
json.dump(out, open("results/three_factor.json","w"), indent=2)
print("\nwrote results/three_factor.json")
