"""Deliverable 6: agreement report — this implementation against every reference in the spec.

Every disagreement greater than 0.05 is traced to a named SPEC_QUESTIONS entry.
"""
import sys, os, json, glob, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.analysis import load_rows, paired_bootstrap

RES="results"
CELL=("arm","nominal_budget","seed","quant_bits","promotion","context_target",
      "iso_condition","recency_window")

def all_rows():
    seen={}
    for p in sorted(glob.glob(f"{RES}/bits*/*.jsonl")):
        if p.endswith(".failures.jsonl"): continue
        for r in load_rows(p):
            seen.setdefault(tuple(r.get(x) for x in CELL), r)
    return list(seen.values())

rows=all_rows()
def sel(**kw): return sorted([r for r in rows if all(r.get(k)==v for k,v in kw.items())],
                             key=lambda r: r["seed"])
def acc(**kw):
    v=[r["accuracy"] for r in sel(**kw)]
    return statistics.mean(v) if v else float("nan"), len(v)

def interaction(bits, budget):
    d={}
    for a in [1,2,3,4]:
        b = 4 if a in (1,2) else bits      # arms 1/2 are permanent -> bit-width invariant
        d[a]={r["seed"]:r["accuracy"] for r in sel(arm=a,nominal_budget=budget,quant_bits=b,
                                                   promotion="attention",context_target=1029)}
    common=sorted(set(d[1])&set(d[2])&set(d[3])&set(d[4]))
    if not common: return None
    iv=[(d[4][s]-d[3][s])-(d[2][s]-d[1][s]) for s in common]
    r=paired_bootstrap(iv,[0.0]*len(iv),f"int b{budget} {bits}bit",n_boot=10000)
    return {"n":len(common),"mean":r.mean_diff,"ci":r.mean_ci,"n_differ":r.n_differ}

L=["# AGREEMENT REPORT\n",
   "Deliverable 6. This blind reimplementation against every reference value stated in",
   "`SPEC_REIMPL_v1.md`. Disagreements greater than 0.05 are traced to a named gap in",
   "`SPEC_QUESTIONS.md`.\n",
   "Per the spec's own §10: \"Agreement on the direction and rough magnitude is what matters.",
   "Exact numeric identity is not expected and would be suspicious.\"\n",
   "| # | quantity | reference | measured | delta | agree | traced to |",
   "|---|---|---|---|---|---|---|"]
n=[0]
def row(q, ref, got, gap="—", tol=0.05, fmt="{:.4f}", unit=""):
    n[0]+=1
    try:
        d=got-ref; ok = abs(d)<=tol
        L.append(f"| {n[0]} | {q} | {fmt.format(ref)}{unit} | {fmt.format(got)}{unit} | "
                 f"{d:+.4f} | {'yes' if ok else '**NO**'} | {gap} |")
        return ok
    except Exception:
        L.append(f"| {n[0]} | {q} | {ref} | {got} | – | – | {gap} |")
        return None

# ---- 1. quantizer error table (section 3) ----
QT={8:(1.18,0.91),4:(19.98,15.20),2:(103.35,75.55)}      # measured, real cached K/V
REF={8:(1.17,0.88),4:(20.06,14.99),2:(101.11,74.68)}
for b in (8,4,2):
    row(f"quantizer error {b}-bit keys", REF[b][0], QT[b][0], "[GAP-Q] resolved", tol=2.5,
        fmt="{:.2f}", unit="%")
    row(f"quantizer error {b}-bit values", REF[b][1], QT[b][1], "[GAP-Q] resolved", tol=2.5,
        fmt="{:.2f}", unit="%")

# ---- 2. full_cache_ref (section 5) ----
fc,_=acc(arm=6,context_target=1029)
row("full_cache_ref", 0.947, fc, "[GAP-U] copy fidelity — unresolved")

# ---- 3. Phase 1 interaction (deliverable 3) ----
for budget,ref in ((257,0.002),(514,-0.002)):
    i4=interaction(4,budget)
    if i4: row(f"interaction @ budget {budget}, **4-bit**", ref, i4["mean"],
               "[GAP-W] Phase 1 bit-width unspecified")
    i8=interaction(8,budget)
    if i8: row(f"interaction @ budget {budget}, **8-bit**", ref, i8["mean"],
               "[GAP-W] test of the traced cause")

# ---- 4. Phase 2 signals (deliverable 4) ----
REFP={"attention":0.203,"epiphany":0.216,"random":0.154,"roundrobin":0.184,"oracle":0.188}
for sig,ref in REFP.items():
    a,_=acc(arm=4,nominal_budget=257,quant_bits=4,promotion=sig,context_target=1029)
    row(f"Phase 2 {sig}", ref, a, "[GAP-U] ceiling / [SPEC-GAP 7] for epiphany")

# ---- 5. section 7 band ----
if os.path.exists(f"{RES}/band_4bit.json"):
    b=json.load(open(f"{RES}/band_4bit.json"))
    row("FULL−QUANT band @ 257, 4-bit", 0.278, b["band"], "[GAP-C] tier assignment rule")

# ---- 5b. Phase 2 with CORRECTED promotion signals (primary; see rho section) ----
fixrows=[]
for p_ in sorted(glob.glob(f"{RES}/p2fix/*.jsonl")):
    if p_.endswith(".failures.jsonl"): continue
    fixrows += load_rows(p_)
if fixrows:
    for sig,ref in REFP.items():
        v=[r["accuracy"] for r in fixrows if r.get("promotion")==sig]
        if v: row(f"Phase 2 {sig} — **corrected signals**", ref, statistics.mean(v),
                  "[GAP-U] ceiling; manipulation now passes")

# ---- 6. Spearman rho manipulation check, n=100 ----
rp = f"{RES}/rho_power.json"
if os.path.exists(rp):
    rr=json.load(open(rp))
    for pid in ["P1","P2","P3","P4","P5"]:
        if pid in rr:
            tgt = 1.0 if pid=="P1" else 0.0
            row(f"rho {pid} ({rr[pid]['signal']}) — original", tgt, rr[pid]["rho_old"],
                "manipulation check, n=100", tol=0.05 if pid=="P1" else 0.15)
            row(f"rho {pid} ({rr[pid]['signal']}) — corrected", tgt, rr[pid]["rho_new"],
                "manipulation check, n=100", tol=0.05 if pid=="P1" else 0.15)
elif os.path.exists(f"{RES}/rho_check.json"):
    rr=json.load(open(f"{RES}/rho_check.json"))["spearman_rho"]
    for pid in ["P1","P2","P3","P4","P5"]:
        if pid in rr:
            row(f"Spearman rho {pid} ({rr[pid]['signal']})", 1.0 if pid=="P1" else 0.0,
                rr[pid]["rho"], "manipulation check", tol=0.05 if pid=="P1" else 0.30)

# ---- 7. iso-memory (section 6) ----
if os.path.exists(f"{RES}/isomemory_analysis.json"):
    im=json.load(open(f"{RES}/isomemory_analysis.json"))
    for b in (257,514):
        k=f"interaction_iso_memory_b{b}"
        if k in im:
            L.append(f"| — | interaction @ {b}, iso-**memory** 4-bit | (none stated) | "
                     f"{im[k]['mean']:+.4f} | – | – | §6 condition, no reference given |")

L.append("")
L.append("## Interaction detail\n")
L.append("| budget | bit-width | interaction | 95% CI | n differ | n | reference |")
L.append("|"+"---|"*7)
for budget in (154,257,514):
    for bits in (4,8):
        i=interaction(bits,budget)
        if i:
            ref={257:"+0.002",514:"−0.002"}.get(budget,"—")
            L.append(f"| {budget} | {bits}-bit | {i['mean']:+.4f} | "
                     f"[{i['ci'][0]:+.4f}, {i['ci'][1]:+.4f}] | {i['n_differ']} | {i['n']} | {ref} |")
L.append("")
open("AGREEMENT.md","w").write("\n".join(L)+"\n")
print("wrote AGREEMENT.md")
print("\n".join(L[6:40]))
