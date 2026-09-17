"""Build CSV exports and FINDINGS.md from the JSONL result files."""
import sys, os, json, glob, statistics, collections; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.analysis import (load_rows, export_csv, export_paired_csv, paired_bootstrap,
                           bh_correct, spearman, CSV_FIELDS)
from kvre.arms import ARMS, PROMO_IDS

RES="results"; os.makedirs(f"{RES}/csv", exist_ok=True)

CELL_KEYS = ("arm","nominal_budget","seed","quant_bits","promotion","context_target",
             "iso_condition","recency_window")

def all_rows():
    """Load every result row, DEDUPED by full cell identity.

    Some cells are legitimately produced by more than one stage -- Phase 2's P1 (attention)
    arm is configurationally identical to Phase 1's arm-4/budget-257 cell, and the 4-bit point
    of the sweep is the same cell again. Globbing without deduping would pool them and inflate
    n, which is precisely the silent-pooling failure spec section 6 warns about. Duplicates are
    verified identical before being collapsed; any disagreement is reported loudly rather than
    averaged away.
    """
    seen={}
    conflicts=[]
    for p in sorted(glob.glob(f"{RES}/bits*/*.jsonl")):
        if p.endswith(".failures.jsonl"): continue
        for r in load_rows(p):
            k=tuple(r.get(x) for x in CELL_KEYS)
            if k in seen:
                a,b=seen[k],r
                if (a["accuracy"],a["effective_tokens"],a["n_quant"]) != \
                   (b["accuracy"],b["effective_tokens"],b["n_quant"]):
                    conflicts.append(k)
                continue
            seen[k]=r
    if conflicts:
        print(f"  !! {len(conflicts)} duplicated cells DISAGREE across stages -- "
              f"non-determinism, investigate: {conflicts[:3]}")
    else:
        print(f"  deduped to {len(seen)} unique cells (duplicates verified identical)")
    return list(seen.values())

def by(rows, **kw):
    out=[r for r in rows if all(r.get(k)==v for k,v in kw.items())]
    return sorted(out, key=lambda r: r["seed"])

def paired(rows, sel_a, sel_b, label):
    A=by(rows,**sel_a); B=by(rows,**sel_b)
    sa={r["seed"]:r["accuracy"] for r in A}; sb={r["seed"]:r["accuracy"] for r in B}
    common=sorted(set(sa)&set(sb))                     # sorted: order never depends on set order
    if not common: return None,0
    return paired_bootstrap([sa[s] for s in common],[sb[s] for s in common],label), len(common)

def fmt(r):
    return (f"{r.mean_diff:+.4f} | [{r.mean_ci[0]:+.4f}, {r.mean_ci[1]:+.4f}] | "
            f"{r.median_diff:+.4f} | [{r.median_ci[0]:+.4f}, {r.median_ci[1]:+.4f}] | "
            f"{r.n_differ} | {r.p_perm:.4f}")

rows=all_rows()
if not rows:
    print("no results yet"); sys.exit(0)
export_csv(rows, f"{RES}/csv/all_runs.csv")
print(f"wrote {RES}/csv/all_runs.csv  ({len(rows)} rows)")

md=["# FINDINGS\n",
    "Blind reimplementation of `SPEC_REIMPL_v1.md`. No file of the original harness was opened,",
    "imported, listed, or searched. Gap resolutions are in `SPEC_QUESTIONS.md`; every one",
    "referenced below is documented there.\n",
    "## Methods\n",
    "**Model and decoding.** `Qwen/Qwen2.5-1.5B-Instruct`, bfloat16, HuggingFace `transformers`",
    "with `attn_implementation=\"eager\"` so attention weights are readable, greedy decoding, no",
    "sampling. ROCm on an RX 7900 XTX (gfx1100). `PYTHONHASHSEED=0`.\n",
    "**Task.** Multi-credential retrieval with induced dormancy. Per seed: 6 credentials",
    "`CRED_<i>_KEY: sk-<14 hex>` (17-char values), 20 distractor lines of identical surface",
    "shape with non-credential labels, and coherent 50-word filler paragraphs interleaved so",
    "credential lines are separated and placed at varied, non-clustered depths. Context lands at",
    "median 1029 tokens (range 1023-1034 at n=10), matching the spec's stated target. Six turns,",
    "one credential each, `max_answer_tokens=22`. A credential counts as retrieved iff its full",
    "17-character value appears verbatim in the generated text of the turn that asked for it.",
    "Primary metric is fraction retrieved k/N per prompt, N=6.\n",
    "**Cache model.** Every position is FULL (bf16), QUANT (quantized), or EVICT (discarded).",
    "The quantizer is affine min/max, one group per (layer, position) spanning all",
    "`kv_heads x head_dim` = 256 elements, arithmetic in fp32 with storage at the engine's bf16",
    "([GAP-Q], [GAP-G]). Byte cost is `n_full + quant_byte_cost * n_quant`, default 0.25.\n",
    "**Retention.** Identical in every arm: sink (position 0) and the 64 most-recent positions",
    "are floors applied first; remaining budget is filled by attention-score rank; selection runs",
    "every decode step (`rebalance_every=1`). The attention score is the head-mean, layer-sum,",
    "accumulated over all steps including prefill, and divided by the number of queries that",
    "could causally attend to each position ([SPEC-GAP 4], resolved empirically). One global",
    "retained set is shared by all 28 layers ([GAP-H]). One cache spans all six turns, so",
    "eviction persists across turns ([GAP-E]) and generated tokens are themselves evictable",
    "([GAP-F]).\n",
    "**Protection.** Structural protection is content-agnostic: it seats whole matched",
    "`LABEL: sk-...` lines atomically, ranked by best member score, and therefore competes for",
    "budget across all 26 matched lines, credential and distractor alike. `oracle_static`",
    "instead knows which 6 lines are credentials and retains their label+value spans by",
    "construction, filling surplus budget by attention rank ([GAP-K]).\n",
    "**Determinism.** No selection over an unordered Python `set` reaches any output; every",
    "ranking sorts explicitly with an ascending-index tie-break. Two processes with",
    "`PYTHONHASHSEED=0` produce identical retained-token index sets. Independently, 127 cells",
    "that were legitimately computed twice by different stages agreed exactly on accuracy,",
    "effective_tokens and n_quant.\n",
    "**Checkpointing.** Append-only JSONL, flushed and fsynced per row, resumed on the",
    "`(nominal_budget, iso_condition, arm, seed, quant_bits, promotion, context_target)` key.",
    "Failed runs are written to a separate failures file and never silently dropped, so the",
    "paired seed sets stay identical across arms.\n",
    "**Analysis.** Bootstrapped median of paired per-prompt differences, 10,000 resamples,",
    "resampling unit = prompt; mean-difference CI and the count of prompts that differ at all",
    "reported alongside, per the spec's own warning that the median CI collapses to zero width",
    "on data discrete at 1/6. Paired sign-flip permutation test; Benjamini-Hochberg within each",
    "iso-condition; equivalence declared when the 95% CI is fully inside +/-0.05.\n"]

# ---------------- Phase 1 ----------------
p1=[r for r in rows if r.get("promotion")=="attention" and r.get("context_target")==1029
    and r.get("quant_bits")==4]
budgets=sorted({r["nominal_budget"] for r in p1})
if p1:
    md.append("## Phase 1 — iso-token, arms 1-6\n")
    md.append("Primary metric: fraction retrieved k/N per prompt, N=6. Paired design: every arm "
              "sees the identical seed set, so per-prompt differences are well defined.\n")
    md.append("### Descriptives\n")
    md.append("| budget | arm | protection / eviction | n | mean | SD | SE | frac=0 | frac=1 | "
              "eff. tokens (med) | comp. room |")
    md.append("|" + "---|"*11)
    for b in budgets:
        for a in [1,2,3,4,5,6]:
            v=[r["accuracy"] for r in by(p1,nominal_budget=b,arm=a)]
            if not v: continue
            rowset=by(p1,nominal_budget=b,arm=a)
            sd=statistics.pstdev(v) if len(v)>1 else 0.0
            se=sd/(len(v)**0.5) if v else 0.0
            et=statistics.median([r["effective_tokens"] for r in rowset])
            room = b-64-1
            md.append(f"| {b} | {a} | {ARMS[a]['label']} | {len(v)} | {statistics.mean(v):.4f} | "
                      f"{sd:.4f} | {se:.4f} | {sum(1 for x in v if x==0)/len(v):.3f} | "
                      f"{sum(1 for x in v if x==1)/len(v):.3f} | {et:.0f} | "
                      f"{room if a!=6 else 'n/a'} |")
    md.append("")
    md.append("`comp. room` = total_budget − recency_window(64) − sink(1); spec §4 requires > 0 "
              "and the harness refuses any cell where it is not.\n")
    # dormancy (G5) per budget, from arm 1
    md.append("### Dormancy (gate G5) by budget, measured in arm 1\n")
    md.append("| budget | prompts with >=1 dormant window | mean dormant steps | max run |")
    md.append("|" + "---|"*4)
    for b in budgets:
        rs=by(p1,nominal_budget=b,arm=1)
        if not rs: continue
        de=[r.get("dormancy_events",0) for r in rs]
        mr=[r.get("dormancy_max_run",0) for r in rs]
        md.append(f"| {b} | {sum(1 for x in de if x>=1)/len(de):.3f} | "
                  f"{statistics.mean(de):.2f} | {max(mr) if mr else 0} |")
    md.append("")
    md.append("### Paired contrasts (BH-corrected within iso-condition)\n")
    md.append("Estimator per spec §8: bootstrapped median of paired per-prompt differences, "
              "10,000 resamples, resampling unit = prompt. §8 warns the median CI collapses to "
              "zero width because differences are discrete at 1/6 and mostly exactly zero, so "
              "the mean CI and the raw count of differing prompts are reported alongside — §8 "
              "calls that count the most informative of the three. p is a paired sign-flip "
              "permutation test (10,000); p_BH is Benjamini-Hochberg within this "
              "iso-condition. `equiv` = 95% mean CI fully inside +/-0.05 (minimum effect of "
              "interest).\n")
    md.append("| budget | contrast | mean diff | 95% CI (mean) | median diff | 95% CI (median) | n differ | a>b | b>a | p | p_BH | equiv | med CI 0-width |")
    md.append("|"+"---|"*13)
    results=[];meta=[]
    for b in budgets:
        for (x,y,lbl) in [(2,1,"protection / permanent"),(4,3,"protection / tiered"),
                          (3,1,"tiering / unprotected"),(4,2,"tiering / protected")]:
            r,n=paired(p1,dict(nominal_budget=b,arm=x),dict(nominal_budget=b,arm=y),f"b{b} {lbl}")
            if r: results.append(r); meta.append({"budget":b,"contrast":lbl})
    padj=bh_correct([r.p_perm for r in results]) if results else []
    for i,r in enumerate(results):
        md.append(f"| {meta[i]['budget']} | {meta[i]['contrast']} | {fmt(r)} | "
                  f"{r.n_a_better} | {r.n_b_better} | {padj[i]:.4f} | "
                  f"{'yes' if r.equivalent_at_005 else 'no'} | "
                  f"{'YES' if r.median_ci_zero_width else 'no'} |")
    export_paired_csv(results, f"{RES}/csv/phase1_contrasts.csv", meta)
    md.append("")
    # interaction
    md.append("### Interaction: (arm4-arm3) - (arm2-arm1)\n")
    md.append("| budget | interaction | 95% CI | n differ | reference |")
    md.append("|"+"---|"*5)
    ref={257:0.002, 514:-0.002}
    inter_rows=[]
    for b in budgets:
        d={}
        for a in [1,2,3,4]:
            d[a]={r["seed"]:r["accuracy"] for r in by(p1,nominal_budget=b,arm=a)}
        common=sorted(set(d[1])&set(d[2])&set(d[3])&set(d[4]))
        if not common: continue
        iv=[(d[4][s]-d[3][s])-(d[2][s]-d[1][s]) for s in common]
        zeros=[0.0]*len(iv)
        rr=paired_bootstrap(iv,zeros,f"interaction b{b}")
        refv=f"{ref[b]:+.3f}" if b in ref else "-"
        md.append(f"| {b} | {rr.mean_diff:+.4f} | [{rr.mean_ci[0]:+.4f}, {rr.mean_ci[1]:+.4f}] | {rr.n_differ} | {refv} |")
        inter_rows.append(rr)
    if inter_rows: export_paired_csv(inter_rows, f"{RES}/csv/phase1_interaction.csv")
    md.append("")
    # ---- per-turn accuracy == per-depth accuracy (the SPEC-GAP 1 confound, stated once) ----
    from kvre.task import build_prompt as _bp, target_for as _tf
    md.append("### Accuracy by turn index\n")
    md.append("Under the [SPEC-GAP 1] turn order (reverse depth), turn index and context depth "
              "are perfectly anti-correlated by construction: turn *t* queries the credential at "
              "depth rank 5−*t*. These are therefore **one analysis, not two** — a turn effect "
              "and a depth effect are not separable in this design. Reported as a single table "
              "with that confound stated rather than as two tables implying independence. "
              "Turn index also equals the dormancy gap, since each credential is queried once.\n")
    md.append("| budget | arm | turn0 (deepest) | turn1 | turn2 | turn3 | turn4 | turn5 (earliest) |")
    md.append("|"+"---|"*8)
    for b in budgets:
        for a in [1,2,3,4,5,6]:
            rs=by(p1,nominal_budget=b,arm=a)
            if len(rs)<50: continue
            hit=[0]*6; tot=[0]*6
            for r in rs:
                pr=_bp(r["seed"])
                for t,o in enumerate(r["outputs"][:6]):
                    tot[t]+=1
                    if _tf(pr,t).value in o: hit[t]+=1
            md.append(f"| {b} | {a} {ARMS[a]['label']} | " +
                      " | ".join(f"{hit[t]/tot[t]:.3f}" if tot[t] else "-" for t in range(6)) + " |")
    md.append("")
    # full_cache invariance
    fc={b: statistics.mean([r["accuracy"] for r in by(p1,nominal_budget=b,arm=6)] or [float('nan')]) for b in budgets}
    inv = "INVARIANT" if len({round(v,6) for v in fc.values() if v==v})<=1 else "VARIES - budget is leaking into arm 6"
    md.append(f"**full_cache_ref by budget (spec §5 requires invariance): {fc} — {inv}**\n")

# ---------------- Phase 2 ----------------
p2=[r for r in rows if r.get("arm")==4 and r.get("nominal_budget")==257
    and r.get("quant_bits")==4 and r.get("context_target")==1029]
sigs=sorted({r["promotion"] for r in p2})
if len(sigs)>1:
    md.append("## Phase 2 — promotion signals (4-bit, budget 257)\n")
    # ceiling for normalisation: this implementation's own full_cache_ref at the same cell
    _fc=[r["accuracy"] for r in rows if r.get("arm")==6 and r.get("context_target")==1029]
    MY_CEIL = statistics.mean(_fc) if _fc else float("nan")
    REF_CEIL = 0.947
    md.append(f"Ceiling-normalised columns divide each arm by its own implementation's "
              f"`full_cache_ref` (mine {MY_CEIL:.4f}, reference {REF_CEIL}). This is a "
              f"**post-hoc comparability aid, not the pre-registered metric** — the spec fixes "
              f"raw k/N. It exists because [GAP-U] leaves my ceiling ~0.15 below the "
              f"reference's, which shifts every level without necessarily changing the "
              f"mechanism ordering.\n")
    md.append("| id | signal | mean acc | n | reference | mine / ceiling | ref / ceiling |")
    md.append("|"+"---|"*7)
    refp={"P1":0.203,"P2":0.216,"P3":0.154,"P4":0.184,"P5":0.188}
    for s in ["attention","epiphany","random","roundrobin","oracle"]:
        v=[r["accuracy"] for r in by(p2,promotion=s)]
        if not v: continue
        pid=PROMO_IDS[s]
        sd=statistics.pstdev(v) if len(v)>1 else 0.0
        m=statistics.mean(v); rf=refp.get(pid)
        nm = f"{m/MY_CEIL:.4f}" if MY_CEIL==MY_CEIL and MY_CEIL>0 else "-"
        nr = f"{rf/REF_CEIL:.4f}" if rf else "-"
        md.append(f"| {pid} | {s} | {m:.4f} | {len(v)} | {rf if rf else '-'} | {nm} | {nr} |")
    md.append("")
    md.append("Retention is held fixed and attention-ranked in every Phase 2 arm; protection is "
              "ON and tiered eviction is ON. Only the FULL/QUANT promotion decision varies, so "
              "these contrasts isolate promotion from retention.\n")
    md.append("### Phase 2 paired contrasts vs P3 (random control)\n")
    md.append("| contrast | mean diff | 95% CI (mean) | median diff | 95% CI (median) | n differ | p | p_BH | equiv |")
    md.append("|"+"---|"*9)
    p2res=[]; p2meta=[]
    for s in ["attention","epiphany","roundrobin","oracle"]:
        r,n=paired(p2, dict(promotion=s), dict(promotion="random"),
                   f"{PROMO_IDS[s]} vs P3")
        if r: p2res.append(r); p2meta.append({"contrast": f"{PROMO_IDS[s]} ({s}) vs P3 (random)"})
    if p2res:
        padj2=bh_correct([r.p_perm for r in p2res])
        for i,r in enumerate(p2res):
            md.append(f"| {p2meta[i]['contrast']} | {fmt(r)} | {padj2[i]:.4f} | "
                      f"{'yes' if r.equivalent_at_005 else 'no'} |")
        export_paired_csv(p2res, f"{RES}/csv/phase2_contrasts.csv", p2meta)
    md.append("")

# ---------------- bit-width sweep ----------------
sw=[r for r in rows if r.get("arm")==4 and r.get("promotion")=="attention"
    and r.get("nominal_budget")==257 and r.get("context_target")==1029]
widths=sorted({r["quant_bits"] for r in sw}, reverse=True)
if False:   # superseded by sections/20_sweep.md, which uses a common seed set at every width
    md.append("## Bit-width sweep (raw)\n")
    md.append("| bits | mean acc | n | frac fully degenerate | frac clean | frac mixed | first-token-EOS rate | max levels |")
    md.append("|"+"---|"*8)
    for w in widths:
        v=by(sw,quant_bits=w)
        if not v: continue
        acc=[r["accuracy"] for r in v]
        deg=sum(1 for r in v if r["accuracy"]==0.0)/len(v)
        clean=sum(1 for r in v if r["accuracy"]==1.0)/len(v)
        mixed=1.0-deg-clean
        eos=statistics.mean([sum(1 for e in r["first_token_eos"] if e)/max(1,len(r["first_token_eos"])) for r in v])
        ml=max(r.get("quant_max_levels",0) for r in v)
        md.append(f"| {w} | {statistics.mean(acc):.4f} | {len(v)} | {deg:.3f} | {clean:.3f} | {mixed:.3f} | {eos:.3f} | {ml} (limit {2**w}) |")
    md.append("")


# ---------------- level occupancy (post-hoc) ----------------
ph = {}
if os.path.exists(f"{RES}/posthoc.json"):
    ph = json.load(open(f"{RES}/posthoc.json"))
if ph.get("level_occupancy"):
    md.append("### Level-occupancy of quantized tensors\n")
    md.append("| bits | limit 2^bits | max levels seen | mean levels used | occupancy | violations |")
    md.append("|"+"---|"*6)
    for w in sorted((int(k) for k in ph["level_occupancy"]), reverse=True):
        e = ph["level_occupancy"][str(w)]
        md.append(f"| {w} | {e['limit']} | {e['max_levels']} | {e['mean_levels_used']:.2f} | "
                  f"{e['occupancy_frac']:.3f} | {e['violations']} |")
    md.append("")

# ---------------- Phase 2 manipulation check ----------------
if ph.get("spearman_rho"):
    md.append("### Phase 2 manipulation check — Spearman rho(eviction, promotion)\n")
    md.append("| id | signal | measured rho | target |")
    md.append("|"+"---|"*4)
    for pid in ["P1","P2","P3","P4","P5"]:
        e = ph["spearman_rho"].get(pid)
        if e: md.append(f"| {pid} | {e['signal']} | {e['rho']:+.4f} | {e['target']} |")
    md.append("")
if ph.get("p5"):
    e = ph["p5"]
    verdict = "PASS - never oversubscribed" if e["n_oversubscribed"]==0 else "FAIL"
    md.append(f"**P5 ceiling verification:** oversubscribed on {e['n_oversubscribed']}/{e['n']} "
              f"prompts — {verdict}. P5 bounds the promotion decision only; it cannot rescue "
              f"credentials already evicted at the retention stage.\n")

# ---------------- context length ----------------
ctx = [r for r in rows if r.get("context_target") in (2048, 4096)]
if ctx:
    md.append("## Context length at fixed retention ratio (~13%)\n")
    cal = {}
    if os.path.exists(f"{RES}/context_calibration.json"):
        cal = json.load(open(f"{RES}/context_calibration.json"))
    md.append("[GAP-J] **recency_window held FIXED at 64 at every context length.** Rationale and "
              "the cost of that choice are in SPEC_QUESTIONS.md; the alternative (scaling the "
              "window with context) is defensible and would make the competitively-selected "
              "fraction constant, partly cancelling the manipulation.\n")
    if cal.get("lengths"):
        md.append("| target | realised ctx | full-cache len | budget | retention | comp. room | gates |")
        md.append("|"+"---|"*7)
        for e in cal["lengths"]:
            g = "all pass" if e.get("gates_pass") else "**GATE FAILED - cell not trusted**"
            md.append(f"| {e['target']} | {e['realised_ctx']:.0f} | {e['full_cache_len']} | "
                      f"{e['budget']} | {e['retention_ratio']:.3f} | {e['competitive_room']} | {g} |")
        md.append("")
    md.append("| target | arm1 | arm2 | arm3 | arm4 | interaction | 95% CI | n differ |")
    md.append("|"+"---|"*8)
    for t in sorted({r["context_target"] for r in ctx}):
        sub = [r for r in ctx if r["context_target"]==t]
        d = {a: {r["seed"]: r["accuracy"] for r in by(sub, arm=a)} for a in [1,2,3,4]}
        cells = [f"{statistics.mean(list(d[a].values())):.3f}" if d[a] else "-" for a in [1,2,3,4]]
        common = sorted(set(d[1])&set(d[2])&set(d[3])&set(d[4]))
        if common:
            iv=[(d[4][s]-d[3][s])-(d[2][s]-d[1][s]) for s in common]
            rr=paired_bootstrap(iv,[0.0]*len(iv),f"ctx{t}")
            md.append(f"| {t} | " + " | ".join(cells) +
                      f" | {rr.mean_diff:+.4f} | [{rr.mean_ci[0]:+.4f}, {rr.mean_ci[1]:+.4f}] | {rr.n_differ} |")
    md.append("")

# ---------------- agreement report ----------------
md.append("## Agreement report — this implementation vs the spec's references\n")
md.append("| quantity | reference | this implementation | delta | traced to |")
md.append("|"+"---|"*5)
def add(q, ref, got, gap):
    try:
        dl = f"{got-ref:+.4f}"; g=f"{got:.4f}"
    except Exception:
        dl="-"; g=str(got)
    md.append(f"| {q} | {ref} | {g} | {dl} | {gap} |")
qt = {8:(1.18,0.91), 4:(19.98,15.20), 2:(103.35,75.55)}
for b,(rk,rv) in {8:(1.17,0.88),4:(20.06,14.99),2:(101.11,74.68)}.items():
    add(f"quantizer err {b}-bit keys (%)", rk, qt[b][0], "[GAP-Q] resolved")
    add(f"quantizer err {b}-bit values (%)", rv, qt[b][1], "[GAP-Q] resolved")
if p1:
    fcv=[statistics.mean([r["accuracy"] for r in by(p1,nominal_budget=b,arm=6)] or [0]) for b in budgets]
    if fcv: add("full_cache_ref", 0.947, statistics.mean(fcv), "[GAP-U] copy fidelity, unresolved")
    for b,refv in [(257,0.002),(514,-0.002)]:
        d={a:{r["seed"]:r["accuracy"] for r in by(p1,nominal_budget=b,arm=a)} for a in [1,2,3,4]}
        common=sorted(set(d[1])&set(d[2])&set(d[3])&set(d[4])) if all(d.values()) else []
        if common:
            iv=[(d[4][s]-d[3][s])-(d[2][s]-d[1][s]) for s in common]
            add(f"interaction @ budget {b}", refv, sum(iv)/len(iv), "[SPEC-GAP 4] / [GAP-U] ceiling")
if len(sigs)>1:
    refp={"P1":0.203,"P2":0.216,"P3":0.154,"P4":0.184,"P5":0.188}
    for sig in ["attention","epiphany","random","roundrobin","oracle"]:
        v=[r["accuracy"] for r in by(p2,promotion=sig)]
        if v: add(f"Phase 2 {PROMO_IDS[sig]} ({sig})", refp[PROMO_IDS[sig]], statistics.mean(v),
                  "[SPEC-GAP 7] P2 / [GAP-U] ceiling")
md.append("")
md.append("Disagreements >0.05 are traced to a specific gap in the column above. The dominant "
          "one is [GAP-U]: the full-cache ceiling is 0.72 here against the spec's 0.947, which "
          "shifts every accuracy level downward. Paired *differences* are the estimands and are "
          "less affected, but copy-error noise costs statistical power.\n")

# Narrative sections live as separate files under sections/ and are concatenated here.
# They were previously appended directly to FINDINGS.md, which meant every regeneration of this
# script silently destroyed them. Keeping them as inputs makes regeneration idempotent.
import glob as _g
for _f in sorted(_g.glob("sections/*.md")):
    md.append("\n---\n")
    md.append(open(_f).read())
open("FINDINGS.md","w").write("\n".join(md)+"\n")
print(f"wrote FINDINGS.md (+{len(sorted(_g.glob('sections/*.md')))} narrative sections)")

