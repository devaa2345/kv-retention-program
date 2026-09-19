"""Natural-text vs synthetic LEDGER: method-to-floor ratio at matched approximate cost.

Natural ratios: out/natC_report.json (this experiment, paired bootstrap, 50,000 draws).
LEDGER ratios:  out/submission_audit_20260918.json `ratio_intervals` (Paper 3 Stage 4 plane,
                paired bootstrap, 20,000 draws, admissibility mask applied). Both are
                pointwise 95% intervals, conditional on the admissibility rule.
Two budget matchings, because the contexts differ (natural 4096, LEDGER 2048):
  same absolute C   natural C=512 vs LEDGER C=512   (12.5% vs 25% of context)
  same fraction     natural C=512 vs LEDGER C=256   (12.5% of context each)
Cost matching: natural level 1 (c ~ 17) vs LEDGER c ~ 19; level 3 (c ~ 26-28) has no LEDGER
partner (LEDGER c in {1, 8, 19, 40}); level 5 (c ~ 41-46) vs LEDGER c ~ 40.
"""
import json

NAT = json.load(open("out/natC_report.json"))
LED = json.load(open("out/submission_audit_20260918.json"))["ratio_intervals"]
METHODS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff")
NAT_C = {"M2": {"1": 16.6, "3": 27.8, "5": 46.1}, "M3": {"1": 16.6, "3": 25.8, "5": 41.1}}


def fmt(r, ci, status=""):
    if status == "X":
        return "X (floor<0.05)"
    if r is None:
        return "n/a"
    return f"{r:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def ledger(model, method, target_c, C):
    rows = [r for r in LED if r["model"] == model and r["method"] == method and r["C"] == C]
    if not rows:
        return None
    r = min(rows, key=lambda r: abs(r["c"] - target_c))
    if abs(r["c"] - target_c) > 3:
        return None
    return r


def nat(model, lv, C, method):
    for r in NAT[model]:
        if r["level"] == lv and r["C"] == C and r["arm"] == method:
            return r


out = []
P = out.append
P("# Natural text vs LEDGER: method / floor_pos accuracy ratio at matched cost\n")
P("R = A_method / A_floor_pos; R < 1 means the method loses to the positional baseline. "
  "Pointwise 95% paired-bootstrap intervals over instances. `X` = floor < 0.05, cell excluded "
  "by the frozen admissibility rule; `V` = all compressed arms <= 0.02 (vacuous, reported raw). "
  "Natural-text cells marked `†` are marginal (M3 level 1 anchor 0.968 vs 0.97 ceiling; "
  "all C=256 cells).\n")
for cost_pair, lv, target in (("c ~ 17 (natural L1) vs LEDGER c ~ 19", "1", 19),
                              ("c ~ 41-46 (natural L5) vs LEDGER c ~ 40", "5", 40)):
    for label, nC, lC in (("same absolute C = 512", 512, 512),
                          ("same fraction of context (natural 512 / LEDGER 256)", 512, 256)):
        P(f"\n## {cost_pair}; {label}\n")
        P("| model | method | natural R (c) | LEDGER R (c) | natural floor | LEDGER floor |")
        P("|---|---|---|---|---|---|")
        for m in ("M2", "M3"):
            for meth in METHODS:
                n = nat(m, lv, nC, meth)
                l = ledger(m, meth, target, lC)
                if n is None:
                    continue
                dag = "†" if (m == "M3" and lv == "1") else ""
                ntxt = fmt(n["R"], n["R_ci"], n["status"]) + f" ({NAT_C[m][lv]}){dag}" + (" V" if n["status"] == "V" else "")
                if l is None:
                    ltxt, lf = "not in plane", ""
                else:
                    ltxt = (fmt(l["ratio"], l["ratio_ci"], "X" if l["status"] != "admissible" else "")
                            + f" ({l['c']:.1f})" + (" V" if l["status"] == "vacuous" else ""))
                    lf = f"{l['floor_accuracy']:.3f}"
                P(f"| {m} | {meth} | {ntxt} | {ltxt} | {n['floor']:.3f} | {lf} |")
open("out/NATURAL_VS_LEDGER.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
print("\nLEDGER status values present:", sorted({r["status"] for r in LED}))
