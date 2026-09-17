"""Paper 4 Stage 2 analysis. Rules are STAGE12_RULES.md, committed before any run.

    python stage2_analyse.py verify    -> out/stage2_verify.txt   exit 0 PASS / 1 FAIL
    python stage2_analyse.py pilot     -> out/stage2_pilot.txt    exit 0 always (verdict in text)

All numbers produced on the RTX 5070 (Machine N).
"""
from __future__ import annotations

import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
P3RUNS = HERE.parent / "paper3" / "runs" / "nvidia"
OUT = HERE / "out"
TAGS = ("M2", "M3")
C = 512
N = 50
R_BOOT = 10000
FLOOR_DEGENERATE = 0.05
PAIRS = (("snapkv", "U-snapkv"), ("adakv_snapkv", "U-adakv_snapkv"))
HDR = "Produced on RTX 5070 (Machine N). Paper 4 Stage 2."


def jl(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def boot(vals, key, stat=np.mean):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(zlib.crc32(("p4|boot|" + key).encode()) & 0xFFFFFFFF)
    idx = rng.integers(0, len(v), size=(R_BOOT, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(num, den, key):
    a, b = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(zlib.crc32(("p4|ratio|" + key).encode()) & 0xFFFFFFFF)
    idx = rng.integers(0, len(a), size=(R_BOOT, len(a)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a[idx].mean(1) / b[idx].mean(1)
    r = r[np.isfinite(r)]
    point = a.mean() / b.mean() if b.mean() > 0 else float("nan")
    return float(point), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def fmt(t, p=3):
    return ("%+.*f [%+.*f, %+.*f]" % (p, t[0], p, t[1], p, t[2]))


# ------------------------------------------------------------------------------ verify
def verify():
    lines = [HDR, "Package: verify (prefill only, real scores). Blocking gate before generation.", ""]
    ok = True
    fields = ("p_g", "q_complete", "units_complete", "units_touched")
    for tag in TAGS:
        rows = jl(RUNS / f"p4_verify_{tag}.jsonl")
        fails = jl(RUNS / f"p4_verify_{tag}.failures.jsonl")
        ids = list({(r["label"], r["arm"], r["instance"]): r
                    for r in jl(RUNS / f"p4_identity_{tag}.jsonl")}.values())
        lines.append(f"== {tag}: {len(rows)} rows, {len(fails)} failures, {len(ids)} identity checks")
        exp = N * 2 * 8
        if len(rows) != exp or fails:
            ok = False
            lines.append(f"  COVERAGE FAIL: expected {exp} rows and 0 failures")
        by = {(r["label"], r["arm"], r["instance"]): r for r in rows}
        # (a) reproduction of Stage 4's stored captures
        stored = {}
        for r in jl(P3RUNS / f"stage4_capture_{tag}.jsonl"):
            if r["C"] == C and r["label"] in ("c=1", "c=40") and r["arm"] in (
                    "snapkv", "adakv_snapkv", "expected_attn", "keydiff"):
                i = int(r["key"]["instance_id"][3:])
                if i < N:
                    stored[(r["label"], r["arm"], i)] = r
        n_cmp = n_bad = 0
        worst = 0.0
        for k, s in stored.items():
            if k not in by:
                continue
            n_cmp += 1
            d = max(abs(by[k][f] - s[f]) for f in fields)
            worst = max(worst, d)
            n_bad += d > 1e-9
        lines.append(f"  (a) X reproduces Stage 4 stored captures: {n_cmp - n_bad}/{n_cmp} exact "
                     f"(max abs diff {worst:.2e})")
        a_ok = n_bad == 0 and n_cmp == N * 2 * 4
        if not a_ok:
            # AMENDMENT 1 (STAGE12_RULES.md section 5): where the stored captures come from a
            # different session, (a) is satisfied by fresh-process determinism evidence instead.
            det = OUT / f"stage2_det_check_{tag}.json"
            if det.exists():
                dr = json.loads(det.read_text(encoding="utf-8"))["rows"]
                rer = sum(r["rerun_keepsets_identical"] for r in dr)
                fs = sum(r["fresh_vs_session"] <= 1e-9 for r in dr)
                fst = sum(r["fresh_vs_stored"] <= 1e-9 for r in dr)
                a_ok = len(dr) > 0 and rer == len(dr) and fs == len(dr) and n_cmp == N * 2 * 4
                lines.append(f"      AMENDMENT 1: fresh-process determinism: rerun identical {rer}/{len(dr)}, "
                             f"fresh == this session {fs}/{len(dr)}, fresh == stored {fst}/{len(dr)} "
                             f"-> {'harness deterministic; mismatch is the stored session' if a_ok else 'NOT deterministic'}")
        ok &= a_ok
        # (b) identity
        id_ok = all(r["identity_ok"] for r in ids) and len(ids) == 2 * 2 * 4
        ex = sum(r["slots_exact"] for r in ids)
        tot = sum(r["slots"] for r in ids)
        lines.append(f"  (b) U-X with singleton units == X up to exact ties: "
                     f"{sum(r['identity_ok'] for r in ids)}/{len(ids)} checks; "
                     f"{ex}/{tot} slots set-identical")
        ok &= id_ok
        # (c) direction
        lines.append("  (c) keep-set direction at matched budget (means over 50 instances; paired "
                     "U-X minus X with 95% bootstrap CI)")
        lines.append("      %-5s %-14s %8s %8s %24s %8s %8s %24s %8s %8s %4s" % (
            "cell", "method", "X touch", "U touch", "d touched", "X compl", "U compl",
            "d complete", "X q_c", "U q_c", "fb"))
        for label in ("c=40", "c=1"):
            for m in ("snapkv", "adakv_snapkv", "expected_attn", "keydiff"):
                xs = [by[(label, m, i)] for i in range(N) if (label, m, i) in by]
                us = [by[(label, "U-" + m, i)] for i in range(N) if (label, "U-" + m, i) in by]
                if len(xs) != N or len(us) != N:
                    continue
                dt = boot([u["units_touched"] - x["units_touched"] for u, x in zip(us, xs)], f"v|{tag}|{label}|{m}|t")
                dc = boot([u["units_complete"] - x["units_complete"] for u, x in zip(us, xs)], f"v|{tag}|{label}|{m}|c")
                fb = sum(u["fallback"] for u in us)
                good = dt[0] < 0 and dc[0] > 0
                if label == "c=40":
                    ok &= good
                lines.append("      %-5s %-14s %8.2f %8.2f %24s %8.2f %8.2f %24s %8.3f %8.3f %4d %s" % (
                    label, m, np.mean([x["units_touched"] for x in xs]),
                    np.mean([u["units_touched"] for u in us]), fmt(dt, 2),
                    np.mean([x["units_complete"] for x in xs]),
                    np.mean([u["units_complete"] for u in us]), fmt(dc, 2),
                    np.mean([x["q_complete"] for x in xs]), np.mean([u["q_complete"] for u in us]),
                    fb, ("ok" if good else "NOT AS PREDICTED") if label == "c=40" else "(control, no prediction)"))
        lines.append("")
    lines.append("VERIFY GATE: %s" % ("PASS" if ok else "FAIL -- stop, do not generate"))
    OUT.mkdir(exist_ok=True)
    (OUT / "stage2_verify.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


# ------------------------------------------------------------------------------ pilot
def pilot():
    L = [HDR, "Package: pilot (generation). LEDGER-C c~40 and MARK-1 c=1, C=512, n=50 per cell.",
         "Paired per instance; 95% percentile bootstrap, 10,000 resamples, CRC32-seeded.", ""]
    res = dict(cells={})
    gate_c40 = []
    gate_c1 = []
    for tag in TAGS:
        rows = jl(RUNS / f"p4_pilot_{tag}.jsonl")
        fails = jl(RUNS / f"p4_pilot_{tag}.failures.jsonl")
        by = {(r["label"], r["arm"], r["instance"]): r for r in rows}
        L.append(f"==================== {tag}: {len(rows)} rows, {len(fails)} failures")
        for label in ("c=40", "c=1"):
            arms = ("floor_pos",) + tuple(a for p in PAIRS for a in p)
            got = {a: [by.get((label, a, i)) for i in range(N)] for a in arms}
            complete = [i for i in range(N) if all(got[a][i] is not None for a in arms)]
            acc = {a: [got[a][i]["score"] for i in complete] for a in arms}
            n = len(complete)
            if n == 0:
                L.append(f"  {label}: no complete instances")
                continue
            fl = float(np.mean(acc["floor_pos"]))
            degen = fl < FLOOR_DEGENERATE
            L.append(f"  ---- {label}  (n={n} complete instances; floor_pos acc {fl:.3f}"
                     f"{' -> DEGENERATE, EXCLUDED' if degen else ''})")
            L.append("    accuracy: " + "  ".join("%s %.3f" % (a, np.mean(acc[a])) for a in arms))
            cell = dict(n=n, floor=fl, degenerate=degen, acc={a: float(np.mean(acc[a])) for a in arms})
            L.append("    [1] U-X vs X at matched budget (paired accuracy difference)")
            for x, u in PAIRS:
                d = boot([a - b for a, b in zip(acc[u], acc[x])], f"p|{tag}|{label}|{x}|acc")
                excl = d[1] > 0 or d[2] < 0
                L.append("        %-28s %s  %s" % (f"{u} - {x}", fmt(d),
                                                   "CI excludes 0" if excl else "CI includes 0"))
                cell[f"d_acc|{x}"] = d
                if not degen:
                    (gate_c40 if label == "c=40" else gate_c1).append((tag, x, d))
            L.append("    [2] method vs floor_pos (ratio of mean accuracy, method/floor; <1 = loses)")
            for a in arms[1:]:
                r = boot_ratio(acc[a], acc["floor_pos"], f"p|{tag}|{label}|{a}|ratio")
                dd = boot([p - q for p, q in zip(acc[a], acc["floor_pos"])], f"p|{tag}|{label}|{a}|dfloor")
                loss = (1 / r[0]) if r[0] > 0 else float("inf")
                L.append("        %-16s ratio %.3f [%.3f, %.3f]  (floor/method %.2fx)  diff %s" % (
                    a, r[0], r[1], r[2], loss, fmt(dd)))
                cell[f"ratio|{a}"] = r
                cell[f"dfloor|{a}"] = dd
            L.append("    [3] completion from the SAME keep-sets that generated (mean over slots)")
            L.append("        %-16s %9s %9s %9s %8s %8s %6s" % ("arm", "touched", "complete", "q_compl", "q_any", "p_g", "fb"))
            for a in arms:
                rr = [got[a][i] for i in complete]
                L.append("        %-16s %9.2f %9.2f %9.3f %8.3f %8.3f %6s" % (
                    a, np.mean([r["units_touched"] for r in rr]), np.mean([r["units_complete"] for r in rr]),
                    np.mean([r["q_complete"] for r in rr]), np.mean([r["q_any"] for r in rr]),
                    np.mean([r["p_g"] for r in rr]),
                    "-" if rr[0].get("fallback") is None else str(sum(r["fallback"] for r in rr))))
            for x, u in PAIRS:
                for f in ("units_complete", "units_touched", "q_complete"):
                    d = boot([got[u][i][f] - got[x][i][f] for i in complete], f"p|{tag}|{label}|{x}|{f}")
                    L.append("        d %-10s %-15s %s" % (f"{u}-{x}"[:10], f, fmt(d, 3)))
                    cell[f"d_{f}|{x}"] = d
            # conversion: accuracy per whole queried fact retained
            for a in arms:
                qc = np.mean([got[a][i]["q_complete"] for i in complete])
                cell[f"conv|{a}"] = float(np.mean(acc[a]) / qc) if qc > 0 else None
            L.append("    conversion acc/q_complete: " + "  ".join(
                "%s %s" % (a, "n/a" if cell[f"conv|{a}"] is None else "%.2f" % cell[f"conv|{a}"]) for a in arms))
            res["cells"][f"{tag}|{label}"] = cell
            L.append("")
        # session check vs Paper 3 stored generations
        stored = defaultdict(dict)
        for r in jl(P3RUNS / f"stage4_plane_{tag}.jsonl"):
            if r["C"] == C and r["label"] in ("c=40", "c=1") and r["arm"] in ("floor_pos", "snapkv", "adakv_snapkv"):
                i = int(r["key"]["instance_id"][3:])
                if i < N:
                    stored[(r["label"], r["arm"])][i] = r["gen"]
        L.append("  session check: X-arm generations byte-identical to Paper 3 Stage 4's stored ones")
        for (label, arm), g in sorted(stored.items()):
            same = sum(1 for i, gen in g.items() if (label, arm, i) in by and by[(label, arm, i)]["gen"] == gen)
            have = sum(1 for i in g if (label, arm, i) in by)
            L.append(f"    {label:5s} {arm:14s} {same}/{have}")
        L.append("")

    L.append("==================== GATE (STAGE12_RULES.md section 4)")
    pos40 = [(t, x, d) for t, x, d in gate_c40 if d[1] > 0]
    eff1 = [(t, x, d) for t, x, d in gate_c1 if d[1] > 0 or d[2] < 0]
    L.append("  c=40 pairs with U-X > X, CI excluding 0: %s" % (
        ", ".join(f"{t}/{x} {fmt(d)}" for t, x, d in pos40) or "NONE"))
    L.append("  c=1 pairs with CI excluding 0 (must be none): %s" % (
        ", ".join(f"{t}/{x} {fmt(d)}" for t, x, d in eff1) or "none"))
    L.append(f"  admissible comparisons: c=40 {len(gate_c40)}, c=1 {len(gate_c1)} (4 each if no degenerate cell); "
             "no multiplicity correction -- 4 chances at c=40, stated.")
    if eff1:
        verdict = "STOP AND DIAGNOSE: the c=1 control shows an effect"
    elif pos40:
        verdict = "PASS"
    else:
        verdict = "FAIL: U-X does not beat X at c=40 on either model (negative result, reported as such)"
    L.append("  VERDICT: " + verdict)
    res["gate"] = dict(verdict=verdict, c40_positive=[(t, x, d) for t, x, d in pos40],
                       c1_effects=[(t, x, d) for t, x, d in eff1])
    (OUT / "stage2_pilot.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "stage2_pilot.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(verify() if sys.argv[1] == "verify" else pilot())
