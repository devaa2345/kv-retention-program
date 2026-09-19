"""Recompute G_m and I under both readings of `floor_pos`, and say whether the signs survive.

READING A (as run, PREREG sec 3.12(b)): floor_pos retains B = C + 72
READING B (PREREG sec 4.1 taken literally): floor_pos retains C

Only A_floor differs. A_m, A_causal and A_presc are read unchanged from the main grid, because
the oracle and method arms all retain B independently of how the floor ARM is parameterised
(`ladder._floors` + `_pad_from_floor`; `methods.make_floor_constrained`).

Every contrast is paired per instance and the bootstrap is propagated THROUGH the ratio:
numerator and denominator are recomputed on the same resample, per PREREG sec 5. The
refusal-to-normalise rule (headroom < 0.15 -> report raw only) is applied to each reading
separately, since the headroom differs between them.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 50_000
REFUSE = 0.15
LADDER = {"full_cache", "null", "random", "floor_pos", "oracle_causal", "oracle_prescient"}


def load_grid(tag):
    """(C, arm) -> {instance_id: score} from the canonical agnostic grid."""
    by = defaultdict(dict)
    p = ROOT / f"runs/nvidia/grid_{tag}_ledger_agnostic.jsonl"
    for line in p.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        by[(r["C"], r["key"]["arm"])][r["key"]["instance_id"]] = r["score"]
    return by


def load_floorC(tag):
    by = defaultdict(dict)
    meta = {}
    p = ROOT / f"runs/nvidia/floor_readingC_{tag}.jsonl"
    if not p.exists():
        return by, meta
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        by[r["C"]][r["key"]["instance_id"]] = r["score"]
        meta.setdefault(r["C"], dict(n_kept_C=r["n_kept_readingC"],
                                     n_kept_B=r["n_kept_readingB"],
                                     contains_floor=r["readingC_contains_mandatory_floor"],
                                     missing=r["n_mandatory_missing_in_readingC"]))
    return by, meta


def boot_ratio(num_a, num_b, den_a, den_b, seed):
    """CI on (mean(num_a)-mean(num_b)) / (mean(den_a)-mean(den_b)), paired, through the ratio."""
    rng = np.random.default_rng(seed)
    n = len(num_a)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    num = num_a[idx].mean(1) - num_b[idx].mean(1)
    den = den_a[idx].mean(1) - den_b[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den != 0, num / den, np.nan)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main() -> int:
    report = {}
    print("=" * 108)
    print("FLOOR SENSITIVITY  --  does the paper survive both readings of floor_pos?")
    print("=" * 108)

    for tag in ("M2", "M3"):
        grid = load_grid(tag)
        floorC, meta = load_floorC(tag)
        if not floorC:
            print(f"\n### {tag}: no Reading-B records yet")
            continue
        budgets = sorted(floorC)
        methods = sorted({a for (_, a) in grid if a not in LADDER})
        print(f"\n{'#'*100}\n### {tag}   methods: {', '.join(methods)}\n{'#'*100}")

        # ---- what the two readings actually retain -------------------------
        print(f"\n  {'C':>5} {'keptA(B)':>9} {'keptB(C)':>9} {'ratio':>7} "
              f"{'B contains mandatory 8+64 floor?':>34}")
        for C in budgets:
            m = meta[C]
            note = "yes" if m["contains_floor"] else f"NO - {m['missing']} mandatory tokens absent"
            print(f"  {C:5d} {m['n_kept_B']:9d} {m['n_kept_C']:9d} "
                  f"{m['n_kept_C']/m['n_kept_B']:7.2f} {note:>34}")

        cells = {}
        print(f"\n  {'C':>5} {'A_floor A':>10} {'A_floor B':>10} {'headroom A':>11} "
              f"{'headroom B':>11} {'I (A)':>16} {'I (B)':>16}")
        for C in budgets:
            ids = sorted(set(floorC[C]) & set(grid[(C, "floor_pos")])
                         & set(grid[(C, "oracle_causal")]) & set(grid[(C, "oracle_prescient")]))
            if not ids:
                continue
            fa = np.array([grid[(C, "floor_pos")][i] for i in ids])
            fb = np.array([floorC[C][i] for i in ids])
            ca = np.array([grid[(C, "oracle_causal")][i] for i in ids])
            pr = np.array([grid[(C, "oracle_prescient")][i] for i in ids])

            hA, hB = ca.mean() - fa.mean(), ca.mean() - fb.mean()
            IA = (pr.mean() - ca.mean()) / (pr.mean() - fa.mean()) if pr.mean() != fa.mean() else 0.0
            IB = (pr.mean() - ca.mean()) / (pr.mean() - fb.mean()) if pr.mean() != fb.mean() else 0.0
            IAci = boot_ratio(pr, ca, pr, fa, seed=C)
            IBci = boot_ratio(pr, ca, pr, fb, seed=C + 1)
            print(f"  {C:5d} {fa.mean():10.4f} {fb.mean():10.4f} {hA:11.4f} {hB:11.4f} "
                  f"{IA:6.4f}[{IAci[0]:.3f},{IAci[1]:.3f}] {IB:6.4f}[{IBci[0]:.3f},{IBci[1]:.3f}]")

            cell = dict(C=C, n=len(ids),
                        A_floor_A=round(float(fa.mean()), 4), A_floor_B=round(float(fb.mean()), 4),
                        headroom_A=round(float(hA), 4), headroom_B=round(float(hB), 4),
                        refused_A=bool(hA < REFUSE), refused_B=bool(hB < REFUSE),
                        I_A=round(float(IA), 4), I_A_ci=[round(x, 4) for x in IAci],
                        I_B=round(float(IB), 4), I_B_ci=[round(x, 4) for x in IBci],
                        G_m={})
            for meth in methods:
                if not all(i in grid[(C, meth)] for i in ids):
                    continue
                am = np.array([grid[(C, meth)][i] for i in ids])
                gA = (am.mean() - fa.mean()) / hA if hA != 0 else float("nan")
                gB = (am.mean() - fb.mean()) / hB if hB != 0 else float("nan")
                gAci = boot_ratio(am, fa, ca, fa, seed=C + 7)
                gBci = boot_ratio(am, fb, ca, fb, seed=C + 8)
                cell["G_m"][meth] = dict(
                    A_m=round(float(am.mean()), 4),
                    G_A=round(float(gA), 4), G_A_ci=[round(x, 4) for x in gAci],
                    G_B=round(float(gB), 4), G_B_ci=[round(x, 4) for x in gBci])
            cells[C] = cell

        # ---- G_m table, both readings -------------------------------------
        print(f"\n  G_m under each reading  (A = as published, B = sec 4.1 literal)")
        print(f"  {'C':>5} {'method':16} {'A_m':>8} {'G_m (A)':>20} {'G_m (B)':>20} {'sign flips?':>12}")
        flips = 0
        total = 0
        for C in budgets:
            if C not in cells:
                continue
            for meth, g in sorted(cells[C]["G_m"].items()):
                total += 1
                flip = (g["G_A"] <= 0) != (g["G_B"] <= 0)
                flips += int(flip)
                print(f"  {C:5d} {meth:16} {g['A_m']:8.4f} "
                      f"{g['G_A']:+7.4f}[{g['G_A_ci'][0]:+.3f},{g['G_A_ci'][1]:+.3f}] "
                      f"{g['G_B']:+7.4f}[{g['G_B_ci'][0]:+.3f},{g['G_B_ci'][1]:+.3f}] "
                      f"{('YES' if flip else '-'):>12}")
        report[tag] = dict(cells=cells, n_method_cells=total, n_sign_flips=flips)
        print(f"\n  {tag}: {total - flips} of {total} method-cells keep their sign; "
              f"{flips} flip.")

    # ---------------- verdict ------------------------------------------------
    print("\n" + "=" * 108)
    print("VERDICT")
    print("=" * 108)
    tot = sum(r["n_method_cells"] for r in report.values())
    fl = sum(r["n_sign_flips"] for r in report.values())
    leA = leB = 0
    for tag, r in report.items():
        for C, c in r["cells"].items():
            for meth, g in c["G_m"].items():
                leA += int(g["G_A"] <= 0)
                leB += int(g["G_B"] <= 0)
    print(f"  G_m <= 0 under Reading A (published): {leA} of {tot} method-cells")
    print(f"  G_m <= 0 under Reading B (sec 4.1)  : {leB} of {tot} method-cells")
    print(f"  sign flips between readings          : {fl}")
    print("\n  I at binding budgets:")
    for tag, r in report.items():
        for C, c in sorted(r["cells"].items()):
            if c["I_A"] > 0.01 or c["I_B"] > 0.01:
                print(f"    {tag} C={C:<4d} I(A)={c['I_A']:.4f} {c['I_A_ci']}   "
                      f"I(B)={c['I_B']:.4f} {c['I_B_ci']}"
                      f"{'   [B REFUSED: headroom<0.15]' if c['refused_B'] else ''}")

    out = ROOT / "gates/nvidia/floor_sensitivity.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
