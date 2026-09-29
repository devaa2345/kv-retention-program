"""Score tier-1 probes against CANDIDATES.md (commit 3156e6c).

Every conjunct of every REAL rule is printed separately -- no rule is summarised into a single
boolean, because a script that implemented half a rule already produced a wrong verdict once
(MECHANISM_RESULTS.md, C1).
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
OUT = HERE / "out"
R_BOOT = 10000

# candidate -> (arm, comparator, direction, extra point-estimate conjuncts)
RULES = {
    "A1 head-consensus": ("consensus-snapkv", "U-snapkv", +1, [("ge_floor", "floor_pos")]),
    "A2 early-prior": ("early-snapkv", "U-snapkv", +1, []),
    "A3 floor-hybrid": ("hybrid-snapkv", "floor_pos", +1, [("ge_u", "U-snapkv")]),
    "A4 supra-units": ("supra-snapkv", "U-snapkv", +1, []),
    "A4 sub-units (directional)": ("sub-snapkv", "U-snapkv", +1, []),
    "A5 sink32": ("floor_sink32", "floor_pos", +1, []),
    "A6 gather-order (expect NOT-IT)": ("ascend-snapkv", "snapkv", 0, []),
}


def load(tag):
    p = RUNS / f"p4_probe_{tag}.jsonl"
    if not p.exists():
        return {}, 0
    rows, sess = {}, set()
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["arm"], r["instance"])] = r
            sess.add(r["session_id"])
    if len(sess) > 1:
        raise SystemExit(f"{tag}: probe rows span {len(sess)} sessions -- not scorable")
    n = 1 + max(i for _, i in rows)
    return rows, n


def boot(d, key):
    d = np.asarray(d, float)
    rng = np.random.default_rng(zlib.crc32(key.encode()) & 0xFFFFFFFF)
    m = d[rng.integers(0, len(d), size=(R_BOOT, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    L = ["Tier-1 probe results, scored against CANDIDATES.md (commit 3156e6c).",
         "n=20 screen: CI half-width ~0.10, so NOT-IT means 'no LARGE effect', never 'no effect'.",
         ""]
    per_model = {}
    for tag in ("M2", "M3"):
        rows, n = load(tag)
        if not rows:
            continue
        arms = sorted({a for a, _ in rows})
        L.append(f"== {tag} (n={n}) arms: {len(arms)}")
        acc = {a: [rows[(a, i)]["score"] for i in range(n)] for a in arms}
        L.append("   accuracy: " + "  ".join("%s %.3f" % (a, np.mean(v))
                                             for a, v in sorted(acc.items(),
                                                                key=lambda kv: -np.mean(kv[1]))))
        res = {}
        for name, (arm, comp, direction, extras) in RULES.items():
            if arm not in acc or comp not in acc:
                continue
            d = boot([acc[arm][i] - acc[comp][i] for i in range(n)], f"probe|{tag}|{arm}|{comp}")
            excl = d[1] > 0 or d[2] < 0
            conj = {"paired_CI_excludes_zero_positive": bool(d[1] > 0),
                    "paired_CI_excludes_zero": bool(excl)}
            for label, other in extras:
                conj[f"{label} ({arm} >= {other})"] = bool(np.mean(acc[arm]) >= np.mean(acc[other]))
            if direction == 0:      # A6 hygiene: identity expected
                ident = sum(1 for i in range(n) if rows[(arm, i)]["gen"] == rows[(comp, i)]["gen"])
                conj["generations_byte_identical"] = f"{ident}/{n}"
                conj["hygiene_violation"] = bool(excl or ident < n)
            res[name] = dict(arm=arm, comp=comp, delta=d, conjuncts=conj,
                             acc_arm=float(np.mean(acc[arm])), acc_comp=float(np.mean(acc[comp])))
            L.append(f"   {name}: {arm} {np.mean(acc[arm]):.3f} vs {comp} "
                     f"{np.mean(acc[comp]):.3f}  delta %+.3f [%+.3f, %+.3f]" % d[0:3])
            for k, v in conj.items():
                L.append(f"       {k}: {v}")
        per_model[tag] = res
        L.append("")

    L.append("== VERDICTS (replication rule: must clear on BOTH models)")
    for name in RULES:
        m2 = per_model.get("M2", {}).get(name)
        m3 = per_model.get("M3", {}).get(name)
        if not m2:
            continue
        if name.startswith("A6"):
            v = ("NOT-IT (no hygiene violation -- gather order is inert)"
                 if not m2["conjuncts"]["hygiene_violation"] else "REAL (HYGIENE VIOLATION)")
            if m3:
                v += f"; M3 {'clean' if not m3['conjuncts']['hygiene_violation'] else 'VIOLATION'}"
            L.append(f"  {name}: {v}")
            continue
        first = m2["conjuncts"]["paired_CI_excludes_zero_positive"]
        extras_ok = all(v for k, v in m2["conjuncts"].items()
                        if k not in ("paired_CI_excludes_zero_positive", "paired_CI_excludes_zero"))
        if not first:
            L.append(f"  {name}: NOT-IT on M2 (discovery model) -- delta {m2['delta'][0]:+.3f} "
                     f"[{m2['delta'][1]:+.3f}, {m2['delta'][2]:+.3f}]")
        elif m3 is None:
            L.append(f"  {name}: clears M2 (delta {m2['delta'][0]:+.3f}), "
                     f"other conjuncts {'met' if extras_ok else 'NOT met'} -- "
                     "UNREPLICATED, M3 required before it counts")
        else:
            rep = m3["conjuncts"]["paired_CI_excludes_zero_positive"]
            extras3 = all(v for k, v in m3["conjuncts"].items()
                          if k not in ("paired_CI_excludes_zero_positive",
                                       "paired_CI_excludes_zero"))
            L.append(f"  {name}: M2 clears; M3 {'clears' if rep else 'null'}; "
                     f"extra conjuncts M2 {'met' if extras_ok else 'NOT met'}, "
                     f"M3 {'met' if extras3 else 'NOT met'} -> "
                     + ("REAL" if (rep and extras_ok and extras3) else
                        "AMBIGUOUS (unreplicated or a conjunct failed)"))
    OUT.mkdir(exist_ok=True)
    (OUT / "probe_tier1.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "probe_tier1.json").write_text(json.dumps(per_model, indent=1, default=float),
                                          encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
