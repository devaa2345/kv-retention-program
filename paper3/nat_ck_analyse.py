"""ChunkKV on the natural-text dataset vs the four admitted methods (no GPU).

Rows: runs/nvidia/natC_<tag>_C*.jsonl (Experiment C) and runs/nvidia/natCK_<tag>_C*.jsonl
(ChunkKV, same instances, same scorer v2). R = A_method / A_floor_pos with paired instance
bootstrap (50,000 draws, same resample for every arm in a cell). Marginal cells (M3 level 1,
all C = 256) are daggered; V = all compressed arms (incl. ChunkKV) <= 0.02.
ChunkKV is PROVISIONAL (G1-G3 not run) and under-budget by 0-19 tokens; see CHUNKKV_STATUS.md.
"""
import glob
import json
import zlib
from collections import defaultdict

import numpy as np

NB = 50_000
ARMS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff", "chunkkv")


def load(tag):
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted(glob.glob(f"runs/nvidia/natC_{tag}_C*.jsonl") + glob.glob(f"runs/nvidia/natCK_{tag}_C*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r["arm"] not in ARMS + ("floor_pos",):
                continue
            for v, s in zip(r["variants"], r["score"]):
                acc[(r["arm"], r["C"], v["level"])][r["iid"]].append(s)
    return {k: {i: float(np.mean(x)) for i, x in d.items() if len(x) == 4} for k, d in acc.items()}


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


out, res = [], []
for tag in ("M2", "M3"):
    D = load(tag)
    out.append(f"\n## {tag}\n")
    out.append("| level | C | n | floor | " + " | ".join(f"{a} R [95% CI]" for a in ARMS) + " | chunkkv A | chunkkv deficit |")
    out.append("|---|---|---|---|" + "---|" * (len(ARMS) + 2))
    for lv in (1, 3, 5):
        for C in (256, 512):
            keys = [("floor_pos", C, lv)] + [(a, C, lv) for a in ARMS]
            if any(k not in D for k in keys):
                continue
            ids = sorted(set.intersection(*[set(D[k]) for k in keys]))
            n = len(ids)
            A = {k[0]: np.array([D[k][i] for i in ids]) for k in keys}
            rng = np.random.default_rng(zlib.crc32(f"ck|{tag}|{lv}|{C}".encode()))
            idx = rng.integers(0, n, size=(NB, n))
            bfl = A["floor_pos"][idx].mean(1)
            fl = A["floor_pos"].mean()
            assert (bfl == 0).sum() == 0
            vac = all(A[a].mean() <= 0.02 for a in ARMS)
            dag = "†" if (C == 256 or (tag == "M3" and lv == 1)) else ""
            cells = []
            for a in ARMS:
                rb = A[a][idx].mean(1) / bfl
                cells.append(f"{A[a].mean() / fl:.2f} [{ci(rb)[0]:.2f}, {ci(rb)[1]:.2f}]")
                res.append(dict(tag=tag, level=lv, C=C, arm=a, A=float(A[a].mean()), R=float(A[a].mean() / fl),
                                R_ci=ci(rb), vacuous=vac, marginal=bool(dag)))
            out.append(f"| {lv}{'†' if (tag == 'M3' and lv == 1) else ''} | {C}{'†' if C == 256 else ''}{' V' if vac else ''} | {n} | {fl:.3f} | "
                       + " | ".join(cells) + f" | {A['chunkkv'].mean():.3f} | 0-19 tok |")
open("out/natCK_report.md", "w", encoding="utf-8").write("\n".join(out))
json.dump(res, open("out/natCK_report.json", "w"), indent=1)
print("\n".join(out))
