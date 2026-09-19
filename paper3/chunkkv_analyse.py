"""ChunkKV vs the four admitted methods on the LEDGER plane at C = 512 (no GPU).

Reads runs/nvidia/stage4_plane_<tag>.jsonl (floor_pos, snapkv, adakv_snapkv, expected_attn,
keydiff, oracle_causal; deduped by key_digest) and runs/nvidia/chunkkv_plane_<tag>.jsonl.
Same instances (s4_00000..), same scorers. Paired instance bootstrap, 20,000 draws (the draw
count used for the LEDGER Table 1 intervals), numerator and denominator on the SAME resample.
R = A_method / A_floor.   G_m = (A_m - A_floor) / (A_causal - A_floor); refused when
A_causal - A_floor < 0.15 (Paper 2 rule). Floor < 0.05 -> X (excluded, frozen rule 9.1).
"""
import json
import zlib

import numpy as np

NB = 20_000
ARMS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff", "chunkkv")
C_LABELS = ("c=1", "c=8", "c=19", "c=40")
C = 512


def load(path, want_arms):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["C"] != C or r["arm"] not in want_arms:
            continue
        d[(r["arm"], r["label"], r["key"]["instance_id"])] = (r["score"], r["c"], r.get("deficit"))
    return d


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


out = []
res = []
for tag in ("M2", "M3"):
    base = load(f"runs/nvidia/stage4_plane_{tag}.jsonl",
                ("floor_pos", "snapkv", "adakv_snapkv", "expected_attn", "keydiff", "oracle_causal"))
    ck = load(f"runs/nvidia/chunkkv_plane_{tag}.jsonl", ("chunkkv",))
    base.update(ck)
    out.append(f"\n## {tag}, C = 512 (LEDGER, n per cell in table)\n")
    out.append("| c | n | floor | causal | " + " | ".join(f"{a} R [95% CI]" for a in ARMS) + " | chunkkv G_m [CI] | chunkkv deficit (tokens) |")
    out.append("|---|---|---|---|" + "---|" * (len(ARMS) + 2))
    for lab in C_LABELS:
        ids = sorted({k[2] for k in base if k[1] == lab and k[0] == "chunkkv"})
        ids = [i for i in ids if all((a, lab, i) in base for a in ARMS + ("floor_pos", "oracle_causal"))]
        n = len(ids)
        if n == 0:
            continue
        A = {a: np.array([base[(a, lab, i)][0] for i in ids]) for a in ARMS + ("floor_pos", "oracle_causal")}
        cval = base[("chunkkv", lab, ids[0])][1]
        rng = np.random.default_rng(zlib.crc32(f"chunkkv|{tag}|{lab}".encode()))
        idx = rng.integers(0, n, size=(NB, n))
        bfl = A["floor_pos"][idx].mean(1)
        bca = A["oracle_causal"][idx].mean(1)
        fl, ca = A["floor_pos"].mean(), A["oracle_causal"].mean()
        cells = []
        status = "X" if fl < 0.05 else ""
        for a in ARMS:
            ba = A[a][idx].mean(1)
            if fl == 0 or status == "X":
                cells.append(f"{A[a].mean() / fl:.2f} (X)" if fl > 0 else "n/a")
                res.append(dict(tag=tag, c=lab, arm=a, R=None, status=status))
                continue
            assert (bfl == 0).sum() == 0
            rb = ba / bfl
            cells.append(f"{A[a].mean() / fl:.2f} [{ci(rb)[0]:.2f}, {ci(rb)[1]:.2f}]")
            res.append(dict(tag=tag, c=lab, cval=cval, arm=a, A=float(A[a].mean()),
                            R=float(A[a].mean() / fl), R_ci=ci(rb), status=status, n=n))
        head = ca - fl
        if head < 0.15 or status == "X":
            g = "n/a (headroom %.3f)" % head
        else:
            gb = (A["chunkkv"][idx].mean(1) - bfl) / (bca - bfl)
            g = f"{(A['chunkkv'].mean() - fl) / head:.2f} [{ci(gb)[0]:.2f}, {ci(gb)[1]:.2f}]"
        dfs = [base[("chunkkv", lab, i)][2] for i in ids if base[("chunkkv", lab, i)][2] is not None]
        out.append(f"| {lab} ({cval:.1f}) {status} | {n} | {fl:.3f} | {ca:.3f} | " + " | ".join(cells)
                   + f" | {g} | {np.mean(dfs):.1f} (max {max(dfs)}) |")
open("out/chunkkv_report.md", "w", encoding="utf-8").write("\n".join(out))
json.dump(res, open("out/chunkkv_report.json", "w"), indent=1)
print("\n".join(out))
