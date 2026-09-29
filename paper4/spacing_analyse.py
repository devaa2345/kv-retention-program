"""Score PREREG_SPACING.md + Amendment S1 (1201f616...) + the n-raise scale-up, per instruction.

Pools contamination-tagged queries across SnapKV and ExpectedAttention (AdaKV blocked, S1).
Confound check (d p_g, d units_complete) computed PER ARM before pooling. Merges the frozen
Stage 3 grid file (instances 0-49, PREREG_P4_S3.md's own n=50, untouched) with the spacing
test's own extra-unconstrained file (instances >= 50) so raising n here never depends on or
retroactively alters the Stage 3 prereg's frozen sample size.
"""
from __future__ import annotations

import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "nvidia"
OUT = HERE / "out"
R_BOOT = 10000
C_TAG, C_BUDGET = 40, 512
REC = re.compile(r'^R(\d{3}) \| (.+)$')
BASES = ("snapkv", "expected_attn", "adakv_snapkv")   # AdaKV added, Amendment S2
TARGET_N = {"M2": 120, "M3": 150}   # per instruction: scale-up targets, per model


def load_s3_merged(tag, arm, n):
    """floor_pos / U-X rows for instances [0, n): Stage 3 grid file (0-49) + spacing's own
    extra-unconstrained file (>=50)."""
    rows = {}
    for fname in (f"p4_s3_{tag}.jsonl", f"p4_spacing_extra_unconstrained_{tag}.jsonl"):
        p = RUNS / fname
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("c_tag") == C_TAG and r.get("C") == C_BUDGET and r.get("arm") == arm \
                    and r["instance"] < n:
                rows[r["instance"]] = r
    return rows


def load_spacing(tag, base, n):
    rows = {}
    p = RUNS / f"p4_spacing_{base}_{tag}.jsonl"
    if p.exists():
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["instance"] < n:
                rows[r["instance"]] = r
    return rows


def merged_keepset(tag, arm, n):
    """z-like accessor merging the Stage 3 keepset npz (0-49) with the spacing extra one (>=50)."""
    stores = []
    base_npz = RUNS / f"p4_s3_keepsets_{tag}_c{C_TAG}_C{C_BUDGET}.npz"
    if base_npz.exists():
        stores.append(np.load(base_npz))
    extra_npz = RUNS / f"p4_spacing_extra_keepsets_{tag}.npz"
    if extra_npz.exists():
        stores.append(np.load(extra_npz))

    class _Merged:
        def __contains__(self, key):
            return any(key in s for s in stores)

        def __getitem__(self, key):
            for s in stores:
                if key in s:
                    return s[key]
            raise KeyError(key)
    return _Merged()


def contamination_tagged_queries(tag, unconstrained_rows, model, base_arm, n):
    from transformers import AutoTokenizer
    from p4 import common as CM
    tok = AutoTokenizer.from_pretrained(model)
    floor_rows = load_s3_merged(tag, "floor_pos", n)
    rev = floor_rows[0]["key"]["model_revision"]
    z = merged_keepset(tag, "U-" + base_arm, n)
    tagged = {}
    for i in range(n):
        if i not in floor_rows or i not in unconstrained_rows:
            continue
        b = CM.build(tag, model, rev, tok, C_TAG, "s4_%05d" % i)
        recs = {}
        for line in b["inst"].context.split("\n"):
            m = REC.match(line)
            if m:
                recs[f"R{m.group(1)}"] = m.group(2)
        fl, ur = floor_rows[i], unconstrained_rows[i]
        key = f"{i}|U-{base_arm}"
        bm = np.unpackbits(z[key], axis=1, count=b["n_ctx"]).astype(bool) if key in z else None
        for qi, v in enumerate(b["inst"].variants):
            fl_ok = fl["per_variant"][qi] == 1.0
            u_ok = ur["per_variant"][qi] == 1.0
            if not (fl_ok and not u_ok):
                continue
            if bm is not None:
                fact = next(f for f in b["facts"] if f.fact_id == v.rec_id)
                gold_tok = sorted({t for s in fact.spans for t in s})
                if not bool(bm[:, gold_tok].all(axis=1).any()):
                    continue
            u_ans = ur["gen"][qi] or ""
            u_words = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", u_ans))
            match_rec, match_score = None, 0
            for rid, fields in recs.items():
                if rid == v.rec_id:
                    continue
                rw = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", fields))
                ov = len(u_words & rw)
                if ov > match_score:
                    match_score, match_rec = ov, rid
            rec_order = list(recs.keys())
            try:
                dist = abs(rec_order.index(v.rec_id) - rec_order.index(match_rec)) if match_rec else None
            except ValueError:
                dist = None
            is_contam = bool(match_rec and dist is not None and dist <= 2 and match_score >= 6)
            tagged[(i, qi)] = is_contam
    return tagged


def boot(d, key):
    d = np.asarray(d, float)
    rng = np.random.default_rng(zlib.crc32(key.encode()) & 0xFFFFFFFF)
    m = d[rng.integers(0, len(d), size=(R_BOOT, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    L = ["PREREG_SPACING.md + Amendment S1 (1201f616...) -- n-raise scale-up.",
         "K=24, c=40, C=512. Target n: M2=120, M3=150 (per instruction).",
         "Pooled across SnapKV + ExpectedAttention (AdaKV blocked). Confound check per arm "
         "before pooling. Stage 3's frozen n=50 grid file is read, never modified.", ""]
    res = {}
    for model, tag in (("Qwen/Qwen2.5-3B-Instruct", "M2"),
                      ("meta-llama/Llama-3.2-3B-Instruct", "M3")):
        n_target = TARGET_N[tag]
        L.append(f"== {tag}  (target n={n_target})")
        arm_data = {}
        confound_notes = []
        for base in BASES:
            u_rows = load_s3_merged(tag, "U-" + base, n_target)
            sp_rows = load_spacing(tag, base, n_target)
            have = sorted(set(u_rows) & set(sp_rows))
            n_have = len(have)
            L.append(f"  [{base}] rows present: {n_have}/{n_target} (u_rows={len(u_rows)}, "
                     f"sp_rows={len(sp_rows)})")
            if n_have < 10:
                L.append(f"  [{base}] too few rows to proceed -- skipped from pool")
                continue
            matched = [i for i in have
                      if abs(sp_rows[i]["units_complete"] - u_rows[i]["units_complete"]) <= 1.0]
            unmatched = [i for i in have if i not in matched]
            relax = [sp_rows[i]["n_relaxations"] for i in have]
            fully = [sp_rows[i]["layers_fully_relaxed"] for i in have]
            L.append(f"  [{base}] completion-matching: {len(matched)}/{n_have} matched, "
                     f"{len(unmatched)} outside band")
            L.append(f"  [{base}] relaxation log: mean n_relaxations/instance {np.mean(relax):.1f} "
                     f"(range {min(relax)}-{max(relax)}); layers fully relaxed mean "
                     f"{np.mean(fully):.2f}/instance")

            dpg = float(np.mean([sp_rows[i]["p_g"] - u_rows[i]["p_g"] for i in matched])) if matched else float("nan")
            duc = float(np.mean([sp_rows[i]["units_complete"] - u_rows[i]["units_complete"] for i in matched])) if matched else float("nan")
            confounded = dpg > 0.01
            L.append(f"  [{base}] confound check: d p_g {dpg:+.4f}, d units_complete {duc:+.2f}"
                     + ("  <-- CONFOUNDED" if confounded else "  clean"))
            confound_notes.append((base, confounded, dpg, duc))

            tagged = contamination_tagged_queries(tag, u_rows, model, base, n_target)
            contam_queries = [(i, qi) for (i, qi), is_c in tagged.items() if is_c and i in matched]
            L.append(f"  [{base}] contamination-tagged queries (matched instances): {len(contam_queries)}")
            arm_data[base] = dict(sp_rows=sp_rows, contam_queries=contam_queries, matched=matched)

        pooled = []
        for base, d in arm_data.items():
            for (i, qi) in d["contam_queries"]:
                pooled.append((base, i, qi, d["sp_rows"][i]["per_variant"][qi]))
        L.append(f"  POOLED contamination-tagged queries: {len(pooled)}  (floor: 20)")
        if len(pooled) < 20:
            L.append(f"  PRECONDITION FAILS: {tag} reported as NOT SCOREABLE at this cell, at "
                     f"n={n_target}. Per instruction, not chased further.")
            res[tag] = dict(verdict="not_scoreable", n_pooled=len(pooled), n_target=n_target)
            L.append("")
            continue

        sp_correct = [1.0 if pv == 1.0 else 0.0 for (_, _, _, pv) in pooled]
        clusters = defaultdict(list)
        for k, (base, i, qi, pv) in enumerate(pooled):
            clusters[(base, i)].append(k)
        cl_keys = list(clusters)
        rng = np.random.default_rng(zlib.crc32(f"spacing|pool|{tag}|nraise".encode()) & 0xFFFFFFFF)
        arr = np.array(sp_correct)
        boots = []
        for _ in range(R_BOOT):
            sel = np.concatenate([clusters[cl_keys[j]] for j in
                                  rng.integers(0, len(cl_keys), len(cl_keys))])
            boots.append(arr[sel].mean())
        d = (float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        L.append(f"  pooled contamination-tagged accuracy: spaced {d[0]:.3f} vs unconstrained "
                 f"0.000 (all wrong by construction)  delta {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        any_confound = [b for b, c, _, _ in confound_notes if c]
        if any_confound:
            L.append(f"  CAVEAT: arm(s) {any_confound} carried a confound -- pooled result cannot "
                     "be attributed to spacing alone without excluding them.")
        verdict = "REAL" if d[1] > 0 else "NOT-IT"
        if any_confound and verdict == "REAL":
            verdict += " (CONFOUNDED -- see caveat)"
        L.append(f"  VERDICT ({tag}): {verdict}")
        res[tag] = dict(verdict=verdict, delta=d, n_pooled=len(pooled), n_target=n_target)
        L.append("")

    OUT.mkdir(exist_ok=True)
    (OUT / "spacing_results_nraise.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "spacing_results_nraise.json").write_text(json.dumps(res, indent=1, default=float),
                                                     encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
