"""ChunkKV admission gates G2 and G3 -- the SAME procedure as paper2/n4a_admission_gates.py.

The four admitted methods (SnapKV, AdaKV-SnapKV, ExpectedAttention, KeyDiff) went through
paper2/n4a_admission_gates.py under PREREG_P2_v2 section 7: n=24 LEDGER instances, budgets C in {32, 128, 512},
one model at a time, pinned WSL environment. This script reuses that file's instance construction, random-arm
reference, statistics, thresholds and decision rules unchanged (imported or copied verbatim). The only
ChunkKV-specific parts are (i) how the press is built and (ii) how its retained set is recorded, because
ChunkKV keeps whole chunks and its retained set is not a top-k of token scores:

  * press:   ChunkKVBudget over the floor-constrained SnapKV scorer, chunk length 20 (the Paper 3 grid setting).
  * G2 permutation: the SnapKV token-score vector INSIDE ChunkKV is permuted (make_permuted) before the floors
    are pinned, exactly as n4a permutes a method's own score before the floors. ChunkKV has no score of its own
    other than the chunk mean of that vector.
  * capture: the retained positions are recomputed from the same scores with the same chunk logic and recorded
    per (layer, head). ChunkKV selects chunks per layer, so all heads of a layer hold the same set.

Run in WSL:  PYTHONPATH=paper3:paper2 /opt/p2venv/bin/python chunkkv_admission.py --model ... --n 24 --budgets 32,128,512
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvpress import SnapKVPress

from harness import ladder, methods, press, stats
from harness.keys import seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts
from n4a_admission_gates import region_of, indicator, N_SINK, N_WINDOW, PREREG
from p3.chunkkv import ChunkKVBudget, CHUNK, expected_kept


class ChunkKVBudgetCap(ChunkKVBudget):
    """ChunkKVBudget that also records the retained positions per (layer, head)."""

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        ratio = self.press.compression_ratio
        if ratio == 0:
            return keys, values
        kv_len = keys.shape[2]
        L = self.chunk_length
        B = int(kv_len * (1 - ratio))
        rem, ncomp = kv_len % L, kv_len // L
        ntot = ncomp + (1 if rem else 0)
        n_kept = (B - rem) // L + 1 if rem else B // L
        n_kept = max(1, min(ntot, n_kept))
        adj = 1.0 - (n_kept + 0.5) / ntot
        self.press.compression_ratio = adj
        try:
            # recompute exactly the selection kvpress.ChunkKVPress.compress will make
            gs = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
            if ncomp > 0:
                ms = gs[..., : ncomp * L]
                mc = ms.sum(dim=1).view(-1, ncomp, L).mean(dim=-1)
                if rem > 0:
                    rc = gs[..., -rem:].sum(dim=1).mean(dim=-1, keepdim=True)
                    cs = torch.cat([mc, rc], dim=-1)
                else:
                    cs = mc
                nk = max(1, int((ncomp + (rem > 0)) * (1 - adj)))
                top = cs.topk(nk, dim=-1).indices[0].tolist()
                kept = set()
                for ci in top:
                    if ci < ncomp:
                        kept.update(range(ci * L, ci * L + L))
                    else:
                        kept.update(range(ncomp * L, kv_len))
                li = methods._layer_idx(module)
                cap = self._capture
                cap.seq_len = kv_len
                for h in range(keys.shape[1]):
                    cap.per_head[(li, h)] = set(kept)
                    cap.n_kept[(li, h)] = len(kept)
            return super(ChunkKVBudget, self).compress(module, hidden_states, keys, values, attentions, kwargs)
        finally:
            self.press.compression_ratio = ratio


def run_chunkkv(model, tok, pre, posts, inst, C, n_ctx, seed, permuted=False):
    B = C + N_SINK + N_WINDOW
    ratio = press._ratio_for(B, n_ctx)
    inner = SnapKVPress(compression_ratio=ratio)
    if permuted:
        inner = methods.make_permuted(inner, seed)
    inner = methods.make_floor_constrained(inner, n_ctx, N_SINK, N_WINDOW)   # floors AFTER the permutation
    p = ChunkKVBudgetCap(press=inner, chunk_length=CHUNK)
    p.compression_ratio = ratio
    cap = methods.Capture()
    p._capture = cap
    outs = generate_with(model, tok, pre, posts, p)
    return ledger.score_instance(outs, inst), cap, n_ctx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--budgets", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    budgets = [int(x) for x in args.budgets.split(",")]

    insts = []
    for i in range(args.n):
        iid = f"g_{i:05d}"
        sd = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev,
            arm="n4a", B=1, protocol="agnostic", device="nvidia", backend="cuda-12.8",
            torch_version=torch.__version__, transformers_version="5.2.0",
            kvpress_version="0.5.4", dtype="bfloat16")
        insts.append((ledger.build(sd, iid, target_tokens=2048, tokenizer=tok), sd))

    res = {"model": args.model, "model_revision": rev, "n": args.n, "prereg_sha256": PREREG,
           "procedure": "paper2/n4a_admission_gates.py (unchanged decision rules)", "chunk_length": CHUNK,
           "budgets": budgets, "methods": {}}

    rnd_ref = {}
    for C in budgets:
        sc = []
        for inst, sd in insts:
            pre, _ = templated_parts(tok, inst.context, "")
            facts, n_ctx = facts_and_ctx(inst, tok, pre)
            posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
            p, _ = press.build_arm("random", n_ctx=n_ctx, C=C, n_sink=N_SINK, n_window=N_WINDOW, facts=facts, seed=sd)
            sc.append(ledger.score_instance(generate_with(model, tok, pre, posts, p), inst))
        rnd_ref[C] = sc
        print(f"  [ref] random arm @C={C}: {statistics.fmean(sc):.4f}", flush=True)

    name = "chunkkv"
    m: dict = {"budgets": {}, "errors": []}
    try:
        for C in budgets:
            orig, perm, ovl, rhos, deficits = [], [], [], [], []
            for inst, sd in insts:
                pre, _ = templated_parts(tok, inst.context, "")
                facts, n_ctx = facts_and_ctx(inst, tok, pre)
                posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
                region = sorted(region_of(n_ctx))
                s_o, cap_o, _ = run_chunkkv(model, tok, pre, posts, inst, C, n_ctx, sd)
                s_p, cap_p, _ = run_chunkkv(model, tok, pre, posts, inst, C, n_ctx, sd, permuted=True)
                orig.append(s_o)
                perm.append(s_p)
                B = C + N_SINK + N_WINDOW
                deficits.append(B - next(iter(cap_o.n_kept.values())))
                for key in cap_o.heads():
                    if key not in cap_p.per_head:
                        continue
                    a = indicator(cap_o.per_head[key], region)
                    b = indicator(cap_p.per_head[key], region)
                    if a.std() == 0 or b.std() == 0:
                        continue
                    rhos.append(float(spearmanr(a, b).statistic))
                oc = ladder.oracle_causal(n_ctx, C, facts, n_sink=N_SINK, n_window=N_WINDOW)
                ok = set(oc.kept) & set(region)
                if ok:
                    for key in cap_o.heads():
                        ovl.append(len(cap_o.per_head[key] & ok) / len(ok))
            g2 = stats.paired_contrast(orig, perm, name=f"{name}-perm@C{C}", seed=C, n_boot=20000)
            m["budgets"][f"C{C}"] = {
                "C": C, "acc_original": round(statistics.fmean(orig), 4), "acc_permuted": round(statistics.fmean(perm), 4),
                "acc_random_arm": round(statistics.fmean(rnd_ref[C]), 4),
                "orig_minus_perm": {"mean": round(g2.mean_diff, 4), "ci": [round(g2.ci_low, 4), round(g2.ci_high, 4)]},
                "rho_retained_orig_vs_perm": round(statistics.fmean(rhos), 4) if rhos else None,
                "oracle_overlap_per_head": round(statistics.fmean(ovl), 4) if ovl else None,
                "tokens_under_budget_mean": round(statistics.fmean(deficits), 2),
            }
            r = m["budgets"][f"C{C}"]
            print(f"  C={C:4d} orig {r['acc_original']:.4f} perm {r['acc_permuted']:.4f} rand {r['acc_random_arm']:.4f}  "
                  f"o-p {r['orig_minus_perm']['mean']:+.4f} {r['orig_minus_perm']['ci']}  rho {r['rho_retained_orig_vs_perm']}  "
                  f"ovl {r['oracle_overlap_per_head']}  under-budget {r['tokens_under_budget_mean']}", flush=True)

        # ---- gate decisions: copied verbatim from paper2/n4a_admission_gates.py -------------
        bs = list(m["budgets"].values())
        g2_ablation = all(b["acc_permuted"] <= b["acc_random_arm"] + 0.02 for b in bs)
        g2_rho = all((b["rho_retained_orig_vs_perm"] is None or abs(b["rho_retained_orig_vs_perm"]) < 0.10) for b in bs)
        g2_pass = any(b["orig_minus_perm"]["ci"][0] > 0 for b in bs)
        g2_budgets_informative = [b["C"] for b in bs if b["orig_minus_perm"]["ci"][0] > 0]
        ovs = [b["oracle_overlap_per_head"] for b in bs]
        g3_pos = all(o is not None and o > 0 for o in ovs)
        g3_mono = all(ovs[i] <= ovs[i + 1] + 1e-9 for i in range(len(ovs) - 1)) if g3_pos else False
        m["G2"] = {"pass": bool(g2_pass and g2_rho and g2_ablation), "beats_own_permutation_somewhere": bool(g2_pass),
                   "informative_at_budgets": g2_budgets_informative, "permuted_falls_to_random_arm": bool(g2_ablation),
                   "rho_near_zero": bool(g2_rho)}
        m["G3"] = {"pass": bool(g3_pos and g3_mono), "overlap_positive": bool(g3_pos), "monotone_in_C": bool(g3_mono), "overlaps": ovs}
        m["admitted_on_G2_G3"] = bool(m["G2"]["pass"] and m["G3"]["pass"])
    except Exception as e:
        import traceback
        traceback.print_exc()
        m["errors"].append(f"{type(e).__name__}: {e}"[:300])
        m["G2"] = {"pass": False}
        m["G3"] = {"pass": False}
        m["admitted_on_G2_G3"] = False
    res["methods"][name] = m
    print(f"  -> G2 {m['G2']['pass']}  G3 {m['G3']['pass']}", flush=True)
    tag = args.model.split("/")[-1].replace(".", "_")
    p = Path(__file__).resolve().parent / "out" / f"chunkkv_admission_{tag}.json"
    p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
