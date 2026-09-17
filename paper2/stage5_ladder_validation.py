"""Stage 5 (package N3) — build and VALIDATE the reference ladder on kvpress. PINNED ENV ONLY.

This runs BEFORE N1. N1 is a hard gate whose result is only meaningful if the compression
underneath it is the real one; validating the ladder first is what stops a misdescribed
reference arm from entering the design, and Paper 2's entire result is a ratio of reference
arms.

Checks, all four blocking (frozen PREREG_P2.md §4.1 + v1 §7 Stage 5):

  A. tripwire ordering      null <= random <= floor_pos, in every admitted cell
  B. ceiling ordering       floor_pos <= oracle_causal <= oracle_prescient
  C. VOID refusal           oracle_prescient RAISES on C < k_gold rather than truncating gold
  D. mechanical integrity   every arm actually generates tokens; get_seq_length() does not
                            shrink for head-wise presses; prefill logits are identical at
                            every compression ratio

Check D matters because a press that silently no-ops still constructs, still returns scores,
and still produces plausible text (decision 9(c): "every press check must generate tokens").
The prefill-logits check is the sharp one: kvpress compresses in a forward hook AFTER the
attention output is computed, so the prefill logits must be bit-identical regardless of ratio.
If they move, compression is leaking into the prefill and every arm is contaminated.

Any ordering violation is investigated to root cause before anything else runs. No provisional
proceeding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, press, stats
from harness.keys import seed_key_without_seed
from harness.tasks import ledger

N_SINK, N_WINDOW = 8, 64
MAX_NEW = 32


def templated_parts(tok, context: str, query: str) -> tuple[str, str]:
    """Split the chat-templated prompt at the context/query boundary.

    The query is appended AFTER compression (agnostic protocol, §5.2), so the template must be
    cut rather than applied to the concatenation.
    """
    marker = "␟QUERY␟"
    full = tok.apply_chat_template(
        [{"role": "user", "content": context + "\n\n" + marker}],
        tokenize=False, add_generation_prompt=True)
    pre, post = full.split(marker)
    return pre, query + post


def facts_and_ctx(inst, tok, pre: str):
    """Token-index facts over the TEMPLATED prefix (that is what gets compressed)."""
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    n_ctx = len(off)
    base = pre.index(inst.context)          # offset of the raw context inside the template

    def idx(a: int, b: int):
        a, b = a + base, b + base
        return tuple(ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a)

    by_id: dict[str, list] = {}
    for sp in inst.candidates:
        by_id.setdefault(sp.rec_id, []).append(idx(sp.start, sp.end))
    facts = [ladder.FactSpans(rid, tuple(sp)) for rid, sp in by_id.items()]
    return facts, n_ctx


@torch.inference_mode()
def generate_with(model, tok, pre: str, posts: list[str], p) -> list[str]:
    """Prefill `pre` under press `p` (None = full_cache), then answer each templated query.

    **Position convention — this is load-bearing and matches kvpress's own pipeline.**
    Evicted tokens are removed but the retained keys keep the RoPE phase they were given at
    their ORIGINAL positions. So the question must continue from the *uncompressed* context
    length, not from the compressed cache length:

        position_ids = arange(context_length, context_length + len(question))

    kvpress does exactly this (`pipeline.generate_answer`), and substitutes
    `cache.get_seq_length()` only for the presses that actually re-rotate keys
    (`KeyRerotationPress`, `FinchPress`). Letting `cache_position` default to the compressed
    length instead places the question ~1900 positions before the content it must attend to,
    which silently zeroes every compressed arm while `full_cache` still looks fine — the exact
    shape of a misdescribed reference arm.
    """
    dev = model.device
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(dev)
    context_length = ids["input_ids"].shape[1]          # UNCOMPRESSED length
    ctx_mgr = p(model) if p is not None else _null_ctx()
    with ctx_mgr:
        out = model(**ids, use_cache=True)
    base_cache = out.past_key_values
    outs = []
    for post in posts:
        cache = _clone(base_cache)
        q = tok(post, add_special_tokens=False, return_tensors="pt").to(dev)["input_ids"]
        pos = torch.arange(context_length, context_length + q.shape[1],
                           device=dev).unsqueeze(0)
        cur, toks = q, []
        for _ in range(MAX_NEW):
            o = model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True)
            cache = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            t = int(nxt)
            if t == tok.eos_token_id:
                break
            toks.append(t)
            cur = nxt
            pos = pos[:, -1:] + 1
        outs.append(tok.decode(toks, skip_special_tokens=True))
    return outs


class _null_ctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def _clone(past):
    import copy
    layers = getattr(past, "layers", None)
    new = copy.copy(past)
    if layers is not None:
        new.layers = [copy.copy(l) for l in layers]
        for l in new.layers:
            l.keys, l.values = l.keys.clone(), l.values.clone()
        return new
    new.key_cache = [k.clone() for k in past.key_cache]
    new.value_cache = [v.clone() for v in past.value_cache]
    return new


def seq_len_of(past) -> int:
    layers = getattr(past, "layers", None)
    if layers is not None:
        return layers[0].keys.shape[2]
    return past.key_cache[0].shape[2]


@torch.inference_mode()
def check_D(model, tok, pre: str, budgets: list[int], n_ctx: int) -> dict:
    """Mechanical integrity: cache actually shrinks, get_seq_length agrees, prefill invariant."""
    dev = model.device
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(dev)
    ref = model(**ids, use_cache=True)
    ref_logits = ref.logits.float().clone()
    rows = []
    for C in budgets:
        B = C + N_SINK + N_WINDOW
        p = press.FloorPosPress(n_sink=N_SINK)
        p.compression_ratio = press._ratio_for(B, n_ctx)
        with p(model):
            o = model(**ids, use_cache=True)
        n_cache = seq_len_of(o.past_key_values)
        rep = o.past_key_values.get_seq_length()
        same = torch.equal(o.logits.float(), ref_logits)
        rows.append({
            "C": C, "B": B,
            "cache_len": int(n_cache), "cache_len_equals_B": bool(n_cache == B),
            "get_seq_length": int(rep),
            "get_seq_length_not_shrunk": bool(rep >= n_cache),
            "prefill_logits_identical_to_full": bool(same),
        })
    return {"n_ctx": n_ctx, "rows": rows,
            "pass": all(r["cache_len_equals_B"] and r["prefill_logits_identical_to_full"]
                        for r in rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--budgets", required=True, help="admitted C values, comma-separated")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    budgets = [int(x) for x in args.budgets.split(",")]

    res: dict = {"model": args.model, "model_revision": rev, "n": args.n,
                 "budgets": budgets, "cells": {}, "checks": {}}

    # ---- build instances once ------------------------------------------------
    insts = []
    for i in range(args.n):
        iid = f"s5_{i:05d}"
        seed = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev,
            arm="stage5", B=1, protocol="agnostic", device="nvidia", backend="cuda-12.8",
            torch_version=torch.__version__, transformers_version="5.2.0",
            kvpress_version="0.5.4", dtype="bfloat16")
        insts.append((ledger.build(seed, iid, target_tokens=2048, tokenizer=tok), seed))

    # ---- check D (cheap, one instance) ---------------------------------------
    pre0, _ = templated_parts(tok, insts[0][0].context, "")
    _, n_ctx0 = facts_and_ctx(insts[0][0], tok, pre0)
    res["checks"]["D_mechanical"] = check_D(model, tok, pre0, budgets, n_ctx0)
    d = res["checks"]["D_mechanical"]
    print(f"  [D] mechanical integrity: {'PASS' if d['pass'] else '** FAIL **'}")
    for r in d["rows"]:
        print(f"      C={r['C']:4d} cache={r['cache_len']:4d} (B={r['B']:4d}) "
              f"seq_len_ok={r['get_seq_length_not_shrunk']} "
              f"prefill_identical={r['prefill_logits_identical_to_full']}")

    # ---- check C (VOID refusal) ----------------------------------------------
    facts0, _ = facts_and_ctx(insts[0][0], tok, pre0)
    gold0 = next(f for f in facts0 if f.fact_id == insts[0][0].variants[0].rec_id)
    try:
        press.build_arm("oracle_prescient", n_ctx=n_ctx0, C=8, facts=facts0, gold=gold0)
        cres = {"pass": False, "note": "did NOT raise on a VOID cell"}
    except ValueError as e:
        cres = {"pass": "VOID" in str(e), "raised": str(e)[:140]}
    res["checks"]["C_void_refusal"] = cres
    print(f"  [C] VOID refusal: {'PASS' if cres['pass'] else '** FAIL **'}")

    # ---- checks A + B (arm means and orderings per cell) ---------------------
    for C in budgets:
        arm_scores = {a: [] for a in press.LADDER_ARMS}
        for inst, seed in insts:
            pre, _ = templated_parts(tok, inst.context, "")
            facts, n_ctx = facts_and_ctx(inst, tok, pre)
            posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]

            for arm in press.LADDER_ARMS:
                if arm == "oracle_prescient":
                    # gold differs per variant: one compressed cache per query
                    sc = []
                    for v, post in zip(inst.variants, posts):
                        gf = next(f for f in facts if f.fact_id == v.rec_id)
                        p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                               n_window=N_WINDOW, facts=facts, gold=gf,
                                               seed=seed)
                        o = generate_with(model, tok, pre, [post], p)
                        sc.append(1.0 if v.answer in o[0] else 0.0)
                    arm_scores[arm].append(sum(sc) / len(sc))
                else:
                    p, _ = press.build_arm(arm, n_ctx=n_ctx, C=C, n_sink=N_SINK,
                                           n_window=N_WINDOW, facts=facts, seed=seed)
                    outs = generate_with(model, tok, pre, posts, p)
                    arm_scores[arm].append(ledger.score_instance(outs, inst))

        m = {a: sum(v) / len(v) for a, v in arm_scores.items()}
        order = stats.check_ladder_ordering(
            null=m["null"], random=m["random"], floor_pos=m["floor_pos"],
            oracle_causal=m["oracle_causal"], oracle_prescient=m["oracle_prescient"],
            full_cache=m["full_cache"], tol=1e-9)
        # Checks A and B are the blocking criteria. `denoising` (oracle_causal > full_cache)
        # is NOT one of them: PREREG §4.1 says it "triggers the ceiling-validity audit", not
        # that it fails validation. Keeping them separate stops an audit flag from being read
        # as an ordering failure, and stops an ordering failure from hiding behind one.
        A = m["null"] <= m["random"] <= m["floor_pos"]
        B = m["floor_pos"] <= m["oracle_causal"] <= m["oracle_prescient"]
        res["cells"][f"C{C}"] = {"C": C, "means": {k: round(v, 4) for k, v in m.items()},
                                 "check_A_tripwire": bool(A), "check_B_ceiling": bool(B),
                                 "ordering_ok": order.ok, "violations": order.violations,
                                 "denoising_audit_required": order.denoising}
        print(f"  C={C:4d}  null {m['null']:.4f} random {m['random']:.4f} "
              f"floor {m['floor_pos']:.4f} causal {m['oracle_causal']:.4f} "
              f"presc {m['oracle_prescient']:.4f} full {m['full_cache']:.4f}  "
              f"{'ordering OK' if order.ok else '** VIOLATION **'}")
        for v in order.violations:
            print(f"        ! {v}")

    res["gate"] = {
        "pass": bool(
            res["checks"]["D_mechanical"]["pass"] and res["checks"]["C_void_refusal"]["pass"]
            and all(c["check_A_tripwire"] and c["check_B_ceiling"]
                    for c in res["cells"].values())),
        "A_all_cells": all(c["check_A_tripwire"] for c in res["cells"].values()),
        "B_all_cells": all(c["check_B_ceiling"] for c in res["cells"].values()),
        "denoising_cells": [k for k, c in res["cells"].items()
                            if c["denoising_audit_required"]],
    }
    print(f"\n  STAGE 5 GATE: {'PASS' if res['gate']['pass'] else '** FAIL **'}")

    tag = args.model.split("/")[-1].replace(".", "_")
    p = Path(__file__).resolve().parent / "gates" / "nvidia" / f"stage5_ladder_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {p}")
    return 0 if res["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
