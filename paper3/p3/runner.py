"""Shared Stage 2 runner: arms, generation, budget parity, capture.

Carries the Paper 2 harness invariants forward explicitly, because every one of them was a
defect found the hard way:

  * `position_ids` continue from the **UNCOMPRESSED** context length. Evicted tokens are
    removed but retained keys keep the RoPE phase of their original positions, so the query
    must be placed after the original length. Letting it default to the compressed length puts
    the question ~1900 positions before the content and silently zeroes every compressed arm
    while `full_cache` still looks fine.
  * **Every arm check generates tokens.** Compression happens in a forward hook AFTER attention
    is computed, so prefill logits are identical at every ratio, and `get_seq_length()` does not
    shrink for head-wise presses. Any check short of generation reads a working press as inert.
  * **Method arms honour the mandatory sink and window floors.** Left unconstrained a kvpress
    press evicts the sink and spends the whole budget on its own ranking; N1 measured the sink
    alone as worth ~0.27 accuracy.
  * **Realised budget parity is asserted, per instance, per arm**, from the captured keep-sets
    rather than from the requested ratio.
  * **max_new_tokens scales with the answer**, and an EOS-vs-cap flag is recorded. Paper 2's
    pinned `max_new=32` was a model-dependent format confound; here the answer is up to ~37
    tokens at c = 40, so a fixed cap would truncate the long-`c` cells and be read as a
    retention failure.
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

P2 = Path(__file__).resolve().parents[2] / "paper2"
if str(P2) not in sys.path:
    sys.path.insert(0, str(P2))

from harness import ladder, methods, press  # noqa: E402

N_SINK, N_WINDOW = 8, 64
MAX_NEW_SLACK = 16
LADDER_ARMS = ("full_cache", "null", "random", "floor_pos",
               "oracle_causal", "oracle_prescient")
SYNTHETIC_PREFIX = "contig_matched_"


def templated_parts(tok, context: str, query: str):
    marker = "␟QUERY␟"
    full = tok.apply_chat_template([{"role": "user", "content": context + "\n\n" + marker}],
                                   tokenize=False, add_generation_prompt=True)
    pre, post = full.split(marker)
    return pre, query + post


def facts_and_ctx(inst, tok, pre: str):
    """Token-index facts over the TEMPLATED prefix -- that is what gets compressed.

    Multi-span facts (SPLIT-LEDGER's /A and /B halves) group into ONE FactSpans with two
    spans, so `is_complete_in` requires both, which is what makes the scattered control a
    control rather than a relabelling.
    """
    enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    n_ctx = len(off)
    base = pre.index(inst.context)

    def idx(a: int, b: int):
        a, b = a + base, b + base
        return tuple(ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a)

    by_id: dict[str, list] = {}
    for sp in inst.candidates:
        by_id.setdefault(sp.rec_id, []).append(idx(sp.start, sp.end))
    facts = [ladder.FactSpans(rid, tuple(sp)) for rid, sp in by_id.items()]
    return facts, n_ctx


@contextmanager
def _null_ctx():
    yield


def _clone(cache):
    import copy
    layers = getattr(cache, "layers", None)
    new = copy.copy(cache)
    if layers is not None:
        new.layers = [copy.copy(l) for l in layers]
        for l in new.layers:
            l.keys = l.keys.clone()
            l.values = l.values.clone()
        return new
    new.key_cache = [k.clone() for k in cache.key_cache]
    new.value_cache = [v.clone() for v in cache.value_cache]
    return new


@torch.inference_mode()
def generate_with(model, tok, pre: str, posts: list[str], p, max_new: list[int]):
    """Prefill `pre` under press `p` (None = full_cache), then answer each templated query.

    Returns (texts, flags) where each flag is "eos" or "cap".
    """
    dev = model.device
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(dev)
    context_length = ids["input_ids"].shape[1]          # UNCOMPRESSED -- load-bearing
    ctx = p(model) if p is not None else _null_ctx()
    with ctx:
        out = model(**ids, use_cache=True)
    base = out.past_key_values
    texts, flags = [], []
    for post, mn in zip(posts, max_new):
        cache = _clone(base)
        q = tok(post, add_special_tokens=False, return_tensors="pt").to(dev)["input_ids"]
        pos = torch.arange(context_length, context_length + q.shape[1], device=dev).unsqueeze(0)
        cur, toks, stop = q, [], "cap"
        for _ in range(mn):
            o = model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True)
            cache = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            t = int(nxt)
            if t == tok.eos_token_id:
                stop = "eos"
                break
            toks.append(t)
            cur = nxt
            pos = pos[:, -1:] + 1
        texts.append(tok.decode(toks, skip_special_tokens=True))
        flags.append(stop)
    return texts, flags


# --------------------------------------------------------------------- capture / parity

@torch.inference_mode()
def capture_keepsets(model, tok, pre: str, arm: str, C: int, n_ctx: int):
    """Prefill only, no generation: returns the per-(layer, KV-head) keep-sets of a method."""
    cap = methods.Capture()
    ratio = press._ratio_for(C + N_SINK + N_WINDOW, n_ctx)
    base = methods.make_floor_constrained(methods.build_method(arm, ratio),
                                          n_ctx, N_SINK, N_WINDOW)
    p = methods.make_capturing(base, cap)
    p.compression_ratio = ratio
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with p(model):
        model(**ids, use_cache=True)
    return cap


def assert_budget_parity(cap, C: int, n_ctx: int, arm: str, tol: int = 1):
    """Realised retained count per (layer, KV-head) must equal B.

    AdaKV is the exception BY DESIGN: it redistributes a layer's budget across that layer's
    heads, so the invariant it satisfies is a per-LAYER total of B * n_heads. Checking the
    per-head count for AdaKV would fail a correctly-behaving press, which is the trap Paper 2's
    methods.py documents.
    """
    B = C + N_SINK + N_WINDOW
    by_layer: dict[int, list[int]] = {}
    for (li, h), n in cap.n_kept.items():
        by_layer.setdefault(li, []).append(n)
    if not by_layer:
        raise AssertionError(f"{arm}: no keep-sets captured")
    bad = []
    for li, counts in by_layer.items():
        if arm.startswith("adakv"):
            if abs(sum(counts) - B * len(counts)) > tol * len(counts):
                bad.append((li, sum(counts), B * len(counts)))
        else:
            for n in counts:
                if abs(n - B) > tol:
                    bad.append((li, n, B))
                    break
    if bad:
        raise AssertionError(
            f"{arm} C={C}: realised budget != B on {len(bad)} layer(s); first {bad[:3]} "
            f"(expected {B}{' per layer total x heads' if arm.startswith('adakv') else ''})")
    return B


def ladder_kept(arm: str, n_ctx: int, C: int, facts, seed: int, gold=None):
    """The analytic keep-set of a ladder arm, for completion accounting without a capture."""
    if arm == "floor_pos":
        return set(ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts).kept)
    if arm == "null":
        return set(ladder.null_arm(n_ctx, C, N_SINK, N_WINDOW).kept)
    if arm == "random":
        return set(ladder.random_arm(n_ctx, C, seed, N_SINK, N_WINDOW, facts).kept)
    if arm == "oracle_causal":
        return set(ladder.oracle_causal(n_ctx, C, facts, n_sink=N_SINK,
                                        n_window=N_WINDOW).kept)
    if arm == "oracle_prescient":
        return set(ladder.oracle_prescient(n_ctx, C, gold, N_SINK, N_WINDOW, facts).kept)
    if arm == "full_cache":
        return set(range(n_ctx))
    raise ValueError(arm)


# --------------------------------------------------------------------- synthetic arm (2.2)

class ContigMatchedPress(press._ExactSetPress):
    """Probe 2.2 -- same gold-token count as a method, arranged as COMPLETE facts.

    Paper 2's strongest single observation was that at C=512 the methods hold MORE gold tokens
    than `floor_pos` and still score far below it. That is suggestive but confounded: the arms
    differ in token count AND in coherence.

    This arm isolates coherence. Given the method's realised gold-token count `g`, it completes
    as many queried facts as `g` allows (ascending payable cost), then fills the rest of the
    budget from `floor_pos`'s own preference order. So it holds approximately the same number
    of gold tokens as the method, arranged coherently rather than fragmented, and -- unlike
    Paper 2's `floor_matched` diagnostic, which grew a block freely and broke parity -- it
    spends **exactly B**, so it remains comparable to every other arm.
    """


def build_contig_matched(n_ctx: int, C: int, facts, gold_budget: int):
    """Keep-set holding <= `gold_budget` gold tokens, as complete facts, inside exactly B."""
    floor, region = ladder._floors(n_ctx, N_SINK, N_WINDOW)
    region_set = set(region)
    payable = []
    for f in facts:
        toks = {t for s in f.spans for t in s if t in region_set}
        payable.append((len(toks), f.fact_id, toks))
    payable.sort(key=lambda r: (r[0], r[1]))

    chosen: set[int] = set()
    n_complete = 0
    for cost, _fid, toks in payable:
        if cost == 0:
            continue
        if len(chosen | toks) <= min(gold_budget, C):
            chosen |= toks
            n_complete += 1
    kept = ladder._pad_from_floor(floor | set(chosen), region, C - len(chosen))
    p = ContigMatchedPress().set_keep(kept, n_ctx)
    return p, dict(n_facts_complete=n_complete, gold_tokens_held=len(chosen),
                   gold_budget=gold_budget, n_kept=len(kept))


__all__ = ["templated_parts", "facts_and_ctx", "generate_with", "capture_keepsets",
           "assert_budget_parity", "ladder_kept", "build_contig_matched",
           "ContigMatchedPress", "N_SINK", "N_WINDOW", "LADDER_ARMS", "MAX_NEW_SLACK"]
