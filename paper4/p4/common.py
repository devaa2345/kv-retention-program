"""Shared Stage 2 machinery: instance rebuild, arm construction, checked generation, metrics.

Instances are rebuilt with Paper 3 Stage 4's own `spec`/`build_instance` and CRC32 seed (same
key fields, same instance ids), so the pilot's X arms can be checked against Stage 4's stored
captures and generations. Every contrast in Paper 4 is nevertheless generated in ONE session: all
arms of an instance run in one pass (Paper 3's power-loss lesson).
"""
from __future__ import annotations

import statistics as st

import numpy as np
import torch

import stage4_run as S4
from harness import methods, press
from p3 import keys3, runner
from p4 import units as UN
from p4 import unitwrap as UW

N_SINK, N_WINDOW = runner.N_SINK, runner.N_WINDOW
TAGS = S4.TAGS
ENV = S4.ENV                     # includes device="nvidia" -> device is in the dedup key
PRODUCED_ON = "RTX 5070 (Machine N)"
X_METHODS = ("snapkv", "adakv_snapkv", "expected_attn", "keydiff")


def build(tag, model_name, rev, tok, c_tag, iid):
    sp = S4.spec(tag, c_tag=c_tag)
    sd = keys3.instance_seed(task=sp["task"], instance_id=iid, model=model_name,
                             model_revision=rev, n_fields=sp["n_fields"], layout=None,
                             n_records=sp["k"], **ENV)
    inst = S4.build_instance(sp, sd, iid, tok)
    pre, _ = runner.templated_parts(tok, inst.context, "")
    facts, n_ctx = runner.facts_and_ctx(inst, tok, pre)
    posts = [runner.templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]
    line_units = S4.all_unit_tokens(inst, tok, pre, sp["task"])     # Paper 3's accounting
    ui = UN.oracle_unit_index(inst, sp["task"], tok, pre, N_SINK, N_WINDOW)
    if ui.n_ctx != n_ctx:
        raise AssertionError(f"unit tokenisation n_ctx {ui.n_ctx} != {n_ctx}")
    return dict(sp=sp, sd=sd, inst=inst, pre=pre, facts=facts, n_ctx=n_ctx, posts=posts,
                mns=S4.max_new_for(tok, inst, sp["task"]), line_units=line_units, ui=ui)


def base_arm(arm):
    return arm[2:] if arm.startswith("U-") else arm


def build_press(arm, n_ctx, C, ui, cap, stats):
    B = C + N_SINK + N_WINDOW
    ratio = press._ratio_for(B, n_ctx)
    if arm == "floor_pos":
        p, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C, n_sink=N_SINK, n_window=N_WINDOW)
        p = methods.make_capturing(p, cap)
    elif arm.startswith("U-"):
        base = methods.make_floor_constrained(methods.build_method(arm[2:], ratio),
                                              n_ctx, N_SINK, N_WINDOW)
        p = UW.make_unit_aware(base, ui, C, capture=cap, stats=stats)
    else:
        base = methods.make_floor_constrained(methods.build_method(arm, ratio),
                                              n_ctx, N_SINK, N_WINDOW)
        p = methods.make_capturing(base, cap)
    p.compression_ratio = ratio
    return p


@torch.inference_mode()
def prefill(model, tok, pre, p):
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(model.device)
    with p(model):
        model(**ids, use_cache=True)


def _attn_modules(model):
    return [layer.self_attn for layer in model.model.layers]


@torch.inference_mode()
def generate_checked(model, tok, pre, posts, p, max_new, n_ctx, B, headwise):
    """Runner's generation loop with the harness invariants ASSERTED, not assumed."""
    dev = model.device
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(dev)
    context_length = ids["input_ids"].shape[1]              # UNCOMPRESSED -- load-bearing
    if context_length != n_ctx:
        raise AssertionError(f"prefill length {context_length} != n_ctx {n_ctx}")
    with p(model):
        out = model(**ids, use_cache=True)
    base = out.past_key_values
    L0 = base.layers[0].keys.shape[2]
    if headwise:
        if L0 != n_ctx or any(getattr(m, "masked_key_indices", None) is None
                              for m in _attn_modules(model)):
            raise AssertionError("head-wise press left no mask: compression not applied")
    elif L0 != B:
        raise AssertionError(f"cache length {L0} != B {B}: compression not applied")
    texts, flags = [], []
    for post, mn in zip(posts, max_new):
        cache = runner._clone(base)
        q = tok(post, add_special_tokens=False, return_tensors="pt").to(dev)["input_ids"]
        pos = torch.arange(context_length, context_length + q.shape[1], device=dev).unsqueeze(0)
        if int(pos[0, 0]) != n_ctx:
            raise AssertionError("position_ids do not continue from the uncompressed length")
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
        if not toks and stop == "cap":
            raise AssertionError("no tokens generated")
        texts.append(tok.decode(toks, skip_special_tokens=True))
        flags.append(stop)
    for m in _attn_modules(model):                          # never leak a mask to the next arm
        m.masked_key_indices = None
    return texts, flags


def keep_metrics(keeps, facts, line_units, n_ctx):
    """Paper 3 Stage 4's capture accounting, verbatim, plus the floor check."""
    gold = {t for f in facts for s in f.spans for t in s}
    floor = set(range(N_SINK)) | set(range(n_ctx - N_WINDOW, n_ctx))
    pg = [len(gold & kk) / max(1, len(gold)) for kk in keeps]
    qc = [sum(1 for f in facts if f.is_complete_in(kk)) / len(facts) for kk in keeps]
    q_any = st.fmean([max(1.0 if f.is_complete_in(kk) else 0.0 for kk in keeps) for f in facts])
    rc = st.fmean([sum(1 for s in line_units.values() if s and s <= kk) for kk in keeps])
    rt = st.fmean([sum(1 for s in line_units.values() if s & kk) for kk in keeps])
    return dict(p_g=st.fmean(pg), q_complete=st.fmean(qc), q_any=q_any, units_complete=rc,
                units_touched=rt, n_units=len(line_units), n_slots=len(keeps),
                floor_ok=all(floor <= kk for kk in keeps))


def identity_check(capX, capU, scores, arm):
    """U-X with all-singleton units vs X on the SAME real scores: equal up to exact ties.

    Compared as the multiset of kept SCORES (per slot for scorer presses, pooled per layer for
    AdaKV, whose per-head counts may trade places across tied scores).
    """
    exact = total = 0
    ok = True
    layers = sorted({li for li, _ in capX.per_head})
    for li in layers:
        S = scores[li]
        heads = sorted(h for l2, h in capX.per_head if l2 == li)
        px, pu = [], []
        for h in heads:
            kx, ku = capX.per_head[(li, h)], capU.per_head[(li, h)]
            total += 1
            exact += int(kx == ku)
            vx = np.sort(S[h, sorted(kx)])
            vu = np.sort(S[h, sorted(ku)])
            if base_arm(arm).startswith("adakv"):
                px.append(vx)
                pu.append(vu)
            elif vx.shape != vu.shape or not np.array_equal(vx, vu):
                ok = False
        if px:
            a, b = np.sort(np.concatenate(px)), np.sort(np.concatenate(pu))
            if a.shape != b.shape or not np.array_equal(a, b):
                ok = False
    return dict(identity_ok=ok, slots_exact=exact, slots=total)
