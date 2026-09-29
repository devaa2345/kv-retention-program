"""LEDGER-C reinterpretation, tightened: proximity insertion at EQUAL KV (no extra out-of-budget
tokens), using the verified single-pass eviction method from single_prefill_equivalence.py
(drop e tokens from the already-compressed B0 cache; floor's own rule = oldest non-sink/window
position, NOT protecting the selected/inserted record, matching the corrected, verified
reconstruction from that check).

Two arms, both at the SAME B0 budget as the frozen floor_pos/floor+mask arms:
  (A) equal-KV proximity, unmasked:  drop e tokens, insert record text into the query, decode
                                     unmasked.
  (B) equal-KV proximity, masked:    same eviction + insertion, decode with the frozen LEDGER
                                     schema-free literal-copy mask (reused from
                                     _copyconstrained_text.py / _step5_primary_run.py, unmodified).

Reuses the ALREADY-STORED max_rid from the frozen LEDGER-C confirmatory run -- no attention
recomputation needed for the pointer itself.
"""
import argparse, gc, json, sys, types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "out")
from harness import methods, press
from p3 import runner as P3R
from p4 import common as CM
from _copyconstrained_text import build_format_set, SchemaFreeSpanAutomaton

C_TAG, C_BUDGET = 40, 512
MODELS = {"M2": "Qwen/Qwen2.5-3B-Instruct", "M3": "meta-llama/Llama-3.2-3B-Instruct"}
REVS = {"M2": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "M3": "0cb88a4f764b7a12671c53f0838cd831a0843b95"}


def record_reference_text(b, rid):
    import re
    m = re.search(rf'(?m)^({rid}) \| (.+)$', b["inst"].context)
    return f"{m.group(1)} | {m.group(2)}" if m else None


def build_scored_press(n_ctx, cap, stats):
    """EXACT reuse of _step5_primary_run.py's build_scored_floor_press pattern: make_capturing
    first (gives cap.per_head, the retained SET per (layer, kv_head)), THEN bind a .score() hook
    on the INSTANCE (types.MethodType, applied AFTER make_capturing so it survives -- make_capturing
    builds a new instance internally, so binding before this point gets silently dropped, per that
    file's own documented lesson). This is the pattern proven safe across every LEDGER run tonight.

    An earlier version of this function overrode .compress() directly instead (mirroring
    single_prefill_equivalence.py's build_press_with_scores) and leaked ~5-6GB (this
    environment's own inflated reporting units) per instance, reproducibly crashing on instance 2
    every time and unresponsive to gc.collect()/empty_cache() -- consistent with a live reference
    (most likely kvpress's own hook/compress lifecycle being disrupted by replacing .compress()
    wholesale) rather than uncollected garbage. Patching .score() instead leaves kvpress's own
    compress() gather logic completely untouched; only the score computation is observed."""
    NS, NW = CM.N_SINK, CM.N_WINDOW
    p, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C_BUDGET, n_sink=NS, n_window=NW)
    p = methods.make_capturing(p, cap)
    p.compression_ratio = press._ratio_for(C_BUDGET + NS + NW, n_ctx)
    orig_score = type(p).score

    def hooked(self, module, hidden_states, keys, values, attentions, kwargs):
        s = orig_score(self, module, hidden_states, keys, values, attentions, kwargs)
        stats.setdefault("scores", {})[int(module.layer_idx)] = \
            s[0].detach().to(torch.float64).cpu().numpy()
        return s
    p.score = types.MethodType(hooked, p)
    return p


def choose_drop_floor(retained_positions, e, sink, window, n_ctx):
    """floor's verified rule: oldest non-sink/window position, no protection of the inserted
    record -- matches the corrected single_prefill_equivalence.py exactly. retained_positions:
    the SET of original token positions this head retained (from cap.per_head), order-agnostic
    since the rule sorts by position value directly."""
    protected = set(range(sink)) | set(range(n_ctx - window, n_ctx))
    droppable = sorted(p for p in retained_positions if p not in protected)
    chosen = droppable[:e]
    if len(chosen) != e:
        raise AssertionError(f"only {len(chosen)} droppable, need e={e}")
    return set(chosen)


def get_layers(cache):
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return [(l.keys, l.values) for l in layers], "layers"
    return list(zip(cache.key_cache, cache.value_cache)), "kv_cache_lists"


def set_layer(cache, li, keys, values, fmt):
    if fmt == "layers":
        cache.layers[li].keys = keys
        cache.layers[li].values = values
    else:
        cache.key_cache[li] = keys
        cache.value_cache[li] = values


@torch.inference_mode()
def decode_unmasked(model, tok, cache, pos, first_logits, mn):
    nxt = first_logits.argmax(dim=-1, keepdim=True)
    toks, stop = [], "cap"
    t = int(nxt)
    if t == tok.eos_token_id:
        return "", "eos"
    toks.append(t)
    cur, pos = nxt, pos[:, -1:] + 1
    for _ in range(mn - 1):
        o = model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True)
        cache = o.past_key_values
        nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        t = int(nxt)
        if t == tok.eos_token_id:
            stop = "eos"
            break
        toks.append(t)
        cur, pos = nxt, pos[:, -1:] + 1
    return tok.decode(toks, skip_special_tokens=True), stop


def decode_masked(model, tok, cache, pos, first_logits, mn, auto, format_ids):
    dev = first_logits.device
    logits = first_logits.clone()
    allowed_format = format_ids - {tok.eos_token_id}
    if auto.eos_legal():
        allowed_format = allowed_format | {tok.eos_token_id}
    legal = set(auto.legal_from_pool()) | allowed_format
    mask = torch.full_like(logits, float("-inf"))
    idx = torch.tensor(sorted(legal), device=dev, dtype=torch.long)
    mask[0, idx] = logits[0, idx]
    nxt = mask.argmax(dim=-1, keepdim=True)
    toks, stop = [], "cap"
    t = int(nxt)
    if t == tok.eos_token_id:
        return "", "eos"
    toks.append(t)
    auto.feed(t, tok.decode([t]))
    cur, pos = nxt, pos[:, -1:] + 1
    for _ in range(mn - 1):
        o = model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True)
        cache = o.past_key_values
        logits = o.logits[:, -1, :].clone()
        allowed_format = format_ids - {tok.eos_token_id}
        if auto.eos_legal():
            allowed_format = allowed_format | {tok.eos_token_id}
        legal = set(auto.legal_from_pool()) | allowed_format
        mask = torch.full_like(logits, float("-inf"))
        idx = torch.tensor(sorted(legal), device=dev, dtype=torch.long)
        mask[0, idx] = logits[0, idx]
        nxt = mask.argmax(dim=-1, keepdim=True)
        t = int(nxt)
        if t == tok.eos_token_id:
            stop = "eos"
            break
        toks.append(t)
        auto.feed(t, tok.decode([t]))
        cur, pos = nxt, pos[:, -1:] + 1
    return tok.decode(toks, skip_special_tokens=True), stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("M2", "M3"), required=True)
    ap.add_argument("--start", type=int, default=120)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--exclude", default="", help="comma list of instances to skip")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    exclude = set(int(x) for x in args.exclude.split(",") if x)

    stored_path = f"out/_step5_primary_{args.model}.jsonl"
    stored = {}
    for line in open(stored_path, encoding="utf-8"):
        r = json.loads(line)
        if r["arm"] == "floor_mask_ordered":
            stored[(r["instance"], r["qi"])] = r["max_rid"]

    name = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to("cuda").eval()
    FORMAT_IDS = build_format_set(tok)
    NS, NW = CM.N_SINK, CM.N_WINDOW

    n_fail = 0
    with args.out.open("w", encoding="utf-8") as fout:
        for i in range(args.start, args.start + args.n):
            if i in exclude:
                continue
            b = CM.build(args.model, name, REVS[args.model], tok, C_TAG, "s4_%05d" % i)
            dev = model.device
            ids = tok(b["pre"], add_special_tokens=False, return_tensors="pt").to(dev)
            cl = ids["input_ids"].shape[1]

            cap, stats = methods.Capture(), {}
            p = build_scored_press(b["n_ctx"], cap, stats)
            with p(model):
                scout = model(input_ids=ids["input_ids"], use_cache=True)
            n_layers = len({li for li, h in cap.per_head})
            n_kv_heads = len({h for li, h in cap.per_head if li == 0})
            scout_cache_clone = P3R._clone(scout.past_key_values)
            # per (layer, kv_head), reconstruct the ACTUAL compressed-cache gather order
            # (descending score -- topk's own convention) once per instance, reused for every
            # query below. Exactly _step5_primary_run.py's mass_by_rid_from_attn convention.
            scout_order = {}
            for (li, h), retained in cap.per_head.items():
                S = stats["scores"][li]
                scout_order[(li, h)] = sorted(retained, key=lambda idx: -S[h, idx])

            for qi, v in enumerate(b["inst"].variants):
                mn = b["mns"][qi]
                gold = v.answer
                try:
                    max_rid = stored.get((i, qi))
                    if max_rid is None:
                        raise AssertionError("no stored max_rid")
                    rec_text = record_reference_text(b, max_rid)
                    proximity_query = f"Record {max_rid}: {rec_text}\n\n{v.query}"
                    _, post_plain = CM.runner.templated_parts(tok, b["inst"].context, v.query)
                    _, post_prox = CM.runner.templated_parts(tok, b["inst"].context, proximity_query)
                    base_len = tok(post_plain, add_special_tokens=False,
                                  return_tensors="pt")["input_ids"].shape[1]
                    prox_len = tok(post_prox, add_special_tokens=False,
                                  return_tensors="pt")["input_ids"].shape[1]
                    e = prox_len - base_len
                    if e <= 0 or e >= C_BUDGET:
                        raise AssertionError(f"bad e={e}")

                    cache = P3R._clone(scout_cache_clone)
                    layer_tensors, fmt = get_layers(cache)
                    for li in range(n_layers):
                        keys, values = layer_tensors[li]
                        new_k, new_v = [], []
                        for h in range(n_kv_heads):
                            order = scout_order[(li, h)]
                            drop = choose_drop_floor(cap.per_head[(li, h)], e, NS, NW, b["n_ctx"])
                            keep_js = [j for j, pos_ in enumerate(order) if int(pos_) not in drop]
                            assert len(keep_js) == len(order) - e
                            idx_t = torch.tensor(keep_js, dtype=torch.long, device=keys.device)
                            new_k.append(keys[:, h:h+1, idx_t, :])
                            new_v.append(values[:, h:h+1, idx_t, :])
                        set_layer(cache, li, torch.cat(new_k, dim=1), torch.cat(new_v, dim=1), fmt)

                    q = tok(post_prox, add_special_tokens=False, return_tensors="pt").to(dev)["input_ids"]
                    pos0 = torch.arange(cl, cl + q.shape[1], device=dev).unsqueeze(0)
                    with torch.inference_mode():
                        o0 = model(input_ids=q, past_key_values=cache, position_ids=pos0, use_cache=True)
                    pos_after = torch.arange(cl + q.shape[1] - 1, cl + q.shape[1], device=dev).unsqueeze(0)

                    cache1 = P3R._clone(o0.past_key_values)
                    text_a, _ = decode_unmasked(model, tok, cache1, pos_after, o0.logits[:, -1, :], mn)
                    score_a = CM.S4.SCORE_ONE[b["sp"]["task"]](text_a, gold)

                    field_texts = [max_rid] + [f.strip() for f in rec_text.split("|")[1:]]
                    auto = SchemaFreeSpanAutomaton(tok, rec_text)
                    cache2 = P3R._clone(o0.past_key_values)
                    text_b, _ = decode_masked(model, tok, cache2, pos_after, o0.logits[:, -1, :],
                                              mn, auto, FORMAT_IDS)
                    score_b = CM.S4.SCORE_ONE[b["sp"]["task"]](text_b, gold)

                    fout.write(json.dumps(dict(model=args.model, instance=i, qi=qi, rec_id=v.rec_id,
                              max_rid=max_rid, e=e, gold=gold, a_gen=text_a, a_score=score_a,
                              b_gen=text_b, b_score=score_b)) + "\n")
                    fout.flush()
                except Exception as ex:
                    n_fail += 1
                    print(f"  FAIL inst{i} q{qi}: {type(ex).__name__}: {ex}", flush=True)
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
            del scout, scout_cache_clone
            torch.cuda.empty_cache()
            if (i - args.start + 1) % 10 == 0:
                print(f"    inst {i-args.start+1}/{args.n}  fails={n_fail}", flush=True)
    print(f"DONE {args.model}: fails={n_fail}", flush=True)


if __name__ == "__main__":
    main()
