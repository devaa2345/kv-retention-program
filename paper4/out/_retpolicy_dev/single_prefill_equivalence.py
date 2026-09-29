"""Item 1: single-prefill equivalence check for the frozen tight-KV two-pass system.

Tests whether the SECOND real prefill can be replaced by directly dropping e tokens from the
SCOUT's already-compressed cache (never sink, window, or the selected sentence), then appending
the focus query and decoding with the same frozen literal mask -- versus the frozen two-pass
method's stored result for the same (instance, qi, arm).

Every frozen module is imported and used UNMODIFIED: `_realtext_5070_data`, `_realtext_5070_mask`,
`_realtext_5070_generate` (fingerprint/score), `_realtext_5070_prefill` (build_press/spans_for/
check_orders), `_realtext_focus_generate` (focus_query), `_realtext_pointer_generate` (only for
import parity, not called -- this test reuses the FROZEN RUN's own stored pointer picks, not a
fresh pointer call). No frozen file is written to. Output goes only under out/_retpolicy_dev/.
"""
import argparse, hashlib, json, sys, time, types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "out")
from harness import methods, press
from p3 import runner as P3R
from _realtext_5070_data import MODELS, RATIO, SINK, WINDOW, format_query, split_sentences
from _realtext_5070_generate import fingerprint, score
from _realtext_5070_mask import decode_literal_masked
from _realtext_5070_prefill import build_press, check_orders, spans_for
from _realtext_focus_generate import focus_query

MANIFEST = Path("out/realtext_tight_confirm_160.jsonl")
RESULTS = {"M2": Path("out/realtext_tight_confirm_M2_results.jsonl"),
           "M3": Path("out/realtext_tight_confirm_M3_results.jsonl")}
N_INSTANCES = 10
ARMS = ("floor_pos", "snapkv")


def build_press_with_scores(arm, n_ctx, budget, orders, scores_out):
    """Same press construction as the frozen build_press (_realtext_5070_prefill.py), but ALSO
    captures the raw per-(layer,head) score row over the FULL n_ctx, so the "lowest-scoring
    retained token" claim can be verified directly against actual score values rather than
    assumed from kvpress's topk ordering."""
    C = budget - SINK - WINDOW
    ratio = press._ratio_for(budget, n_ctx)
    if arm == "floor_pos":
        p, _ = press.build_arm("floor_pos", n_ctx=n_ctx, C=C, n_sink=SINK, n_window=WINDOW)
    elif arm == "snapkv":
        p = methods.make_floor_constrained(methods.build_method("snapkv", ratio), n_ctx, SINK, WINDOW)
    else:
        raise ValueError(arm)
    p.compression_ratio = ratio

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        s = self.score(module, hidden_states, keys, values, attentions, kwargs)
        n_kept = int(keys.shape[2] * (1 - self.compression_ratio))
        idx = s.topk(n_kept, dim=-1).indices
        li = int(module.layer_idx)
        for h in range(idx.shape[1]):
            orders[(li, h)] = idx[0, h].detach().cpu().numpy().astype(np.int32)
            scores_out[(li, h)] = s[0, h].detach().float().cpu().numpy()
        gather = idx.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, gather).contiguous(), values.gather(2, gather).contiguous()
    p.compress = types.MethodType(compress, p)
    return p


def choose_drop(order, scores_row, e, arm, sink, window, n_ctx):
    """order: np.array of original positions in TOPK order. scores_row: full-n_ctx score array
    for this (layer,head), used ONLY for snapkv (floor's rule is purely positional). Per the
    corrected instruction: protects sink/window ONLY -- the selected sentence is NOT protected,
    exactly replicating the frozen system's own (unprotective) second-pass floor/snapkv
    recompression."""
    protected = set(range(sink)) | set(range(n_ctx - window, n_ctx))
    droppable_idx = [j for j, p in enumerate(order) if int(p) not in protected]
    if arm == "snapkv":
        # lowest-scoring among retained, verified against ACTUAL captured scores, not assumed
        # topk order
        droppable_idx_sorted = sorted(droppable_idx, key=lambda j: scores_row[order[j]])
        chosen_js = droppable_idx_sorted[:e]
    else:
        # floor: oldest non-sink/window = smallest original position among droppable
        chosen_js = sorted(droppable_idx, key=lambda j: order[j])[:e]
    if len(chosen_js) != e:
        raise AssertionError(f"only {len(chosen_js)} droppable tokens available, need e={e}")
    return {int(order[j]) for j in chosen_js}, chosen_js, droppable_idx


def apply_drop_to_cache(cache, orders_li, drop_by_head):
    """orders_li: {h: np.array(order)} for one layer. drop_by_head: {h: set(positions)}.
    Returns (new_keys, new_values, new_order_li) for that layer -- gathers the COMPLEMENT of the
    dropped positions per head, since different heads may drop different original positions."""
    pass  # implemented inline in main() per-layer, kept here as a doc anchor


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("M2", "M3"), required=True)
    ap.add_argument("--n", type=int, default=N_INSTANCES)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = [json.loads(s) for s in MANIFEST.read_text(encoding="utf-8").splitlines()]
    cases = sorted(cases, key=lambda c: c["index"])[:args.n]
    results = {}
    for line in RESULTS[args.model].read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        results[(r["instance"], r["qi"], r["arm"])] = r

    device = torch.cuda.get_device_name(0)
    name = MODELS[0] if args.model == "M2" else MODELS[1]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to("cuda").eval()

    n_compared = 0
    n_keephash_match = 0
    n_answer_match = 0
    n_score_match = 0
    n_skipped_fallback = 0
    rows = []
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as fout:
        for case in cases:
            sentences = [s[2] for s in split_sentences(case["context"])]
            prepared = []
            for q in case["queries"][:2]:
                pre, post = P3R.templated_parts(tok, case["context"], format_query(q["question"]))
                spans, labels, nctx = spans_for(tok, pre, case["context"])
                prepared.append((pre, post, spans, labels, nctx, q))
            pre = prepared[0][0]
            nctx = prepared[0][4]
            b0 = round(RATIO * nctx)
            ids = tok(pre, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")

            for arm in ARMS:
                scout_orders = {}
                scout_scores = {}
                scout_press = build_press_with_scores(arm, nctx, b0, scout_orders, scout_scores)
                with torch.inference_mode(), scout_press(model):
                    scout = model(input_ids=ids, use_cache=True)
                check_orders(scout_orders, nctx, b0)
                # clone BEFORE any A1 query is appended, per instruction
                scout_cache_clone = P3R._clone(scout.past_key_values)
                n_layers = len({li for li, h in scout_orders})
                n_kv_heads = len({h for li, h in scout_orders if li == 0})
                layer_tensors, fmt = get_layers(scout_cache_clone)

                for qi, (_, post, spans, labels, _, q) in enumerate(prepared):
                    stored = results.get((case["index"], qi, arm + "_tight_focus_mask"))
                    if stored is None:
                        print(f"  MISSING stored result inst{case['index']} q{qi} {arm}", flush=True)
                        continue
                    if stored.get("focus_fallback"):
                        n_skipped_fallback += 1
                        print(f"  SKIP (fallback in frozen run) inst{case['index']} q{qi} {arm}",
                              flush=True)
                        continue
                    e = stored["focus_extra_tokens"]
                    selected = stored["selected_sentence"]
                    max_new = stored["max_new_tokens"]

                    # per-layer, per-head: choose e tokens to drop, apply to a FRESH clone of the
                    # scout cache (independent per (qi,arm) so drops for different queries never
                    # interfere with each other's cache). Selected sentence NOT protected, per
                    # corrected instruction -- matches the frozen system's own unprotective
                    # second-pass recompression exactly.
                    cache = P3R._clone(scout_cache_clone)
                    layer_tensors_c, fmt_c = get_layers(cache)
                    new_orders = {}
                    boundary_checks = []
                    for li in range(n_layers):
                        keys, values = layer_tensors_c[li]
                        drop_by_head = {}
                        for h in range(n_kv_heads):
                            order = scout_orders[(li, h)]
                            scores_row = scout_scores[(li, h)]
                            drop_set, chosen_js, droppable_idx = choose_drop(
                                order, scores_row, e, arm, SINK, WINDOW, nctx)
                            drop_by_head[h] = drop_set
                            if arm == "snapkv" and li == 0 and h == 0:
                                dropped_scores = sorted(scores_row[order[j]] for j in chosen_js)
                                kept_droppable_js = [j for j in droppable_idx if j not in chosen_js]
                                kept_scores = sorted(scores_row[order[j]] for j in kept_droppable_js)
                                boundary_checks.append(dict(
                                    max_dropped=float(max(dropped_scores)) if dropped_scores else None,
                                    min_kept=float(min(kept_scores)) if kept_scores else None,
                                    n_dropped=len(dropped_scores), n_kept_droppable=len(kept_scores)))
                        new_k_heads, new_v_heads = [], []
                        for h in range(n_kv_heads):
                            order = scout_orders[(li, h)]
                            drop = drop_by_head[h]
                            keep_js = [j for j, p in enumerate(order) if int(p) not in drop]
                            assert len(keep_js) == len(order) - e
                            idx_t = torch.tensor(keep_js, dtype=torch.long, device=keys.device)
                            new_k_heads.append(keys[:, h:h+1, idx_t, :])
                            new_v_heads.append(values[:, h:h+1, idx_t, :])
                            new_orders[(li, h)] = [int(order[j]) for j in keep_js]
                        set_layer(cache, li, torch.cat(new_k_heads, dim=1),
                                 torch.cat(new_v_heads, dim=1), fmt_c)

                    if boundary_checks:
                        bc = boundary_checks[0]
                        ok = (bc["max_dropped"] is None or bc["min_kept"] is None or
                             bc["max_dropped"] <= bc["min_kept"])
                        print(f"    [snapkv score boundary, layer0 head0] max_dropped_score="
                              f"{bc['max_dropped']} min_kept_score={bc['min_kept']} "
                              f"invariant_holds={ok}", flush=True)

                    single_pass_hash = fingerprint(new_orders)

                    focused = focus_query(q["question"], sentences[selected])
                    pre2, post2 = P3R.templated_parts(tok, case["context"], focused)
                    assert pre2 == pre
                    focus_ids = tok(post2, add_special_tokens=False,
                                    return_tensors="pt")["input_ids"].to("cuda")
                    assert focus_ids.shape[1] - (tok(post, add_special_tokens=False,
                                                     return_tensors="pt")["input_ids"].shape[1]) == e
                    fpos = torch.arange(nctx, nctx + focus_ids.shape[1], device="cuda").unsqueeze(0)
                    with torch.inference_mode():
                        focus = model(input_ids=focus_ids, past_key_values=cache,
                                     position_ids=fpos, use_cache=True)
                    raw, stop = decode_literal_masked(
                        model, tok, P3R._clone(focus.past_key_values), fpos[:, -1:],
                        focus.logits[:, -1, :], max_new, sentences[selected])
                    sc = score(raw, q["gold"], case["context"])

                    keephash_match = (single_pass_hash == stored["keep_hash"])

                    symdiff_size = None
                    if arm == "snapkv" and not keephash_match:
                        # diagnostic only: run the REAL frozen second-pass prefill at Bfinal to
                        # recover its actual keep positions (the frozen system only saved a hash,
                        # not the raw set), then measure the symmetric difference directly
                        bfinal = b0 - e
                        real_orders = {}
                        real_press = build_press(arm, nctx, bfinal, set(), real_orders)
                        with torch.inference_mode(), real_press(model):
                            real_second = model(input_ids=ids, use_cache=True)
                        check_orders(real_orders, nctx, bfinal)
                        real_flat = set()
                        for order in real_orders.values():
                            real_flat.update(map(int, order))
                        symdiff_size = len(single_pass_flat_preview := set().union(
                            *new_orders.values()) ^ real_flat)
                        del real_second
                        torch.cuda.empty_cache()
                    answer_match = (raw == stored["raw"])
                    score_match = (sc["score"] == stored["score"])
                    n_compared += 1
                    n_keephash_match += keephash_match
                    n_answer_match += answer_match
                    n_score_match += score_match
                    single_pass_flat = set()
                    for v in new_orders.values():
                        single_pass_flat.update(v)
                    row = dict(model=args.model, instance=case["index"], qi=qi, arm=arm, e=e,
                              selected_sentence=selected, single_pass_hash=single_pass_hash,
                              stored_hash=stored["keep_hash"], keephash_match=keephash_match,
                              single_pass_raw=raw, stored_raw=stored["raw"],
                              answer_match=answer_match, single_pass_score=sc["score"],
                              stored_score=stored["score"], score_match=score_match,
                              bf16_flip_only=(answer_match is False and keephash_match),
                              single_pass_union_size=len(single_pass_flat),
                              symdiff_vs_real_second_pass=symdiff_size)
                    rows.append(row)
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                    print(f"  inst{case['index']} q{qi} {arm}: keephash_match={keephash_match} "
                          f"answer_match={answer_match} score_match={score_match}", flush=True)
                del scout, scout_cache_clone
                torch.cuda.empty_cache()
            print(f"instance {case['index']} done, elapsed={(time.time()-t0)/60:.2f} min", flush=True)

    print(f"\nDONE: n_compared={n_compared} skipped_fallback={n_skipped_fallback}", flush=True)
    print(f"keep-hash match rate: {n_keephash_match}/{n_compared}", flush=True)
    print(f"answer match rate: {n_answer_match}/{n_compared}", flush=True)
    print(f"score match rate: {n_score_match}/{n_compared}", flush=True)
    diffs_with_matching_hash = [r for r in rows if r["keephash_match"] and not r["answer_match"]]
    diffs_with_mismatched_hash = [r for r in rows if not r["keephash_match"] and not r["answer_match"]]
    print(f"answer diffs WITH matching keep-hash (bf16-near-tie candidates): "
          f"{len(diffs_with_matching_hash)}", flush=True)
    print(f"answer diffs WITH mismatched keep-hash (NOT bf16 near-ties -- real discrepancy): "
          f"{len(diffs_with_mismatched_hash)}", flush=True)


if __name__ == "__main__":
    main()
