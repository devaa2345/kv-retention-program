"""Item 3: prefill-only retention-policy screen. No answer generation anywhere in this script.

Four retention policies, all at the SAME floor token budget (B0), on the same frozen 160-instance
confirmation manifest, both models:
  - floor_pos       (frozen, unmodified)
  - snapkv          (frozen, unmodified)
  - u_snapkv        NEW: whole-sentence selection by aggregated SnapKV score (sum of per-token
                    score, where per-token score = mean over heads, then mean over layers, from
                    SnapKV's own .score() -- captured via a new hook, not a frozen-file edit).
                    Greedily takes WHOLE sentences from the compressible region in descending
                    score order until the next one would exceed the region budget C, then fills
                    any remainder with the most-recent unselected region tokens (same filler
                    convention as the LEDGER-C session's U-floor).
  - farthest_first  NEW: relevance-free, whole-sentence diversity policy. Distance metric =
                    Jaccard distance over sentence word-sets (lower-cased, casefold, split on
                    whitespace/punctuation via the frozen `terms()`-style tokenization is NOT
                    reused here since that has a question-independent stopword list tuned for
                    BM25 -- this uses a plain lowercase word-set, no stopword removal, since
                    diversity should not be relevance-tuned). Start: the eligible region sentence
                    appearing EARLIEST in document order. Then repeatedly add the remaining
                    sentence that maximizes its MINIMUM Jaccard distance to every already-selected
                    sentence; ties broken by earliest document position. Same whole-sentence +
                    most-recent-filler budget rule as u_snapkv.

Pointer: the SAME frozen lexical BM25 pointer (`_realtext_pointer_screen.lexical`) is applied
UNIFORMLY to all four policies' own retained (eligible = wholly-retained) sentence sets, per the
forward-plan's own instruction to reuse "the frozen lexical pointer on real text" as the common
comparison instrument. This means the entire screen is prefill-only -- no query forward pass,
no attention capture, no generation.
"""
import argparse, json, sys, time, types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "out")
from p3 import runner as P3R
from harness import methods, press
from _realtext_5070_data import MODELS, RATIO, SINK, WINDOW, format_query, split_sentences
from _realtext_5070_prefill import build_press, check_orders, spans_for
from _realtext_pointer_screen import lexical

MANIFEST = Path("out/realtext_tight_confirm_160.jsonl")


def attach_score_capture(p, scores_out):
    """Like the frozen attach_exact_capture, but ALSO saves the raw per-layer score tensor
    (mean over heads) so it can be aggregated to sentence level for u_snapkv. Does not alter
    what the press retains -- compress() still does the identical topk gather."""
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        s = self.score(module, hidden_states, keys, values, attentions, kwargs)
        scores_out[int(module.layer_idx)] = s.mean(dim=1)[0].detach().float().cpu().numpy()
        n_kept = int(keys.shape[2] * (1 - self.compression_ratio))
        idx = s.topk(n_kept, dim=-1).indices
        gather = idx.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        return keys.gather(2, gather).contiguous(), values.gather(2, gather).contiguous()
    p.compress = types.MethodType(compress, p)
    return p


def jaccard_distance(a_words, b_words):
    if not a_words and not b_words:
        return 0.0
    inter = len(a_words & b_words)
    union = len(a_words | b_words)
    return 1.0 - (inter / union if union else 0.0)


def whole_sentence_policy(order_by_score, sentence_word_positions, region, sink, window, n_ctx, C):
    """order_by_score: sentence indices in the order they should be greedily taken.
    sentence_word_positions: {si: set(token positions)} restricted to region-only tokens.
    Returns the final retained set (sink + window + whole sentences + filler), size = C+SINK+WINDOW."""
    floor_set = set(range(sink)) | set(range(n_ctx - window, n_ctx))
    chosen = set()
    remaining = C
    for si in order_by_score:
        toks = sentence_word_positions.get(si, set())
        if toks and len(toks) <= remaining:
            chosen |= toks
            remaining -= len(toks)
    if remaining > 0:
        avail = sorted(region - chosen, reverse=True)  # most-recent-first
        chosen |= set(avail[:remaining])
    return floor_set | chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("M2", "M3"), required=True)
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = sorted((json.loads(s) for s in MANIFEST.read_text(encoding="utf-8").splitlines()),
                  key=lambda c: c["index"])[:args.n]
    name = MODELS[0] if args.model == "M2" else MODELS[1]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to("cuda").eval()

    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as fout:
        for case in cases:
            sentences = [s[2] for s in split_sentences(case["context"])]
            for qi, q in enumerate(case["queries"][:2]):
                pre, post = P3R.templated_parts(tok, case["context"], format_query(q["question"]))
                spans, labels, nctx = spans_for(tok, pre, case["context"])
                b0 = round(RATIO * nctx)
                C = b0 - SINK - WINDOW
                region = set(range(SINK, nctx - WINDOW))
                ids = tok(pre, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
                gold_si = q["gold_sentence"]

                retained_sets = {}

                for arm in ("floor_pos", "snapkv"):
                    orders = {}
                    p = build_press(arm, nctx, b0, set(), orders)
                    with torch.inference_mode(), p(model):
                        model(input_ids=ids, use_cache=True)
                    check_orders(orders, nctx, b0)
                    union = set()
                    for order in orders.values():
                        union.update(map(int, order))
                    retained_sets[arm] = union

                # u_snapkv: aggregated SnapKV score -> whole-sentence greedy
                ratio = press._ratio_for(b0, nctx)
                base_press = methods.make_floor_constrained(
                    methods.build_method("snapkv", ratio), nctx, SINK, WINDOW)
                scores_out = {}
                scored_press = attach_score_capture(base_press, scores_out)
                scored_press.compression_ratio = ratio
                with torch.inference_mode(), scored_press(model):
                    model(input_ids=ids, use_cache=True)
                mean_score = np.mean(list(scores_out.values()), axis=0)  # (n_ctx,)
                sent_word_pos = {si: (spans[si] & region) for si in range(len(sentences))}
                sent_score = {si: float(mean_score[sorted(toks)].sum()) if toks else -1e18
                             for si, toks in sent_word_pos.items()}
                order_u = sorted(sent_score, key=lambda si: -sent_score[si])
                retained_sets["u_snapkv"] = whole_sentence_policy(
                    order_u, sent_word_pos, region, SINK, WINDOW, nctx, C)

                # farthest_first: Jaccard-distance whole-sentence diversity, relevance-free
                region_sentences = [si for si in range(len(sentences)) if sent_word_pos.get(si)]
                word_sets = {si: set(sentences[si].casefold().split()) for si in region_sentences}
                if region_sentences:
                    start = min(region_sentences, key=lambda si: min(sent_word_pos[si]))
                    selected_ff = [start]
                    remaining_pool = [si for si in region_sentences if si != start]
                    while remaining_pool:
                        best_si, best_dist = None, -1.0
                        for si in remaining_pool:
                            d = min(jaccard_distance(word_sets[si], word_sets[s2])
                                   for s2 in selected_ff)
                            if d > best_dist or (d == best_dist and
                                                 (best_si is None or
                                                  min(sent_word_pos[si]) < min(sent_word_pos[best_si]))):
                                best_si, best_dist = si, d
                        selected_ff.append(best_si)
                        remaining_pool.remove(best_si)
                else:
                    selected_ff = []
                retained_sets["farthest_first"] = whole_sentence_policy(
                    selected_ff, sent_word_pos, region, SINK, WINDOW, nctx, C)

                row = dict(model=args.model, instance=case["index"], qi=qi, n_ctx=nctx, budget=b0,
                          gold_sentence=gold_si)
                for arm, retained in retained_sets.items():
                    held = bool(spans[gold_si]) and spans[gold_si] <= retained
                    eligible = [si for si in range(len(sentences))
                               if spans[si] and spans[si] <= retained]
                    if eligible:
                        lex_row = {"eligible": eligible,
                                  "sentences": {str(si): sentences[si] for si in eligible},
                                  "question": q["question"]}
                        scores = lexical(lex_row)
                        selected = max(eligible, key=lambda si: (scores[si], -si))
                        hit = (selected == gold_si)
                    else:
                        selected, hit = None, False
                    row[f"{arm}_size"] = len(retained)
                    row[f"{arm}_held"] = held
                    row[f"{arm}_selected"] = selected
                    row[f"{arm}_hit"] = hit
                fout.write(json.dumps(row) + "\n")
                fout.flush()
            print(f"instance {case['index']} elapsed={(time.time()-t0)/60:.2f} min", flush=True)
    print(f"DONE {args.model}: {len(cases)} instances, elapsed={(time.time()-t0)/60:.2f} min",
          flush=True)


if __name__ == "__main__":
    main()
