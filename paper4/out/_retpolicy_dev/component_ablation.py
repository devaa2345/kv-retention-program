"""Item 2 (follow-up): component ablation, M2, full 160-instance confirmation manifest, BM25
over all sentences (frozen tie rule) picking one sentence per query, held fixed across all 4 arms
below so the ablation isolates ONLY what changes:

  (a) sentence_only:   question + retrieved sentence AS the entire context (no compression, no
                       original document at all) -- literal mask. Tests whether the compressed
                       cache contributes anything beyond having the sentence text available.
  (b) floor_unmasked:  floor cache at B0-e + retrieved sentence appended via the same focus
                       insertion -- NO mask (plain greedy decode). Isolates the mask's own effect
                       given the same cache/insertion as (c).
  (c) floor_masked:    floor cache at B0-e + retrieved sentence + literal mask. Already run and
                       reported (154/320) as out/_retpolicy_dev/bm25_M2_full.jsonl -- reused
                       here unmodified, not recomputed.
  (d) full_cache_masked: entire original context, uncompressed (no retention budget at all) +
                       retrieved sentence + literal mask. Upper bound on what the cache can give
                       given a fixed pointer/mask, since nothing is ever evicted.

All four reuse the same BM25 selection per query (computed once here), the same focus_query
insertion mechanism, and the same literal_v1 mask where used. Frozen modules imported unmodified:
_realtext_5070_data, _realtext_5070_generate (decode_unmasked, score), _realtext_5070_mask
(decode_literal_masked), _realtext_5070_prefill (build_press, check_orders, spans_for),
_realtext_focus_generate (focus_query), _realtext_pointer_screen (lexical).
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "out")
from p3 import runner as P3R
from _realtext_5070_data import MODELS, RATIO, SINK, WINDOW, format_query, split_sentences
from _realtext_5070_generate import decode_unmasked, score
from _realtext_5070_mask import decode_literal_masked
from _realtext_5070_prefill import build_press, check_orders, spans_for
from _realtext_focus_generate import focus_query
from _realtext_pointer_screen import lexical

MANIFEST = Path("out/realtext_tight_confirm_160.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = sorted((json.loads(s) for s in MANIFEST.read_text(encoding="utf-8").splitlines()),
                  key=lambda c: c["index"])[:args.n]
    name = MODELS[0]
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

                row_lex = {"eligible": list(range(len(sentences))),
                          "sentences": {str(i): s for i, s in enumerate(sentences)},
                          "question": q["question"]}
                scores = lexical(row_lex)
                selected = max(range(len(sentences)), key=lambda si: (scores.get(si, 0.0), -si))
                sel_text = sentences[selected]

                focused = focus_query(q["question"], sel_text)
                pre2, post2 = P3R.templated_parts(tok, case["context"], focused)
                assert pre2 == pre
                base_len = tok(post, add_special_tokens=False,
                              return_tensors="pt")["input_ids"].shape[1]
                focus_len = tok(post2, add_special_tokens=False,
                                return_tensors="pt")["input_ids"].shape[1]
                e = focus_len - base_len
                fallback = (b0 - e) <= SINK + WINDOW
                post2_use = post if fallback else post2
                e_use = 0 if fallback else e
                bfinal = b0 - e_use
                max_new = len(tok(q["gold"], add_special_tokens=False)["input_ids"]) + 16

                out_row = dict(model="M2", instance=case["index"], qi=qi, kind=q["kind"],
                              n_ctx=nctx, selected_sentence=selected, gold=q["gold"],
                              gold_sentence=q["gold_sentence"], focus_extra_tokens=e_use,
                              focus_fallback=fallback)

                # (a) sentence_only: the sentence itself is the ENTIRE context, no compression
                s_pre, s_post = P3R.templated_parts(tok, sel_text, format_query(q["question"]))
                s_ids = tok(s_pre, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
                with torch.inference_mode():
                    s_prefill = model(input_ids=s_ids, use_cache=True)
                s_qids = tok(s_post, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
                s_pos = torch.arange(s_ids.shape[1], s_ids.shape[1]+s_qids.shape[1],
                                     device="cuda").unsqueeze(0)
                with torch.inference_mode():
                    s_out = model(input_ids=s_qids, past_key_values=P3R._clone(s_prefill.past_key_values),
                                 position_ids=s_pos, use_cache=True)
                raw_a, stop_a = decode_literal_masked(
                    model, tok, P3R._clone(s_out.past_key_values), s_pos[:, -1:],
                    s_out.logits[:, -1, :], max_new, sel_text)
                out_row.update({f"a_{k}": v for k, v in score(raw_a, q["gold"], case["context"]).items()})
                out_row["a_raw"] = raw_a
                del s_prefill, s_out
                torch.cuda.empty_cache()

                # (b) floor_unmasked: same cache/insertion as (c), no mask
                orders_b = {}
                p_b = build_press("floor_pos", nctx, bfinal, set(), orders_b)
                ids = tok(pre, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
                with torch.inference_mode(), p_b(model):
                    prefix_b = model(input_ids=ids, use_cache=True)
                check_orders(orders_b, nctx, bfinal)
                q_ids_b = tok(post2_use, add_special_tokens=False,
                             return_tensors="pt")["input_ids"].to("cuda")
                pos_b = torch.arange(nctx, nctx+q_ids_b.shape[1], device="cuda").unsqueeze(0)
                with torch.inference_mode():
                    out_b = model(input_ids=q_ids_b, past_key_values=P3R._clone(prefix_b.past_key_values),
                                 position_ids=pos_b, use_cache=True)
                raw_b, stop_b = decode_unmasked(model, tok, P3R._clone(out_b.past_key_values),
                                                pos_b[:, -1:], out_b.logits[:, -1, :], max_new)
                out_row.update({f"b_{k}": v for k, v in score(raw_b, q["gold"], case["context"]).items()})
                out_row["b_raw"] = raw_b
                del prefix_b, out_b
                torch.cuda.empty_cache()

                # (d) full_cache_masked: entire original context, no compression, + mask
                with torch.inference_mode():
                    prefix_d = model(input_ids=ids, use_cache=True)
                q_ids_d = tok(post2_use, add_special_tokens=False,
                             return_tensors="pt")["input_ids"].to("cuda")
                pos_d = torch.arange(nctx, nctx+q_ids_d.shape[1], device="cuda").unsqueeze(0)
                with torch.inference_mode():
                    out_d = model(input_ids=q_ids_d, past_key_values=P3R._clone(prefix_d.past_key_values),
                                 position_ids=pos_d, use_cache=True)
                raw_d, stop_d = decode_literal_masked(
                    model, tok, P3R._clone(out_d.past_key_values), pos_d[:, -1:],
                    out_d.logits[:, -1, :], max_new, sel_text)
                out_row.update({f"d_{k}": v for k, v in score(raw_d, q["gold"], case["context"]).items()})
                out_row["d_raw"] = raw_d
                del prefix_d, out_d
                torch.cuda.empty_cache()

                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                fout.flush()
            print(f"instance {case['index']} elapsed={(time.time()-t0)/60:.2f} min", flush=True)
    print(f"DONE: {len(cases)} instances, elapsed={(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
