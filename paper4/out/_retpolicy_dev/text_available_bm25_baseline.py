"""Item 2 part B: text-available BM25 reference baseline.

Unlike the frozen system's lexical pointer (restricted to sentences WHOLLY RETAINED in the
compressed cache), this baseline ranks ALL sentences of the original document with the same
frozen BM25 function and tie rule, inserts the top one via the same focus_query, and answers
from a SINGLE floor prefill at Bfinal=B0-e (no scout pass needed, since the pointer here never
touches the model at all -- it's pure text). This is a reference for a text-available setting
(the document is assumed fully searchable outside the retained budget), not a competing
deployable claim against the frozen tight-KV system, which must select only from what its own
compression retained.

Frozen modules reused unmodified: _realtext_5070_data, _realtext_5070_prefill (build_press,
spans_for, check_orders), _realtext_5070_mask (decode_literal_masked), _realtext_5070_generate
(fingerprint, score), _realtext_focus_generate (focus_query), _realtext_pointer_screen (lexical).
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "out")
from p3 import runner as P3R
from _realtext_5070_data import MODELS, RATIO, SINK, WINDOW, format_query, split_sentences
from _realtext_5070_generate import score
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
    n_fallback = 0
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as fout:
        for case in cases:
            sentences = [s[2] for s in split_sentences(case["context"])]
            for qi, q in enumerate(case["queries"][:2]):
                pre, post = P3R.templated_parts(tok, case["context"], format_query(q["question"]))
                spans, labels, nctx = spans_for(tok, pre, case["context"])
                b0 = round(RATIO * nctx)

                row = {"eligible": list(range(len(sentences))),
                      "sentences": {str(i): s for i, s in enumerate(sentences)},
                      "question": q["question"]}
                scores = lexical(row)
                selected = max(range(len(sentences)), key=lambda si: (scores.get(si, 0.0), -si))

                focused = focus_query(q["question"], sentences[selected])
                pre2, post2 = P3R.templated_parts(tok, case["context"], focused)
                assert pre2 == pre
                base_len = tok(post, add_special_tokens=False,
                              return_tensors="pt")["input_ids"].shape[1]
                focus_len = tok(post2, add_special_tokens=False,
                                return_tensors="pt")["input_ids"].shape[1]
                e = focus_len - base_len
                fallback = (b0 - e) <= SINK + WINDOW
                if fallback:
                    n_fallback += 1
                    post2 = post
                    e = 0
                bfinal = b0 - e

                orders = {}
                p = build_press("floor_pos", nctx, bfinal, set(), orders)
                ids = tok(pre, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
                with torch.inference_mode(), p(model):
                    prefix = model(input_ids=ids, use_cache=True)
                check_orders(orders, nctx, bfinal)
                union = set()
                for order in orders.values():
                    union.update(map(int, order))
                selected_held = spans[selected] <= union

                q_ids = tok(post2, add_special_tokens=False,
                           return_tensors="pt")["input_ids"].to("cuda")
                pos = torch.arange(nctx, nctx + q_ids.shape[1], device="cuda").unsqueeze(0)
                with torch.inference_mode():
                    out = model(input_ids=q_ids, past_key_values=P3R._clone(prefix.past_key_values),
                               position_ids=pos, use_cache=True)
                max_new = len(tok(q["gold"], add_special_tokens=False)["input_ids"]) + 16
                raw, stop = decode_literal_masked(
                    model, tok, P3R._clone(out.past_key_values), pos[:, -1:],
                    out.logits[:, -1, :], max_new, sentences[selected])
                sc = score(raw, q["gold"], case["context"])

                row_out = dict(model="M2", arm="floor_pos_text_available_bm25",
                              instance=case["index"], qi=qi, kind=q["kind"],
                              n_ctx=nctx, budget=bfinal, floor_budget=b0,
                              focus_extra_tokens=e, focus_fallback=fallback,
                              selected_sentence=selected, selected_held=selected_held,
                              gold=q["gold"], gold_sentence=q["gold_sentence"],
                              raw=raw, stop=stop, max_new_tokens=max_new, **sc)
                fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                fout.flush()
                del prefix, out
                torch.cuda.empty_cache()
            print(f"instance {case['index']} elapsed={(time.time()-t0)/60:.2f} min", flush=True)
    print(f"DONE: {len(cases)} instances, {n_fallback} fallbacks, "
          f"elapsed={(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
