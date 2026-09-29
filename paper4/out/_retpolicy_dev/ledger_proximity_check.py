"""LEDGER-C reinterpretation, item 2: does the copy-mask gain reduce to a proximity effect?

Reuses the ALREADY-STORED max_rid (top-attended record) from the frozen LEDGER-C confirmatory
run (out/_step5_primary_M2.jsonl) -- no attention recomputation. For each of the same 80
instances / 320 queries, inserts that record's full text next to the question (same spirit as
the real-text focus insertion) and decodes UNMASKED from a plain floor_pos cache at the same
budget. Compared against the frozen floor_pos (0.1938) and floor+mask (0.2812) results.

If unmasked-proximity-insertion recovers most of the mask's gain, the LEDGER-C mask's own
contribution is mostly redundant with just having the record nearby -- a proximity effect,
paralleling the real-text ablation's finding.
"""
import json, re, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "out")
from p4 import common as CM
from harness import press

C_TAG, C_BUDGET, START, N = 40, 512, 120, 80
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REV = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def record_reference_text(b, rid):
    m = re.search(rf'(?m)^({rid}) \| (.+)$', b["inst"].context)
    return f"{m.group(1)} | {m.group(2)}" if m else None


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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()
    n_run = args.n
    stored = {}
    for line in open("out/_step5_primary_M2.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["arm"] == "floor_mask_ordered":
            stored[(r["instance"], r["qi"])] = r["max_rid"]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to("cuda").eval()
    out_path = Path("out/_retpolicy_dev/ledger_proximity_M2.jsonl")
    n_fail = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for i in range(START, START + n_run):
            b = CM.build("M2", MODEL, REV, tok, C_TAG, "s4_%05d" % i)
            NS, NW = CM.N_SINK, CM.N_WINDOW
            p, _ = press.build_arm("floor_pos", n_ctx=b["n_ctx"], C=C_BUDGET, n_sink=NS, n_window=NW)
            p.compression_ratio = press._ratio_for(C_BUDGET + NS + NW, b["n_ctx"])
            dev = model.device
            ids = tok(b["pre"], add_special_tokens=False, return_tensors="pt").to(dev)
            cl = ids["input_ids"].shape[1]
            with p(model):
                out0 = model(**ids, use_cache=True)
            base_cache = out0.past_key_values

            for qi, v in enumerate(b["inst"].variants):
                mn = b["mns"][qi]
                gold = v.answer
                try:
                    max_rid = stored.get((i, qi))
                    if max_rid is None:
                        raise AssertionError("no stored max_rid")
                    rec_text = record_reference_text(b, max_rid)
                    proximity_query = f"Record {max_rid}: {rec_text}\n\n{v.query}"
                    _, post = CM.runner.templated_parts(tok, b["inst"].context, proximity_query)
                    q = tok(post, add_special_tokens=False, return_tensors="pt").to(dev)["input_ids"]
                    pos0 = torch.arange(cl, cl + q.shape[1], device=dev).unsqueeze(0)
                    from p3 import runner as P3R
                    with torch.inference_mode():
                        o0 = model(input_ids=q, past_key_values=P3R._clone(base_cache),
                                  position_ids=pos0, use_cache=True)
                    pos_after = torch.arange(cl + q.shape[1] - 1, cl + q.shape[1], device=dev).unsqueeze(0)
                    text, stop = decode_unmasked(model, tok, P3R._clone(o0.past_key_values),
                                                 pos_after, o0.logits[:, -1, :], mn)
                    score = CM.S4.SCORE_ONE[b["sp"]["task"]](text, gold)
                    fout.write(json.dumps(dict(model="M2", arm="floor_proximity_unmasked",
                              instance=i, qi=qi, rec_id=v.rec_id, max_rid=max_rid, gen=text,
                              gold=gold, score=score)) + "\n")
                    fout.flush()
                except Exception as e:
                    n_fail += 1
                    print(f"  FAIL inst{i} q{qi}: {type(e).__name__}: {e}", flush=True)
            del base_cache, out0
            torch.cuda.empty_cache()
            if (i - START + 1) % 10 == 0:
                print(f"    inst {i-START+1}/{n_run}  fails={n_fail}", flush=True)
    print(f"DONE: {n_run} instances, {n_fail} failures", flush=True)


if __name__ == "__main__":
    main()
