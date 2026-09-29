"""Item 4: bridge two-hop pilot, competence gate ONLY. Per PREREG_BRIDGE_TWOHOP_DESIGN.md.
40 instances, both models, two arms:
  - full_cache: entire distractor-setting context (all paragraphs), no compression, no mask.
  - top1_sentence: BM25-over-all-sentences (same frozen tie rule as the real-text thread) picks
    one sentence; that sentence alone (no other context) is fed with the question.

No retention/compression arm is run here -- this is the gate check only, per instruction: if
full_cache is below the [0.55, 0.97] band on either model, stop before any further bridge work.
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "out")
from _realtext_pointer_screen import lexical

MODELS = {"M2": "Qwen/Qwen2.5-3B-Instruct", "M3": "meta-llama/Llama-3.2-3B-Instruct"}
PROMPT_SUFFIX = "\n\nAnswer the question in as few words as possible, using only the text above."


def norm(s):
    return " ".join(s.split()).casefold()


def build_cases(n):
    d = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    bridges = [r for r in d if r["type"] == "bridge"]
    cases = []
    for r in bridges:
        titles = r["context"]["title"]
        sents_per_para = r["context"]["sentences"]
        all_sents = []
        for para_sents in sents_per_para:
            for s in para_sents:
                s = s.strip()
                if s:
                    all_sents.append(s)
        if not all_sents:
            continue
        full_text = " ".join(all_sents)
        if r["answer"].strip().casefold() not in full_text.casefold():
            continue  # skip if the answer isn't even literally in the context (non-extractive)
        cases.append(dict(id=r["id"], question=r["question"], answer=r["answer"],
                          sentences=all_sents, full_text=full_text))
        if len(cases) >= n:
            break
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("M2", "M3"), required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cases = build_cases(args.n)
    print(f"built {len(cases)} extractive bridge cases", flush=True)

    name = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16,
                                                 attn_implementation="eager").to("cuda").eval()

    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as fout:
        for idx, case in enumerate(cases):
            # top-1 sentence via BM25 over ALL sentences
            row = {"eligible": list(range(len(case["sentences"]))),
                  "sentences": {str(i): s for i, s in enumerate(case["sentences"])},
                  "question": case["question"]}
            scores = lexical(row)
            top1 = max(range(len(case["sentences"])), key=lambda si: (scores.get(si, 0.0), -si))
            top1_text = case["sentences"][top1]

            for arm, context_text in (("full_cache", case["full_text"]),
                                      ("top1_sentence", top1_text)):
                prompt = context_text + "\n\nQuestion: " + case["question"] + PROMPT_SUFFIX
                messages = [{"role": "user", "content": prompt}]
                input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                ids = tok(input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
                n_ctx = ids["input_ids"].shape[1]
                max_new = len(tok(case["answer"], add_special_tokens=False)["input_ids"]) + 16
                with torch.inference_mode():
                    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                        pad_token_id=tok.eos_token_id)
                gen_ids = out[0][ids["input_ids"].shape[1]:]
                gen = tok.decode(gen_ids, skip_special_tokens=True)
                exact = int(norm(gen) == norm(case["answer"]))
                contains = int(norm(case["answer"]) in norm(gen))
                fout.write(json.dumps(dict(model=args.model, idx=idx, id=case["id"], arm=arm,
                          n_ctx=n_ctx, question=case["question"], answer=case["answer"],
                          gen=gen, exact=exact, contains=contains)) + "\n")
                fout.flush()
                del out, ids
                torch.cuda.empty_cache()
            if (idx + 1) % 10 == 0:
                print(f"    {idx+1}/{len(cases)} elapsed={(time.time()-t0)/60:.2f} min", flush=True)
    print(f"DONE {args.model}: {len(cases)} cases, elapsed={(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
