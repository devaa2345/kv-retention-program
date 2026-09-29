"""Item 2 part A: CPU-only check that decoding a selected sentence's retained token IDs
reproduces the source sentence text exactly, for every selected sentence in both frozen
confirmation sets (M2, M3). No GPU, no model load -- tokenizer only, frozen spans_for reused
unmodified.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, "out")
from p3 import runner as P3R
from transformers import AutoTokenizer
from _realtext_5070_data import MODELS, format_query, split_sentences
from _realtext_5070_prefill import spans_for

MANIFEST = Path("out/realtext_tight_confirm_160.jsonl")
RESULTS = {"M2": Path("out/realtext_tight_confirm_M2_results.jsonl"),
           "M3": Path("out/realtext_tight_confirm_M3_results.jsonl")}


def main():
    cases = {c["index"]: c for c in
            (json.loads(s) for s in MANIFEST.read_text(encoding="utf-8").splitlines())}
    for model_tag, model_name in (("M2", MODELS[0]), ("M3", MODELS[1])):
        tok = AutoTokenizer.from_pretrained(model_name)
        cache = {}
        n_checked, n_match, mismatches = 0, 0, []
        for line in RESULTS[model_tag].read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            sel = r.get("selected_sentence")
            if sel is None:
                continue
            inst = r["instance"]
            case = cases[inst]
            key = inst
            if key not in cache:
                q0 = case["queries"][0]
                pre, _ = P3R.templated_parts(tok, case["context"], format_query(q0["question"]))
                spans, labels, n_ctx = spans_for(tok, pre, case["context"])
                ids = tok(pre, add_special_tokens=False)["input_ids"]
                sentences = [s[2] for s in split_sentences(case["context"])]
                cache[key] = (spans, ids, sentences)
            spans, ids, sentences = cache[key]
            span_positions = sorted(spans[sel])
            retained_ids = [ids[p] for p in span_positions]
            decoded = tok.decode(retained_ids, skip_special_tokens=False)
            source = sentences[sel]
            n_checked += 1
            if decoded == source:
                n_match += 1
            else:
                mismatches.append(dict(model=model_tag, instance=inst, qi=r["qi"], arm=r["arm"],
                                       selected_sentence=sel, decoded=decoded, source=source))
        print(f"=== {model_tag}: {n_match}/{n_checked} exact reproductions ===")
        for m in mismatches[:15]:
            print(f"  MISMATCH inst{m['instance']} q{m['qi']} {m['arm']} sent{m['selected_sentence']}:")
            print(f"    decoded: {m['decoded']!r}")
            print(f"    source : {m['source']!r}")
        if len(mismatches) > 15:
            print(f"  ... and {len(mismatches)-15} more")
        out_path = Path(f"out/_retpolicy_dev/text_access_mismatches_{model_tag}.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for m in mismatches:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
