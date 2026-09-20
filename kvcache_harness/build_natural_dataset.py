"""Build a shorter variant of the Paper 3 natural-text dataset for THIS harness (Paper 1, Option B item 2).

Paper 3's nat_v1.jsonl is 4,099 tokens per context; this harness prefills the whole first turn in
one forward with output_attentions=True, which does not fit on a 12 GB card at that length (28 layers
x 12 heads x 4099^2 bf16 = ~11 GB of attention weights alone). This script reuses Paper 3's
generator, prose sources and fact template UNCHANGED and only changes the context-length target
(and the distractor count D so the fact-to-context ratio stays workable), tokenizing with the Qwen
tokenizer only (Llama is not needed: the run model is Qwen2.5-1.5B-Instruct).

The output is a NEW dataset and is described as one wherever it is used. It is not nat_v1.

Usage: python -m kvcache_harness.build_natural_dataset --target 2048 --D 16 --n 100
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper3"))
from transformers import AutoTokenizer                       # noqa: E402
from p3.natural import build as B                             # noqa: E402
from p3.natural import prose                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2048)
    ap.add_argument("--tol", type=int, default=32)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--version", default="l2048_v1")
    a = ap.parse_args()
    B.TARGET, B.TOL = a.target, a.tol
    tok = {"M2": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")}
    paras = prose.all_paragraphs()
    out = ROOT / "data" / "natural_l2048" / f"nat_{a.version}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(a.n):
            rec = B.build(a.version, "nat_%05d" % i, a.D, tok, paras)
            for r in rec["renderings"].values():
                r["prose_sources"] = [list(x) for x in r["prose_sources"]]
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            if i % 10 == 0:
                print(i, {t: (v["n_chars"], v["n_tokens"]) for t, v in rec["renderings"].items()},
                      "attempt", rec["attempt"], flush=True)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".jsonl.sha256").write_text(h + "  " + out.name + "\n")
    print(out, out.stat().st_size, h)


if __name__ == "__main__":
    main()
