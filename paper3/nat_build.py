"""Write the natural-text dataset once. Usage: nat_build.py --version v1 --D 32 --n 200"""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from p3.natural import build as B
from p3.natural import prose

MODELS = {"M2": "Qwen/Qwen2.5-3B-Instruct", "M3": "meta-llama/Llama-3.2-3B-Instruct"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--D", type=int, default=32)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    toks = {t: AutoTokenizer.from_pretrained(m) for t, m in MODELS.items()}
    paras = prose.all_paragraphs()
    out = Path("data/natural") / f"nat_{a.version}.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for i in range(a.n):
            rec = B.build(a.version, "nat_%05d" % i, a.D, toks, paras)
            for r in rec["renderings"].values():
                r["prose_sources"] = [list(x) for x in r["prose_sources"]]
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            if i % 20 == 0:
                print(i, {t: (v["n_chars"], v["n_tokens"]) for t, v in rec["renderings"].items()},
                      "attempt", rec["attempt"], flush=True)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.with_suffix(".jsonl.sha256")).write_text(h + "  " + out.name + "\n")
    print(out, out.stat().st_size, h)


if __name__ == "__main__":
    main()
