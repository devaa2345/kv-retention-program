"""Floor sensitivity: is the paper's denominator robust to the preregistration's ambiguity?

PREREG_P2_v2 defines `floor_pos` twice, incompatibly:

  READING A  (sec 3.12(b))  sink + last B-8      -> retains B = C + 72 tokens   [WHAT WAS RUN]
  READING B  (sec 4.1)      n_sink=8 + last C-8  -> retains C tokens

The primary used Reading A, established from `harness/press.py:183-186` (`_ratio_for(B, n_ctx)`
asserts exactly B retained entries). `floor_pos` is the denominator of every G_m and every I in
the paper, so the question a reviewer will ask is not which reading is correct but whether the
conclusions survive either.

Only A_floor has to be re-measured. The oracle arms pad from the MANDATORY floor region
(`ladder._floors` -> sinks + window, then `_pad_from_floor` over the compressible region), so
`oracle_causal` and `oracle_prescient` retain 72 + C = B regardless of how the floor_pos ARM is
parameterised. Verified by reading `harness/ladder.py:92-101,157-168`. Method arms likewise
retain B. So A_m, A_causal and A_presc are taken unchanged from the main grid and only the floor
is re-run: 11 cells x 200 instances = 2,200 records rather than 11,000.

NOTE, and it is the substantive point rather than a caveat: under Reading B at C=16 the floor arm
retains 16 tokens while every other arm retains 88. Reading B therefore breaks the equal-B budget
parity that every other arm in the ladder honours, and at C < 72 it cannot even contain the
mandatory 8-sink + 64-window floor that PREREG sec 4.1's own budget accounting requires of all
arms. That is measured here, not asserted.

Records are written under protocol="agnostic_floorC" so they can never collide with a canonical
cell.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness import ladder, press
from harness.keys import RecordKey, assert_owned_by, seed_key_without_seed
from harness.tasks import ledger
from stage5_ladder_validation import facts_and_ctx, generate_with, templated_parts

N_SINK, N_WINDOW = 8, 64
ENV = dict(backend="cuda-12.8", transformers_version="5.2.0", kvpress_version="0.5.4",
           dtype="bfloat16")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["key_digest"])
            except Exception:
                pass
    print(f"  {args.model}: resuming with {len(done)} records present", flush=True)

    budgets = [int(b) for b in args.budgets.split(",")]
    t0 = time.time()
    written = failed = 0

    for i in range(args.n):
        iid = f"grid_{i:05d}"
        sd = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev,
            arm="grid", B=1, protocol="agnostic", device="nvidia",
            torch_version=torch.__version__, seed=0, **ENV)
        inst = ledger.build(sd, iid, target_tokens=2048, tokenizer=tok)
        pre, _ = templated_parts(tok, inst.context, "")
        facts, n_ctx = facts_and_ctx(inst, tok, pre)
        posts = [templated_parts(tok, inst.context, v.query)[1] for v in inst.variants]

        for C in budgets:
            B = C + N_SINK + N_WINDOW
            k = RecordKey(task="ledger", instance_id=iid, model=args.model,
                          model_revision=rev, arm="floor_pos_readingC", B=B,
                          protocol="agnostic_floorC", device="nvidia",
                          torch_version=torch.__version__, seed=sd, batch_size=1, **ENV)
            assert_owned_by(k, "nvidia")
            if k.digest() in done:
                continue
            try:
                # READING B: retain exactly C entries -- 8 sinks + the most recent C-8.
                p = press.FloorPosPress(n_sink=N_SINK)
                p.compression_ratio = press._ratio_for(C, n_ctx)
                score = ledger.score_instance(
                    generate_with(model, tok, pre, posts, p), inst)

                # what the two readings actually retain, measured not assumed
                sel_B = ladder.floor_pos(n_ctx, C, N_SINK, N_WINDOW, facts)
                mand, _ = ladder._floors(n_ctx, N_SINK, N_WINDOW)
                keptC = set(range(min(N_SINK, n_ctx))) | set(range(max(0, n_ctx - (C - N_SINK)), n_ctx))

                rec = dict(key_digest=k.digest(), key=k.as_dict(), score=score, C=C,
                           n_ctx=n_ctx,
                           n_kept_readingC=len(keptC),
                           n_kept_readingB=len(set(sel_B.kept)),
                           readingC_contains_mandatory_floor=bool(mand <= keptC),
                           n_mandatory_missing_in_readingC=len(mand - keptC))
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                done.add(k.digest())
                written += 1
            except Exception as e:
                failed += 1
                with out.with_suffix(".failures.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key_digest": k.digest(), "C": C,
                                        "error": f"{type(e).__name__}: {e}"[:300]}) + "\n")
        if (i + 1) % 20 == 0:
            el = (time.time() - t0) / 60
            print(f"  [{i+1}/{args.n}] written {written} failed {failed} ({el:.1f} min)",
                  flush=True)

    expected = args.n * len(budgets)
    print(f"\n  COVERAGE {args.tag}: expected {expected} have {len(done)} "
          f"holes {expected - len(done)} failed {failed}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
