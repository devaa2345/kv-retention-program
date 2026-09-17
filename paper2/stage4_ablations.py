"""Stage 4, part 2 — the two forward-pass probes, plus the competence anchor. PINNED ENV ONLY.

Run under the WSL toolchain (torch 2.11.0+cu128, transformers 5.2.0, kvpress 0.5.4), because
these produce admission-gating accuracy numbers and v1 §5.3 makes the backend a stratum rather
than an implementation detail.

Probes (v1 §4.2), both on `full_cache` -- no compression, so they test the TASK, not a policy:

    4. bindings deleted        -> must be ~ chance (1/96).  Verifies the binding sentence is
                                  load-bearing: with no binding there is no way to know which
                                  surname, so the model can only guess among the records.
    5. target record deleted   -> must be ~ 0.       Verifies no memorisation or leakage: the
                                  answer is simply absent, so any score above 0 means the value
                                  is recoverable from something other than its record line.

Also reported, because it is the other half of Stage 4 and comes free from the same forward
passes: the **competence anchor** -- unmodified `full_cache` accuracy, which v1 §4.5 requires
to sit in [0.55, 0.97] for a model to enter the grid. Above 0.97 the ceiling cannot be
distinguished from the floor; below 0.55 the floor effect dominates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from harness.keys import seed_key_without_seed
from harness.tasks import ledger

MAX_NEW = 32  # decision B9: 12 was a model-dependent format confound
CHANCE = 1.0 / ledger.N_RECORDS


def build_prompt(tok, context: str, query: str) -> str:
    msgs = [{"role": "user", "content": f"{context}\n\n{query}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def strip_bindings(inst: ledger.LedgerInstance) -> str:
    """Remove every binding sentence (all H of them), leaving records and filler."""
    ctx = inst.context
    for sp in inst.candidates:
        if sp.kind == "binding":
            ctx = ctx.replace(sp.text, "")
    return ctx


def strip_target_record(inst: ledger.LedgerInstance) -> str:
    """Delete the record line of EVERY variant's target, since all H variants are scored."""
    ctx = inst.context
    for v in inst.variants:
        ctx = ctx.replace(v.gold[1].text + "\n", "").replace(v.gold[1].text, "")
    return ctx


@torch.inference_mode()
def run_variants(model, tok, ctx: str, inst) -> list[str]:
    """One generation per query variant; decision 7 scores their mean."""
    outs = []
    for v in inst.variants:
        msgs = [{"role": "user", "content": f"{ctx}\n\n{v.query}"}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(p, return_tensors="pt").to(model.device)
        g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        outs.append(tok.decode(g[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    rev = getattr(model.config, "_commit_hash", None) or "unresolved"

    insts = []
    for i in range(args.n):
        iid = f"abl{i:05d}"
        seed = seed_key_without_seed(
            task="ledger", instance_id=iid, model=args.model, model_revision=rev,
            arm="full_cache", B=1, protocol="agnostic", device="nvidia",
            backend="cuda-12.8", torch_version=torch.__version__,
            transformers_version="5.2.0", kvpress_version="0.5.4", dtype="bfloat16",
        )
        insts.append(ledger.build(seed, iid, target_tokens=2048, tokenizer=tok))

    variants = {
        "full_cache_anchor": lambda i: i.context,
        "bindings_deleted": strip_bindings,
        "target_record_deleted": strip_target_record,
    }

    res: dict = {"model": args.model, "model_revision": rev, "n": args.n,
                 "chance": round(CHANCE, 4), "variants": {}}
    for name, ctxfn in variants.items():
        acc = wfr = 0.0
        samples: list[str] = []
        for inst in insts:
            outs = run_variants(model, tok, ctxfn(inst), inst)
            acc += ledger.score_instance(outs, inst)          # mean over H variants
            wfr += sum(bool(re.search(r"\b\d{6}\b", o)) for o in outs) / len(outs)
            if len(samples) < 3:
                samples.append(outs[0])
        res["variants"][name] = {
            "accuracy": round(acc / args.n, 4),
            "well_formed_rate": round(wfr / args.n, 4),
            "sample": samples,
        }
        print(f"  {name:24s} acc={acc / args.n:.4f}  well-formed={wfr / args.n:.3f}")
    a = res["variants"]["full_cache_anchor"]["accuracy"]
    b = res["variants"]["bindings_deleted"]["accuracy"]
    c = res["variants"]["target_record_deleted"]["accuracy"]
    res["gates"] = {
        "competence_anchor_in_band": bool(0.55 <= a <= 0.97),
        "competence_anchor": a, "band": [0.55, 0.97],
        "bindings_deleted_near_chance": bool(b <= CHANCE + 0.02),
        "target_deleted_near_zero": bool(c <= 0.02),
    }
    # The two ablations are VACUOUS unless the task is answerable at all: at anchor 0.000 both
    # read <= threshold trivially and prove nothing. The anchor is a precondition, not a
    # co-equal criterion.
    res["gates"]["pass"] = bool(
        res["gates"]["competence_anchor_in_band"]
        and res["gates"]["bindings_deleted_near_chance"]
        and res["gates"]["target_deleted_near_zero"]
    )
    if not res["gates"]["competence_anchor_in_band"]:
        res["gates"]["note"] = (
            "ablation probes are VACUOUS at this anchor -- they read <= threshold only because "
            "nothing is answerable. Fix the anchor before reading them as evidence."
        )
    print(f"\n  competence anchor {a:.4f} in [0.55, 0.97]: "
          f"{'YES' if res['gates']['competence_anchor_in_band'] else 'NO — needs N/H tuning'}")
    print(f"  bindings deleted  <= chance+0.02 ({CHANCE + 0.02:.4f}): "
          f"{'PASS' if res['gates']['bindings_deleted_near_chance'] else 'FAIL'}")
    print(f"  target deleted    <= 0.02: "
          f"{'PASS' if res['gates']['target_deleted_near_zero'] else 'FAIL'}")

    tag = args.model.split("/")[-1].replace(".", "_")
    p = Path(__file__).resolve().parent / "gates" / "nvidia" / f"stage4_ablations_{tag}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
