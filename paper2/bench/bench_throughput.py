"""Machine-N throughput re-measurement (plan v2 §1, footnote to the timings table).

    "Timings assume 2048-token prefill + 12 decode, bf16, batch 1: 5070 ~ 0.6 s (1.5B)
     ... Re-measure in N3 on 200 records and rescale this whole table before committing
     to a schedule."

Produces NO run records and touches no arm. This measures the machine, not the design,
so it is safe to run ahead of the N1 gate. Writes bench/throughput_nvidia.json.

Two configurations are timed, because they are not the same cost:
  * plain      -- prefill + decode, no attention tensors (arms that need no attention ranking)
  * attentions -- output_attentions=True under eager attention, which every attention-ranked
                  arm requires. At 2048 tokens the attention tensors are large enough to be
                  the binding VRAM constraint on a 12 GB card, so this is measured, not assumed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PREFILL = 2048
DECODE = 32  # B9


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _decode_loop(model, out, input_ids, n_steps: int, want_attn: bool):
    """Greedy decode mirroring the real harness: one token at a time, cache carried."""
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for _ in range(n_steps):
        out = model(
            input_ids=nxt,
            past_key_values=past,
            use_cache=True,
            output_attentions=want_attn,
        )
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return past


def time_config(model, ids, want_attn: bool, iters: int, warmup: int) -> dict:
    times: list[float] = []
    peak_mib = 0.0
    for i in range(warmup + iters):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        _sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(
                input_ids=ids, use_cache=True, output_attentions=want_attn
            )
            _decode_loop(model, out, ids, DECODE, want_attn)
        _sync()
        dt = time.perf_counter() - t0
        if torch.cuda.is_available():
            peak_mib = max(peak_mib, torch.cuda.max_memory_allocated() / 1024**2)
        if i >= warmup:
            times.append(dt)
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    times.sort()
    return {
        "seconds_mean": round(statistics.fmean(times), 4),
        "seconds_median": round(statistics.median(times), 4),
        "seconds_p90": round(times[int(0.9 * (len(times) - 1))], 4),
        "seconds_min": round(times[0], 4),
        "seconds_max": round(times[-1], 4),
        "peak_alloc_mib": round(peak_mib),
        "iters": len(times),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--skip-attn", action="store_true")
    ap.add_argument("--attn", default="eager", choices=["eager","sdpa"])
    args = ap.parse_args()

    print(f"loading {args.model} (bfloat16, eager) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn
    ).to("cuda").eval()

    rev = getattr(model.config, "_commit_hash", None) or "unresolved"
    n_layer = model.config.num_hidden_layers
    n_head = model.config.num_attention_heads
    print(f"revision={rev} layers={n_layer} heads={n_head}", flush=True)

    # A realistic prefill of exactly PREFILL tokens (content is irrelevant to timing).
    ids = torch.randint(
        1000, 20000, (1, PREFILL), device="cuda", dtype=torch.long
    )

    result = {
        "model": args.model,
        "model_revision": rev,
        "prefill_tokens": PREFILL,
        "decode_tokens": DECODE,
        "dtype": "bfloat16",
        "attn_implementation": args.attn,
        "batch_size": 1,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "configs": {},
    }

    print("timing: plain (no attention tensors) ...", flush=True)
    result["configs"]["plain"] = time_config(model, ids, False, args.iters, args.warmup)
    print("  ", result["configs"]["plain"], flush=True)

    if not args.skip_attn:
        print("timing: output_attentions=True (attention-ranked arms) ...", flush=True)
        try:
            result["configs"]["attentions"] = time_config(
                model, ids, True, max(4, args.iters // 3), 1
            )
            print("  ", result["configs"]["attentions"], flush=True)
        except torch.cuda.OutOfMemoryError as e:
            result["configs"]["attentions"] = {"oom": True, "error": str(e)[:300]}
            print("  OOM -- recorded, this is a finding not a failure", flush=True)
            torch.cuda.empty_cache()

    tag = args.model.split("/")[-1].replace(".","_")
    out = Path(__file__).resolve().parent / f"throughput_{tag}_{args.attn}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")

    # Rescale the plan's estimates.
    plan_s = 0.6
    for name, cfg in result["configs"].items():
        if "seconds_median" in cfg:
            m = cfg["seconds_median"]
            print(f"  {name:11s} {m:6.3f} s/record  = {m / plan_s:5.2f}x the plan's 0.6 s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
