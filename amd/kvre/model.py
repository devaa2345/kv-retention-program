"""Model loading and prefill helpers (spec section 1)."""
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def load(model_id: str = MODEL_ID, device: str = "cuda"):
    # All model math runs on the GPU; torch's CPU thread pool only does dispatch bookkeeping.
    # It defaults to one thread per core (24 here), which shows up as ~200% CPU per process for
    # no benefit. Capping it keeps the host free without touching GPU throughput.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass          # already initialised in this process
    tok = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, attn_implementation="eager")
    model = model.to(device).eval()
    model.name_or_path_recorded = model_id
    assert model.dtype == torch.bfloat16, f"expected bfloat16, got {model.dtype}"
    return model, tok


def kv_tensors_from_cache(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(l.keys, l.values) for l in cache.layers]
