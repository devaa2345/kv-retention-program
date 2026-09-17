from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load with eager attention — we need real per-step attention weights
    (output_attentions=True) to score tokens for every policy in this
    harness, which SDPA/FlashAttention backends don't expose. That is
    itself a deliberate, measured cost (ties into Phase 3 of the scoping
    doc later), not an oversight.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tokenizer
