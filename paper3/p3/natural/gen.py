"""Generation for the natural-text ladder: identical to runner.generate_with except that a
query stops as soon as the first answer block has ended.

Why: the v2 scorer reads only the first answer block (see score.first_answer), and Llama-3.2
routinely continues past it with an invented 'B: ...' record until the token cap. Under greedy
decoding the text up to the block boundary is identical with or without the early stop, so this
changes cost, not scores; it is what makes a generous cap (needed by Llama's verbose prose
answers, audit finding 2026-09-19) affordable across ~30 arm-budget cells.
"""
import torch

from p3 import runner
from p3.natural.score import _CONT


@torch.inference_mode()
def generate_stop(model, tok, pre, posts, p, max_new):
    dev = model.device
    ids = tok(pre, add_special_tokens=False, return_tensors="pt").to(dev)
    context_length = ids["input_ids"].shape[1]          # UNCOMPRESSED -- load-bearing
    ctx = p(model) if p is not None else runner._null_ctx()
    with ctx:
        out = model(**ids, use_cache=True)
    base = out.past_key_values
    texts, flags = [], []
    for post, mn in zip(posts, max_new):
        cache = runner._clone(base)
        q = tok(post, add_special_tokens=False, return_tensors="pt").to(dev)["input_ids"]
        pos = torch.arange(context_length, context_length + q.shape[1], device=dev).unsqueeze(0)
        cur, toks, stop = q, [], "cap"
        for step in range(mn):
            o = model(input_ids=cur, past_key_values=cache, position_ids=pos, use_cache=True)
            cache = o.past_key_values
            nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            t = int(nxt)
            if t == tok.eos_token_id:
                stop = "eos"
                break
            toks.append(t)
            cur = nxt
            pos = pos[:, -1:] + 1
            if step >= 3 and _CONT.search(tok.decode(toks, skip_special_tokens=True).lstrip()):
                stop = "block"
                break
        texts.append(tok.decode(toks, skip_special_tokens=True))
        flags.append(stop)
    del base
    return texts, flags
