"""Generation engine: chunked prefill, per-step policy application, multi-turn interaction.

Spec ref: sections 1-7. Resolutions: [GAP-E] one cache across all turns, [GAP-F] answer tokens
are evictable, [GAP-M] prefill attention seeds the accumulator, [SPEC-GAP 4] score aggregation.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

import torch

from .cache_engine import QuantAudit, apply_tiers, cache_bytes
from .policy import PolicyConfig, assign_tiers, select_retained
from .task import MAX_ANSWER_TOKENS, Prompt, question_for, target_for

LINE_RE = re.compile(r"^[A-Z][A-Z0-9_]*:\s*sk-[0-9a-f]{14}$")

_PERM_CACHE: dict[int, list[int]] = {}


def _stable_perm(n: int) -> list[int]:
    """A fixed pseudorandom permutation of range(n), stable across processes and runs."""
    p = _PERM_CACHE.get(n)
    if p is None:
        p = list(range(n))
        random.Random(20260904).shuffle(p)
        _PERM_CACHE[n] = p
    return p


@dataclass
class SlotMeta:
    """Per-cache-slot metadata, index_select'd alongside the cache on every eviction."""
    abs_pos: list[int] = field(default_factory=list)
    line_id: list[int] = field(default_factory=list)      # -1 if not part of a matched line
    protected: list[bool] = field(default_factory=list)   # matches the structural line pattern
    cred_idx: list[int] = field(default_factory=list)     # -1 if not a credential span token

    def select(self, keep: list[int]) -> "SlotMeta":
        return SlotMeta(
            abs_pos=[self.abs_pos[i] for i in keep],
            line_id=[self.line_id[i] for i in keep],
            protected=[self.protected[i] for i in keep],
            cred_idx=[self.cred_idx[i] for i in keep],
        )

    def extend(self, n: int, start_abs: int) -> None:
        for j in range(n):
            self.abs_pos.append(start_abs + j)
            self.line_id.append(-1)
            self.protected.append(False)
            self.cred_idx.append(-1)

    def __len__(self) -> int:
        return len(self.abs_pos)


def annotate_context(tok, text: str, prompt: Prompt) -> tuple[list[int], list[bool], list[int]]:
    """Map character spans of matched lines and credential spans onto token indices."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    line_id = [-1] * n
    protected = [False] * n
    cred_idx = [-1] * n

    # structural lines: every LABEL: sk-xxxx line, credential or distractor alike.
    # Structural protection is content-agnostic pattern matching -- it cannot tell which
    # matched lines are credentials. That is what separates arm 2 from arm 5.
    lid = 0
    for m in re.finditer(r"[A-Z][A-Z0-9_]*:\s*sk-[0-9a-f]{14}", text):
        a, b = m.span()
        for i, (s, e) in enumerate(offsets):
            if e > a and s < b:
                line_id[i] = lid
                protected[i] = True
        lid += 1

    # credential label+value spans (oracle only). Labels AND values, per section 5.
    for ci, cred in enumerate(prompt.credentials):
        for m in re.finditer(re.escape(cred.line), text):
            a, b = m.span()
            for i, (s, e) in enumerate(offsets):
                if e > a and s < b:
                    cred_idx[i] = ci
    return line_id, protected, cred_idx


class Engine:
    def __init__(self, model, tok, device="cuda"):
        self.model = model
        self.tok = tok
        self.device = device
        self.n_layers = model.config.num_hidden_layers

    # ---------------------------------------------------------------- attention accumulation
    @staticmethod
    def _bump_counts(counts: dict, kv_len: int, q: int, dev) -> None:
        """Number of queries that could causally attend to each position.

        For a chunk of q queries occupying the last q slots of kv_len, position j is attended
        by queries with global index >= j, i.e. min(q, kv_len - j) of them.
        """
        cur = counts.get("v")
        if cur is None or cur.numel() < kv_len:
            base = torch.zeros(kv_len, dtype=torch.float64, device=dev)
            if cur is not None:
                base[:cur.numel()] = cur
            cur = base
        j = torch.arange(kv_len, dtype=torch.float64, device=dev)
        cur += torch.clamp(kv_len - j, min=0.0, max=float(q))
        counts["v"] = cur

    def _accumulate(self, scores_t: dict, attentions, kv_len: int) -> None:
        """[SPEC-GAP 4]: mean over heads, sum over layers, accumulated over all steps.

        The per-layer reduction order is unchanged from the original list-based version, and
        the accumulator is float64 -- exactly what a Python float list held -- so this is
        bitwise identical to the earlier implementation, not merely equivalent.
        """
        acc = torch.zeros(kv_len, dtype=torch.float32, device=self.device)
        for a in attentions:                       # (B, H, q, kv)
            acc += a[0].float().mean(dim=0).sum(dim=0)[:kv_len]
        cur = scores_t.get("v")
        if cur is None:
            scores_t["v"] = acc.double()
        else:
            if cur.numel() < kv_len:
                cur = torch.cat([cur, torch.zeros(kv_len - cur.numel(),
                                                  dtype=torch.float64, device=self.device)])
            cur[:kv_len] += acc.double()
            scores_t["v"] = cur

    @staticmethod
    def _epiphany(hidden_states, out: list[float]) -> None:
        """[SPEC-GAP 7]: cross-layer representational change, mean over layers, per position."""
        q = hidden_states[0].shape[1]
        acc = torch.zeros(q, dtype=torch.float32, device=hidden_states[0].device)
        for l in range(1, len(hidden_states)):
            prev = hidden_states[l - 1][0].float()
            cur = hidden_states[l][0].float()
            acc += (cur - prev).norm(dim=-1) / (prev.norm(dim=-1) + 1e-6)
        acc /= max(1, len(hidden_states) - 1)
        out.extend(acc.tolist())

    # ---------------------------------------------------------------- forward passes
    def _forward(self, ids, cache, abs_start, scores, epi=None, chunk=512, counts=None):
        """Run `ids` through the model, chunked, accumulating attention (and epiphany)."""
        n = ids.shape[-1]
        logits = None
        for s in range(0, n, chunk):
            piece = ids[:, s:s + chunk]
            q = piece.shape[-1]
            past_len = 0 if cache is None else cache.layers[0].keys.shape[-2]
            pos = torch.arange(abs_start + s, abs_start + s + q,
                               device=self.device).unsqueeze(0)
            cache_pos = torch.arange(past_len, past_len + q, device=self.device)
            attn_mask = torch.ones((1, past_len + q), dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = self.model(
                    input_ids=piece,
                    past_key_values=cache,
                    attention_mask=attn_mask,
                    position_ids=pos,
                    cache_position=cache_pos,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=epi is not None,
                )
            cache = out.past_key_values
            kvl = cache.layers[0].keys.shape[-2]
            self._accumulate(scores, out.attentions, kvl)
            if counts is not None:
                self._bump_counts(counts, kvl, q, self.device)
            if epi is not None:
                self._epiphany(out.hidden_states, epi)
            logits = out.logits[:, -1, :]
            del out
        return cache, logits

    # ---------------------------------------------------------------- policy application
    def _apply_policy(self, cache, meta, scores, epi, cfg, audit, rng, rr, oracle_future,
                      diag_sink: list, counts=None):
        n = len(meta)
        if cfg.eviction == "none" or n <= cfg.total_budget:
            return cache, meta, scores, epi, None

        is_oracle_must = [meta.cred_idx[i] >= 0 for i in range(n)]
        # One host transfer per policy application (was one per position, per forward call).
        # [SPEC-GAP 4] "mean" divides accumulated mass by the number of queries that could
        # causally attend to each position, removing the pure positional artifact whereby
        # early positions accumulate mass simply for having been present longer.
        # The division is done on-device in float64 and transferred once. IEEE-754 division is
        # exactly specified, so this is bitwise identical to dividing the two transferred lists
        # elementwise in Python, while halving host transfers and removing an O(n) Python loop.
        sv = scores["v"][:n]
        if cfg.score_norm == "mean" and counts.get("v") is not None:
            rank_scores = (sv / counts["v"][:n].clamp(min=1.0)).tolist()
        else:
            rank_scores = sv.tolist()
        keep, diag = select_retained(
            n, cfg.total_budget, rank_scores, meta.line_id, meta.protected, is_oracle_must, cfg)

        # [GAP-B] G5 dormancy is evaluated *here*, before eviction discards the evidence.
        # A credential span is dormant this step if its best member score is below the budget
        # cut line while the whole span sits outside the recency window.
        cut = diag.get("cut_score", float("-inf"))
        best: dict[int, float] = {}
        newest: dict[int, int] = {}
        for i in range(n):
            ci = meta.cred_idx[i]
            if ci >= 0:
                if ci not in best or rank_scores[i] > best[ci]:
                    best[ci] = rank_scores[i]
                newest[ci] = max(newest.get(ci, -1), i)
        recency_start = n - cfg.recency_window
        dormant_now = sorted(ci for ci in best
                             if best[ci] < cut and newest[ci] < recency_start)
        diag["dormant_creds"] = dormant_now
        diag["n_creds_present"] = len(best)

        # Explicit promotion score per signal. Section 7 requires rho(eviction, promotion) to be
        # +1.000 for P1 and near zero for the rest; any signal that silently falls back to the
        # eviction score cannot satisfy that, so none of them do now.
        if cfg.promotion == "oracle":
            # P5: membership only. Previously non-must positions inherited rank_scores, which
            # correlated the "oracle" signal with the eviction signal by construction (rho 0.85).
            promo = [1.0 if (meta.cred_idx[i] in oracle_future) else 0.0 for i in range(n)]
        elif cfg.promotion == "epiphany":
            promo = epi if epi else rank_scores
        elif cfg.promotion == "random":
            promo = [rng.random() for _ in range(n)]
        elif cfg.promotion == "roundrobin":
            # P4: rotate over a FIXED arbitrary ordering rather than raw position. Rotating over
            # position made the signal positional, and position correlates with accumulated
            # attention (rho 0.40). The permutation is seed-stable, so runs stay reproducible.
            perm = _stable_perm(n)
            promo = [float((perm[i] + rr) % n) for i in range(n)]
        else:
            promo = rank_scores
        qmask = assign_tiers(keep, n, promo, cfg, rng=rng, rr_counter=rr)

        apply_tiers(cache, keep, qmask, cfg.quant_bits, audit=audit, tag="g")
        meta = meta.select(keep)
        kidx = torch.tensor(keep, dtype=torch.long, device=self.device)
        scores["v"] = scores["v"][:n].index_select(0, kidx)
        if counts.get("v") is not None:
            counts["v"] = counts["v"][:n].index_select(0, kidx)
        epi = [epi[i] for i in keep] if epi else epi
        diag["n_quant"] = sum(qmask)
        diag["n_full"] = len(qmask) - sum(qmask)
        diag_sink.append(diag)
        return cache, meta, scores, epi, diag

    # ---------------------------------------------------------------- main entry
    def run_prompt(self, prompt: Prompt, cfg: PolicyConfig, seed: int,
                   audit: QuantAudit | None = None, collect_dormancy: bool = False) -> dict:
        cfg.assert_budget_valid()
        tok, model = self.tok, self.model
        rng = random.Random(seed)
        scores: dict = {}
        counts: dict = {}
        epi: list[float] = []
        need_epi = cfg.promotion == "epiphany"
        meta = SlotMeta()
        cache = None
        abs_pos = 0
        diags: list = []
        outputs: list[str] = []
        dormancy_events = 0
        dormancy_run = 0
        dormancy_max_run = 0
        retained_signature: list[int] = []

        first_text = tok.apply_chat_template(
            [{"role": "user", "content": prompt.context + "\n\n" + question_for(prompt, 0)}],
            tokenize=False, add_generation_prompt=True)
        line_id, protected, cred_idx = annotate_context(tok, first_text, prompt)
        ids = tok(first_text, return_tensors="pt", add_special_tokens=False).to(self.device)
        n0 = ids["input_ids"].shape[-1]

        cache, logits = self._forward(ids["input_ids"], None, abs_pos, scores,
                                      epi if need_epi else None, counts=counts)
        abs_pos += n0
        meta.abs_pos = list(range(n0)); meta.line_id = line_id[:n0]
        meta.protected = protected[:n0]; meta.cred_idx = cred_idx[:n0]
        ctx_len = n0

        for turn in range(prompt.n_turns):
            if turn > 0:
                q = question_for(prompt, turn)
                qtext = f"<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
                qids = tok(qtext, return_tensors="pt", add_special_tokens=False).to(self.device)
                nq = qids["input_ids"].shape[-1]
                meta.extend(nq, abs_pos)
                cache, logits = self._forward(qids["input_ids"], cache, abs_pos, scores,
                                              epi if need_epi else None, counts=counts)
                abs_pos += nq

            oracle_future = [prompt.turn_order[t] for t in range(turn, prompt.n_turns)]
            gen_ids: list[int] = []
            for step in range(MAX_ANSWER_TOKENS):
                cache, meta, scores, epi, d = self._apply_policy(
                    cache, meta, scores, epi, cfg, audit, rng, len(diags), oracle_future,
                    diags, counts)
                if collect_dormancy and d is not None:
                    if d.get("dormant_creds"):
                        dormancy_events += 1
                        dormancy_run += 1
                        dormancy_max_run = max(dormancy_max_run, dormancy_run)
                    else:
                        dormancy_run = 0
                nxt = int(torch.argmax(logits, dim=-1).item())
                gen_ids.append(nxt)
                if nxt == tok.eos_token_id:
                    break
                meta.extend(1, abs_pos)
                t = torch.tensor([[nxt]], device=self.device)
                cache, logits = self._forward(t, cache, abs_pos, scores,
                                              epi if need_epi else None, counts=counts)
                abs_pos += 1
            outputs.append(tok.decode(gen_ids, skip_special_tokens=True))
            if turn == 0:
                retained_signature = list(meta.abs_pos)

        k = sum(1 for t, g in enumerate(outputs) if target_for(prompt, t).value in g)
        n_quant = diags[-1].get("n_quant", 0) if diags else 0
        n_full = diags[-1].get("n_full", len(meta)) if diags else len(meta)
        return {
            "accuracy": k / len(prompt.turn_order),
            "k": k,
            "n_turns": len(prompt.turn_order),
            "outputs": outputs,
            "context_length": ctx_len,
            "effective_tokens": len(meta),
            "n_full": n_full,
            "n_quant": n_quant,
            "bytes": cache_bytes(n_full, n_quant, cfg.quant_byte_cost),
            "oracle_oversubscribed": any(d.get("oracle_oversubscribed") for d in diags),
            "n_protected_lines": diags[-1].get("n_protected_lines", 0) if diags else 0,
            "dormancy_events": dormancy_events,
            "dormancy_max_run": dormancy_max_run,
            "creds_present_final": diags[-1].get("n_creds_present", 0) if diags else 0,
            "retained_signature": retained_signature,
            "first_token_eos": [o.strip() == "" for o in outputs],
        }
