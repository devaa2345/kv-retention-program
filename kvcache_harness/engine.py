"""The shared generation engine. Every arm (permanent eviction, structural
protection, recoverable tiering, oracle_static) runs through this same loop
— only the `EvictionPolicy` plugged in differs. This is what makes the
comparison matched: identical model, identical decoding, identical budget
accounting, identical scoring instrumentation.

Correctness notes (read before touching the cache-mutation code):

1. We only ever decode ONE new token per forward call (never multi-token
   chunks) once eviction is active. That's what lets us physically shrink
   the DynamicCache's tensors (drop evicted columns via index_select)
   without breaking transformers' causal-mask construction: for a
   query length of 1, transformers' `kv_idx <= q_idx` mask check is trivially
   satisfied for every surviving cache column once `kv_length` (<= budget)
   is smaller than the true absolute position of the new query, which holds
   for the entire generation once eviction has kicked in at all (eviction
   only starts once active length > budget, by which point abs_pos > budget
   already). So no extra masking work is needed for decode steps. Prefill
   (a single forward over the full prompt) happens before any eviction, so
   its causal mask is the standard contiguous one and is unaffected.

2. RoPE correctness: rotation is baked into K/V vectors at the position
   they were computed, before caching. Physically reordering/dropping
   *columns* of an already-rotated cache does not change any vector's
   content, so surviving tokens keep correct relative-position semantics.
   We must, however, always pass the TRUE absolute position
   (`cache_position`/`position_ids`) for each new token — never derived
   from the (shrunk) cache's current length — so newly computed K/V get the
   correct rotation. `self.abs_pos` tracks this explicitly, independent of
   how many columns have been evicted.

3. Scoring is global (mean attention received, averaged over layers and
   heads), not per-layer — see cache/base_policy.py docstring for why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from transformers import DynamicCache

from .cache.base_policy import EvictionPolicy, SelectionContext, TokenStatus, BudgetSpec


SCORE_MODE = "sum"   # "sum" (as shipped for Phases 0-R/1/2) or "mean".
                      # See _score_step: sum carries a -0.717 position correlation.

QUANT_BITS = 8   # bit-width of the simulated quantized recoverable tier.
                  # Module-level so a sweep can vary it; 8 is the Phase 1/2 default.


def _quant_dequant(vec: torch.Tensor, bits: int | None = None) -> torch.Tensor:
    """Per-vector affine quantize+dequantize, simulating the precision loss
    of QEvict's "quantized recoverable tier". Returns a new tensor of the
    same dtype/shape.

    Note (found in Phase 2 diagnostics): at bits=8 this introduces only
    ~1% relative L2 error on realistic KV vectors, which is effectively
    lossless for the credential-retrieval task — a QUANT token reads just
    as well as a FULL one, making the FULL/QUANT distinction a no-op and
    any promotion signal unable to matter. `bits` is exposed so the
    precision at which the tier stops being free can be measured rather
    than assumed.
    """
    bits = QUANT_BITS if bits is None else bits
    levels = float(2 ** bits - 1)
    mn = vec.min()
    mx = vec.max()
    scale = (mx - mn) / levels
    if scale.abs() < 1e-8:
        return vec.clone()
    q = ((vec - mn) / scale).round().clamp(0, levels)
    return (q * scale + mn).to(vec.dtype)


@dataclass
class StepEvent:
    step: int
    promotions: list = field(default_factory=list)
    demotions: list = field(default_factory=list)
    evictions: list = field(default_factory=list)
    wall_time_s: float = 0.0
    materializes_attention_matrix: bool = True  # eager backend, always true here


@dataclass
class GenerationTrace:
    generated_text: str
    generated_token_ids: list
    events: list             # list[StepEvent]
    final_statuses: dict     # pos -> TokenStatus
    scores: dict             # pos -> float
    active_positions: list
    prompt_len: int
    total_wall_time_s: float


class CacheEngine:
    def __init__(self, model, tokenizer, policy: EvictionPolicy, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.policy = policy
        self.device = device

    def _capture_backup(self, cache, pos: int, backups: dict, col_idx: int):
        layer_kv = []
        for layer in cache.layers:
            k = layer.keys[:, :, col_idx, :].detach().to("cpu").clone()
            v = layer.values[:, :, col_idx, :].detach().to("cpu").clone()
            layer_kv.append((k, v))
        backups[pos] = layer_kv

    def _write_column(self, cache, col_idx: int, layer_kv_cpu):
        for layer, (k_cpu, v_cpu) in zip(cache.layers, layer_kv_cpu):
            layer.keys[:, :, col_idx, :] = k_cpu.to(layer.keys.device, dtype=layer.keys.dtype)
            layer.values[:, :, col_idx, :] = v_cpu.to(layer.values.device, dtype=layer.values.dtype)

    def _apply_changes(self, cache, active_positions: list, statuses: dict, backups: dict, new_statuses: dict):
        pos_to_col = {p: i for i, p in enumerate(active_positions)}

        for pos, new_status in new_statuses.items():
            if pos not in pos_to_col:
                continue
            old_status = statuses.get(pos, TokenStatus.FULL)
            col = pos_to_col[pos]
            if new_status == TokenStatus.QUANT and old_status != TokenStatus.QUANT:
                clean = backups[pos]
                quantized = [(_quant_dequant(k), _quant_dequant(v)) for (k, v) in clean]
                self._write_column(cache, col, quantized)
            elif new_status == TokenStatus.FULL and old_status != TokenStatus.FULL:
                self._write_column(cache, col, backups[pos])
            statuses[pos] = new_status

        keep_mask = [statuses.get(p, TokenStatus.FULL) != TokenStatus.EVICTED for p in active_positions]
        if not all(keep_mask):
            keep_idx = torch.tensor([i for i, k in enumerate(keep_mask) if k], dtype=torch.long)
            for layer in cache.layers:
                dev = layer.keys.device
                layer.keys = layer.keys.index_select(2, keep_idx.to(dev))
                layer.values = layer.values.index_select(2, keep_idx.to(dev))
            new_active = [p for p, k in zip(active_positions, keep_mask) if k]
            for p, k in zip(active_positions, keep_mask):
                if not k:
                    backups.pop(p, None)
                    statuses.pop(p, None)
            active_positions[:] = new_active

    @staticmethod
    def _epiphany_scores(hidden_states, new_positions, offset_in_chunk=0):
        """EpiKV-style 'epiphany score': how much a token's representation
        moves across layers during its own forward pass, read from hidden
        states with no attention matrix involved. Computed once per token at
        arrival, which is what makes it structurally independent of the
        accumulated-attention signal that drives eviction.
        """
        out = {}
        n_layers = len(hidden_states) - 1
        if n_layers < 1:
            return out
        for i, pos in enumerate(new_positions):
            idx = offset_in_chunk + i
            total = 0.0
            for l in range(1, len(hidden_states)):
                prev = hidden_states[l - 1][0, idx, :]
                cur = hidden_states[l][0, idx, :]
                denom = prev.norm().item() + 1e-6
                total += (cur - prev).norm().item() / denom
            out[pos] = total / n_layers
        return out

    def _run_select(self, policy, active_positions, statuses, scores, protected, oracle_important,
                     budget, is_prefill, step, recency_window, keep_sink, aux_signals=None):
        """Reserve a foundational sink + recency window (standard in every
        real eviction method — StreamingLLM's sink, H2O/SnapKV's local
        window) *uniformly across all arms*, unconditionally forced to
        FULL and excluded from the competitive budget. Without this, pure
        attention-score ranking can (and in an early smoke test, did) evict
        the tokens immediately preceding the answer, breaking local
        coherence for every arm including oracle_static — that's a harness
        bug, not a research finding, so it's fixed centrally here rather
        than left for each policy to reimplement.
        """
        foundational = set()
        if keep_sink and active_positions:
            foundational.add(active_positions[0])
        if recency_window > 0:
            foundational |= set(active_positions[-recency_window:])

        candidates = [p for p in active_positions if p not in foundational]
        reduced_budget = BudgetSpec(
            total_budget=max(0, budget.total_budget - len(foundational)),
            full_fraction=budget.full_fraction,
        )

        ctx = SelectionContext(
            scores=scores, statuses=statuses, protected=protected,
            oracle_important=oracle_important, active_positions=candidates,
            budget=reduced_budget, is_prefill=is_prefill, step=step,
            aux_signals=aux_signals or {},
        )
        policy.budget = reduced_budget
        result = policy.select(ctx)

        for p in foundational:
            if statuses.get(p, TokenStatus.FULL) != TokenStatus.FULL:
                result.new_statuses[p] = TokenStatus.FULL
                result.promotions.append(p)
        return result

    def _score_step(self, attentions, active_positions_before: list, new_positions: list, scores: dict,
                     attend_counts: dict | None = None):
        """Accumulate per-position attention.

        SCORE_MODE selects the reading. "sum" (as shipped for Phases 0-R/1/2)
        accumulates raw attention mass, which carries a positional artifact:
        in a ~1025-token prefill position 0 is attended by ~1025 queries and
        position 1000 by ~25, so early positions outrank later ones on
        longevity alone (measured Spearman(score, position) = -0.717).
        "mean" divides by the number of queries that have actually attended
        each position, removing that artifact. The two readings are
        near-uncorrelated (rho = +0.014) and share only 13.5% of the retained
        set, so this is a first-order choice, not a refinement.
        """
        stacked = torch.stack(attentions, dim=0)          # (L, B, H, Q, K)
        avg = stacked.mean(dim=(0, 2))                     # (B, Q, K) -> mean over layers & heads
        col_sum = avg[0].sum(dim=0)                         # (K,) sum over queries
        kv_positions = active_positions_before + new_positions
        assert col_sum.shape[0] == len(kv_positions), (col_sum.shape[0], len(kv_positions))
        col_sum_cpu = col_sum.detach().to("cpu")
        for p, s in zip(kv_positions, col_sum_cpu.tolist()):
            scores[p] = scores.get(p, 0.0) + s

        if attend_counts is not None:
            q_len = avg.shape[1]
            # every pre-existing position was attended by all q_len queries this call
            for p in active_positions_before:
                attend_counts[p] = attend_counts.get(p, 0) + q_len
            # a new position at offset i within the chunk is attended causally
            # by the (q_len - i) queries at or after it
            for i, p in enumerate(new_positions):
                attend_counts[p] = attend_counts.get(p, 0) + (q_len - i)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,        # (1, L0)
        max_new_tokens: int,
        protected: set,
        oracle_important: set,
        rebalance_every: int = 1,
        eos_token_id=None,
        recency_window: int = 8,
        keep_sink: bool = True,
    ) -> GenerationTrace:
        device = self.device
        prompt_ids = prompt_ids.to(device)
        L0 = prompt_ids.shape[1]

        cache = DynamicCache(config=self.model.config)
        statuses: dict = {}
        scores: dict = {}
        backups: dict = {}
        active_positions: list = []
        events: list = []

        t0 = time.time()

        # --- Prefill ---
        out = self.model(
            input_ids=prompt_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )
        new_positions = list(range(L0))
        self._score_step(out.attentions, [], new_positions, scores)
        active_positions.extend(new_positions)
        for i, p in enumerate(active_positions):
            statuses[p] = TokenStatus.FULL
            self._capture_backup(cache, p, backups, i)

        self.abs_pos = L0
        original_budget = self.policy.budget  # captured once — _run_select mutates
                                                # self.policy.budget in place each call, so
                                                # always pass this fixed reference, never the
                                                # (possibly already-reduced) live attribute.

        result = self._run_select(
            self.policy, list(active_positions), statuses, scores, protected, oracle_important,
            original_budget, True, 0, recency_window, keep_sink,
        )
        self._apply_changes(cache, active_positions, statuses, backups, result.new_statuses)
        events.append(StepEvent(step=0, promotions=result.promotions, demotions=result.demotions,
                                 evictions=result.evictions, wall_time_s=time.time() - t0))

        next_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated_ids = [next_token.item()]

        # --- Decode ---
        for step in range(1, max_new_tokens):
            step_t0 = time.time()
            pos_ids = torch.tensor([[self.abs_pos]], device=device)
            active_before = list(active_positions)

            out = self.model(
                input_ids=next_token,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                position_ids=pos_ids,
                cache_position=pos_ids[0],
            )
            new_pos = [self.abs_pos]
            self._score_step(out.attentions, active_before, new_pos, scores)
            active_positions.append(self.abs_pos)
            statuses[self.abs_pos] = TokenStatus.FULL
            self._capture_backup(cache, self.abs_pos, backups, len(active_positions) - 1)

            if step % rebalance_every == 0:
                result = self._run_select(
                    self.policy, list(active_positions), statuses, scores, protected, oracle_important,
                    original_budget, False, step, recency_window, keep_sink,
                )
                self._apply_changes(cache, active_positions, statuses, backups, result.new_statuses)
                events.append(StepEvent(step=step, promotions=result.promotions, demotions=result.demotions,
                                         evictions=result.evictions, wall_time_s=time.time() - step_t0))
            else:
                events.append(StepEvent(step=step, wall_time_s=time.time() - step_t0))

            self.abs_pos += 1
            next_logits = out.logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if eos_token_id is not None and tok_id == eos_token_id:
                break

        total_time = time.time() - t0
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return GenerationTrace(
            generated_text=text,
            generated_token_ids=generated_ids,
            events=events,
            final_statuses=dict(statuses),
            scores=dict(scores),
            active_positions=list(active_positions),
            prompt_len=L0,
            total_wall_time_s=total_time,
        )

    @torch.no_grad()
    def generate_multi_turn(
        self,
        turn_texts: list,                # turn_texts[0]: full initial chat text (system+dump+Q0+assistant-open);
                                          # turn_texts[i>0]: incremental "<|im_end|>...assistant\n" strings for turn i
        protected: set,
        oracle_important: set,
        max_answer_tokens: int = 12,
        rebalance_every: int = 1,
        eos_token_id=None,
        recency_window: int = 8,
        keep_sink: bool = True,
        capture_epiphany: bool = False,
        future_need: dict | None = None,
    ):
        """Multi-turn variant of `generate`: feeds each turn's question as a
        teacher-forced chunk (safe as a multi-token forward — see note
        below — since these tokens are never sampled), then generates that
        turn's answer one token at a time exactly like `generate`'s decode
        loop. Eviction/promotion is rebalanced after every chunk and every
        generated token, so a credential that goes quiet for several turns
        and then gets asked about is exactly the regime where a
        recoverable-tiering policy's promotion mechanism has something to
        prove or fail at.

        Extra correctness note (beyond engine.py's module docstring): a
        multi-token forward for a chunk of *brand-new* (never-cached)
        tokens is safe even after eviction has compacted the old prefix.
        transformers' causal mask places new queries as the trailing
        `q_length` columns of the kv axis and compares them against kv
        columns by that same relative layout — so causality among the new
        chunk's own tokens (token 3 can't see token 5) is enforced
        correctly via true physical adjacency, independent of whatever
        column-index/absolute-position mismatch exists in the (always
        fully-visible, per the module docstring's argument) compacted old
        prefix. Only single-token forwards were required for *generated*
        tokens, and that's simply because generation is sequential anyway.
        """
        device = self.device
        cache = DynamicCache(config=self.model.config)
        statuses: dict = {}
        scores: dict = {}
        backups: dict = {}
        active_positions: list = []
        events: list = []
        turn_answers: list = []
        self.first_answer_tokens: list = []

        t0 = time.time()
        self.abs_pos = 0
        original_budget = self.policy.budget
        epiphany: dict = {}
        attend_counts: dict = {}
        aux_signals = {"epiphany": epiphany, "future_need": future_need or {}}

        def ranking_scores():
            """What the policies rank on. Under SCORE_MODE='mean' this divides
            accumulated attention by the number of queries that attended each
            position, removing the longevity artifact."""
            if SCORE_MODE != "mean":
                return scores
            return {p: v / max(1, attend_counts.get(p, 1)) for p, v in scores.items()}

        def feed_chunk(text: str, add_special_tokens: bool):
            nonlocal active_positions
            ids = self.tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt")["input_ids"].to(device)
            chunk_len = ids.shape[1]
            pos_ids = torch.arange(self.abs_pos, self.abs_pos + chunk_len, device=device).unsqueeze(0)
            active_before = list(active_positions)
            out = self.model(
                input_ids=ids, past_key_values=cache, use_cache=True,
                output_attentions=True, output_hidden_states=capture_epiphany,
                position_ids=pos_ids, cache_position=pos_ids[0],
            )
            new_pos = list(range(self.abs_pos, self.abs_pos + chunk_len))
            self._score_step(out.attentions, active_before, new_pos, scores, attend_counts)
            if capture_epiphany:
                epiphany.update(self._epiphany_scores(out.hidden_states, new_pos))
            active_positions.extend(new_pos)
            for i, p in enumerate(new_pos):
                statuses[p] = TokenStatus.FULL
                self._capture_backup(cache, p, backups, len(active_before) + i)
            self.abs_pos += chunk_len
            return out

        def rebalance(step_idx, t_start):
            result = self._run_select(
                self.policy, list(active_positions), statuses, ranking_scores(), protected, oracle_important,
                original_budget, False, step_idx, recency_window, keep_sink, aux_signals=aux_signals,
            )
            self._apply_changes(cache, active_positions, statuses, backups, result.new_statuses)
            events.append(StepEvent(step=step_idx, promotions=result.promotions, demotions=result.demotions,
                                     evictions=result.evictions, wall_time_s=time.time() - t_start))

        global_step = 0
        turn_start_steps = []
        for turn_idx, turn_text in enumerate(turn_texts):
            turn_start_steps.append(global_step)
            step_t0 = time.time()
            out = feed_chunk(turn_text, add_special_tokens=(turn_idx == 0))
            rebalance(global_step, step_t0)
            global_step += 1

            next_logits = out.logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            answer_ids = [next_token.item()]
            self.first_answer_tokens.append(answer_ids[0])

            for _ in range(1, max_answer_tokens):
                if eos_token_id is not None and answer_ids[-1] == eos_token_id:
                    break
                step_t0 = time.time()
                pos_ids = torch.tensor([[self.abs_pos]], device=device)
                active_before = list(active_positions)
                out = self.model(
                    input_ids=next_token, past_key_values=cache, use_cache=True,
                    output_attentions=True, output_hidden_states=capture_epiphany,
                    position_ids=pos_ids, cache_position=pos_ids[0],
                )
                new_pos = [self.abs_pos]
                self._score_step(out.attentions, active_before, new_pos, scores, attend_counts)
                if capture_epiphany:
                    epiphany.update(self._epiphany_scores(out.hidden_states, new_pos))
                active_positions.append(self.abs_pos)
                statuses[self.abs_pos] = TokenStatus.FULL
                self._capture_backup(cache, self.abs_pos, backups, len(active_positions) - 1)

                if global_step % rebalance_every == 0:
                    rebalance(global_step, step_t0)
                else:
                    events.append(StepEvent(step=global_step, wall_time_s=time.time() - step_t0))
                global_step += 1

                self.abs_pos += 1
                next_logits = out.logits[:, -1, :]
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                tok_id = next_token.item()
                answer_ids.append(tok_id)

            turn_answers.append(self.tokenizer.decode(answer_ids, skip_special_tokens=True))

        total_time = time.time() - t0
        return GenerationTrace(
            generated_text="\n".join(turn_answers),
            generated_token_ids=[],
            events=events,
            final_statuses=dict(statuses),
            scores=dict(scores),
            active_positions=list(active_positions),
            prompt_len=0,
            total_wall_time_s=total_time,
        ), turn_answers, turn_start_steps
