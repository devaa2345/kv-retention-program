"""ChunkKV adapter for the project's budget accounting.

STATUS (2026-09-19): `kvpress.ChunkKVPress` EXISTS in the pinned kvpress 0.5.4 (NVIDIA's
implementation of arXiv:2502.00299). No eviction algorithm is written here. This module only
adapts it to the project's budget accounting, and states plainly where the two disagree:

  * ChunkKV keeps WHOLE chunks (default length 20), so it cannot retain an arbitrary B. The
    harness invariant is "realised budget == B, per instance, asserted" (tolerance 1 token).
    Here the realised count is the LARGEST chunk-multiple <= B (the trailing partial chunk is
    always kept, because it lies inside the mandatory window), so ChunkKV spends between B-19
    and B tokens: it is slightly UNDER-budget, never over. That handicaps ChunkKV by <= 19 of B
    tokens (<= 3.3% at B = 584) and is recorded per row as `deficit`.
  * Mandatory floors (8 sink + 64 window) are pinned exactly as for every other method arm, by
    wrapping the underlying SnapKV scorer. At chunk granularity the sink chunk and the chunks
    overlapping the window are forced, which can spend up to ~19 tokens more on the floors than
    the 72 nominal; that is inherent to the method and is recorded, not corrected.
  * Underlying scorer: SnapKV (the ChunkKV paper scores with SnapKV-style observation-window
    attention). Chunk length 20 (kvpress default; the paper reports 10-20 as its sweet spot).

Admission status: kvpress's ChunkKV is third-party code, so the in-house-reimplementation rule
(reproduction gate before entering a table) does not apply. The roster gates G1 (published-number
reproduction), G2 (permutation) and G3 (oracle overlap) have NOT been run for ChunkKV, and G1 is
not on disk for any method (Paper 2 TODO-T5). ChunkKV results are therefore PROVISIONAL.
"""
from harness import methods
from kvpress import ChunkKVPress, SnapKVPress

CHUNK = 20


class ChunkKVBudget(ChunkKVPress):
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        ratio = self.press.compression_ratio
        if ratio == 0:
            return keys, values
        kv_len = keys.shape[2]
        L = self.chunk_length
        B = int(kv_len * (1 - ratio))                 # the budget the harness asked for
        rem, ncomp = kv_len % L, kv_len // L
        ntot = ncomp + (1 if rem else 0)
        n_kept = (B - rem) // L + 1 if rem else B // L
        n_kept = max(1, min(ntot, n_kept))
        self.press.compression_ratio = 1.0 - (n_kept + 0.5) / ntot   # int(ntot*(1-r)) == n_kept
        try:
            return super().compress(module, hidden_states, keys, values, attentions, kwargs)
        finally:
            self.press.compression_ratio = ratio


def build_chunkkv(ratio: float, n_ctx: int, n_sink: int = 8, n_window: int = 64):
    inner = methods.make_floor_constrained(SnapKVPress(compression_ratio=ratio),
                                           n_ctx, n_sink, n_window)
    p = ChunkKVBudget(press=inner, chunk_length=CHUNK)
    p.compression_ratio = ratio
    return p


def expected_kept(n_ctx: int, B: int) -> int:
    rem, ncomp = n_ctx % CHUNK, n_ctx // CHUNK
    ntot = ncomp + (1 if rem else 0)
    n_kept = (B - rem) // CHUNK + 1 if rem else B // CHUNK
    n_kept = max(1, min(ntot, n_kept))
    return (n_kept - 1) * CHUNK + rem if rem else n_kept * CHUNK
