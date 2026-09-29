"""The ONE reading of floor_pos, asserted against real keep-sets.

PREREG_P4_S3.md section 3.2 fixes floor_pos as: retain exactly B = C + n_sink + n_window
tokens = the first n_sink, plus the most recent (B - n_sink) contiguous tokens. This module
is the single place that reading is expressed in code for Paper 4, so prereg and harness
cannot drift apart; the Stage 3 runner calls `assert_floor_pos` on every floor_pos row.
"""
from __future__ import annotations

import numpy as np

N_SINK, N_WINDOW = 8, 64


def expected_keep(n_ctx: int, C: int, n_sink: int = N_SINK, n_window: int = N_WINDOW):
    B = C + n_sink + n_window
    return np.concatenate([np.arange(n_sink), np.arange(n_ctx - (B - n_sink), n_ctx)])


def assert_floor_pos(kept, n_ctx: int, C: int) -> None:
    exp = expected_keep(n_ctx, C)
    got = np.asarray(sorted(kept))
    if got.size != exp.size or not np.array_equal(got, exp):
        raise AssertionError(
            f"floor_pos keep-set does not match the registered reading: kept {got.size} tokens, "
            f"expected {exp.size} (= B) as first {N_SINK} + last {exp.size - N_SINK}")
