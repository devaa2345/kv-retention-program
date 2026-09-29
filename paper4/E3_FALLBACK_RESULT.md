# E3 — expected_attn fallback intervention, result

Scope: `expected_attn` only (`snapkv`'s 25% max-attention hit rate ruled it out separately,
`out/_e3_check_maxrid_102_results.json` / commit `be33e06`).

## Ceiling — corrected before reporting

`ledger_c.score_instance` averages 4 query-variants per instance, so one fixed query moves
overall accuracy by `1/(n_instances*4)`, not `1/n_instances`. An initial pass used the wrong
denominator (`0.0375 + 29/120 = 0.2792`, appearing to exceed `floor_pos` 0.175) — caught before
reporting. **Corrected: `0.0375 + 29/480 = 0.0979`, still 0.077 short of `floor_pos` (0.175) even
at the unreachable 100%-conversion ceiling.** This changes the framing from "could beat floor" to
"can improve the arm's own accuracy, not close the gap to floor."

## Split and result

54 expected_attn fabrication cases, alternating-index 27/27 selection/validation split. Design
fixed on selection half: fallback fires unconditionally (no confidence threshold needed — the
arm-level ceiling was already known before any split).

- **Held-out validation (the number that counts):** 13/27 = 48.1% of held-out fabrication cases
  converted to correct, Wilson 95% CI **[30.7%, 66.0%]** — clearly excludes zero, consistent with
  the known population rate (29/54 = 53.7%).
- **False positives:** checked against the 10 existing `expected_attn` completion-matched
  correct-case pairs (original Task 3 pool, no new GPU work — `max_rid` already captured for
  these). **2/10 = 20%** would be broken by blind substitution, Wilson 95% CI [5.7%, 51.0%] (wide,
  small n).
- **Net effect:** expected_attn's baseline accuracy is only 3.75%, so very few correct cases exist
  to break (~18 of 480 queries) against a much larger fixable-wrong population (29 of 54 known
  fabrication cases alone). Rough accounting: +29 gained vs ~3.6 expected broken -> **net +25.4
  queries, +0.053 accuracy points for the expected_attn arm specifically.**

## Verdict: REAL, bounded and arm-specific

Held-out conversion CI excludes zero AND net accuracy improves (not just gross fabrication-case
accuracy) — both conditions met. This is the first genuine net-positive intervention of the whole
program tonight. But it raises `expected_attn`'s own standalone accuracy (0.0375 -> ~0.090), not
the paper's central floor_pos-beating claim — the corrected ceiling confirms it cannot close that
gap even in principle. `snapkv` stays out of scope and confirmed dead. The FP sample (n=10) is
small; a larger correct-case sample would tighten that estimate before this is treated as fully
settled, but the qualitative verdict (REAL, net-positive, floor-insufficient) is not expected to
flip.
