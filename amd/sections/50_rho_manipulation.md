## Phase 2 manipulation check (§7) at n=100

§7 requires ρ(eviction score, promotion score) to be +1.000 for P1 and near zero for the rest.
An earlier n=12 estimate was too small to act on; this is n=100 prompts per signal with
bootstrap CIs over prompts. Both the **original** signal definitions (those in force when the
completed Phase 2 data was generated) and the **corrected** ones are reported.

| id | signal | ρ original | 95% CI | ρ corrected | 95% CI | target | original verdict |
|---|---|---|---|---|---|---|---|
| P1 | attention | +1.0000 | [+1.0000, +1.0000] | +1.0000 | [+1.0000, +1.0000] | +1.000 | pass |
| P2 | epiphany | +0.0964 | [+0.0887, +0.1041] | +0.0964 | [+0.0887, +0.1041] | ~0 | pass |
| P3 | random | +0.0958 | [+0.0929, +0.0989] | -0.0560 | [-0.0592, -0.0526] | ~0 | pass |
| P4 | roundrobin | +0.4974 | [+0.4944, +0.5006] | -0.0218 | [-0.0250, -0.0186] | ~0 | **FAIL** |
| P5 | oracle | +0.8489 | [+0.8251, +0.8707] | -0.0781 | [-0.0971, -0.0604] | ~0 | **FAIL** |

**P1 is exactly +1.0000 with a zero-width CI**, as it must be: the P1 promotion score *is* the
eviction score by construction. That the check returns exactly 1.000 rather than approximately
1.000 is itself evidence the probe measures the quantity the policy actually uses.

### Two signals failed, and the check is what caught them

P4 (round-robin) at **+0.4974** and P5 (oracle) at **+0.8489** are not marginal — their CIs
exclude zero by a wide margin at n=100. The causes are specific and were found by inspection
once the check flagged them:

- **P5** assigned `1e9` to credential positions the oracle must promote and then *fell back to
  the eviction score* for every other position. An "oracle" signal that reuses the eviction
  ranking for 95% of positions is correlated with it by construction.
- **P4** rotated over raw position index. Position correlates with accumulated attention, so a
  positional rotation inherits that correlation.

Both are defects in this implementation of the signals, not in the spec. The corrected
definitions give P5 membership-only scores (1.0 / 0.0, no fallback) and rotate P4 over a fixed
seed-stable permutation instead of position. Under those, **every signal passes**: P4 −0.0218
and P5 −0.0781, with P1 still exactly +1.0000.

### Consequence for Phase 2, stated plainly

**The originally-completed Phase 2 arms are not a valid test of orthogonal promotion.** Two of
the five signals were substantially correlated with the eviction signal, which is precisely the
condition §7's manipulation check exists to rule out. Reporting those contrasts as though the
manipulation had held would be wrong.

Phase 2 has therefore been **re-run in full with the corrected signals**, and the corrected run
is reported as primary. The original run is retained and reported alongside, because the
difference between them is the measurable cost of the contamination — and because a
manipulation check that fires, is diagnosed, and is acted on is a stronger result than one that
quietly passes.

> Note on scope: the correction changes only the `random`, `roundrobin` and `oracle` promotion
> branches. The `attention` path — which is what Phase 1's tiered arms use — is untouched, and
> that was verified bitwise on 8 cells spanning all arms and budgets. No Phase 1 result is
> affected.

