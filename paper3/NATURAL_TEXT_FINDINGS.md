# Natural-text evidence ladder — Experiment C findings (C = 256, 512; both models; n = 100)

Dataset `data/natural/nat_v1.jsonl` (sha256 in `nat_v1.jsonl.sha256`): real public-domain prose
with systematically generated fact sentences (**not fully found data**), L = 4096 ± 32 per
tokenizer, H = 4, D = 32. Plan and gate frozen in `NATURAL_LADDER_PLAN.md` before any GPU run.
C = 1024 was **not run** (stopped by request; the 6-hour budget projected at ~11 h).
Full tables: `out/natC_report.md` (G_m, I, R with paired bootstrap CIs) and
`out/NATURAL_VS_LEDGER.md` (side-by-side with LEDGER).

Marginal cells, flagged † in every table: **M3 level 1** (full_cache anchor 0.968 against a 0.97
ceiling) and **every C = 256 cell** (perfect-reader floor ceiling 0.089).
Scorer: v2 (first answer block only) after the M3 audit; M3 anchors 0.968 / 0.845 / 0.800.

## What is established

1. **Methods lose to `floor_pos` on natural text in every non-marginal cell.** 48 method cells
   (2 models × 3 levels × 2 budgets × 4 methods): the upper 95% bound of R = A_method / A_floor is
   ≤ 0.87 in all of them and G_m is negative in all of them (max −0.077). This reproduces Paper 2's
   G_m ≤ 0 finding outside synthetic data. **Limit on how much it says:** 24 of the 48 cells are
   vacuous by the frozen rule (every compressed arm ≤ 0.02): 20 of the 24 M2 cells and 4 of the
   24 M3 cells. In those cells "loses to the floor"
   means "scores about zero", and says nothing about the ordering among methods.
2. **I = 0 at C = 512 (and at C = 256).** `oracle_prescient` equals `oracle_causal` in every cell on
   both models, I = 0.000 [0.000, 0.000]. The budget exceeds the total payable cost of the four
   candidates (66 / 111 / 184 tokens on M2; 66 / 103 / 164 on M3 at levels 1 / 3 / 5), which is
   the structural transition Paper 2 predicts. It confirms the transition; it is not a finding
   about eviction methods.
3. **No crossover is observable.** The natural fact template has a minimum cost of c ≈ 17 tokens;
   there is no natural-text analogue of LEDGER's c = 1 control. Methods are already below the floor
   at the cheapest level tested, so this experiment cannot show the reversal disappearing — only
   that methods already lose at the cost level tested. The level 1 → 3 → 5 trend is weak (M3
   AdaKV 0.60 → 0.44 → 0.39; M2 is flat inside the vacuous region).

## Natural text versus LEDGER at matched cost — **a model-specific result, not a general one**

Matched pair: natural level 1 (c ≈ 17) against LEDGER c ≈ 19, at C = 512 (same absolute budget) and
at natural 512 / LEDGER 256 (same 12.5% of context).

| model | comparison | natural R | LEDGER R (same C) | LEDGER R (same fraction) |
|---|---|---|---|---|
| M2 | SnapKV / AdaKV | 0.13 / 0.18 | 0.77 / 0.77 | 0.43 / 0.61 |
| M2 | KeyDiff | 0.03 | 0.47 | 0.50 |
| M2 | ExpectedAttention | 0.17 | 0.32 | 0.25 |
| M3 † | SnapKV / AdaKV | 0.32 / 0.60 | 0.17 / 0.47 | 0.13 / 0.30 |
| M3 † | KeyDiff / ExpectedAttention | 0.15 / 0.08 | 0.35 / 0.24 | 0.22 / 0.10 |

- **M2: natural text is harsher than LEDGER at matched cost**, by a wide margin for SnapKV, AdaKV
  and KeyDiff (natural 0.03–0.18 against LEDGER 0.43–0.77, intervals not overlapping at the same
  budget for SnapKV/AdaKV/KeyDiff). Same picture at c ≈ 41–46 against LEDGER c ≈ 40 (0.04–0.10 vs
  0.10–0.31, same budget), with the caveat that those natural cells are vacuous.
- **M3: natural text is not harsher.** M3's natural ratios (0.08–0.60) overlap LEDGER's (0.17–0.47);
  SnapKV and AdaKV are higher on natural text, ExpectedAttention and KeyDiff lower. M3 level 1 is a
  marginal cell.
- So the earlier statement that natural text is harsher across the board is **only supported for
  M2**. It is reportable as an M2 observation. It is also confounded three ways: different fact
  template (prose sentence vs record line), different context length (4096 vs 2048), and a
  different floor (natural floor 0.15 against LEDGER's 0.07 on M2 at matching fraction, 0.30 on M3
  at the same C). It should not be written up as a property of natural text until a run isolates
  those factors.

## Not done / open

- C = 1024 (stopped by request). Level-3 c ≈ 26–28 has no LEDGER partner at matched cost.
- No compressed-arm validity check of ExpectedAttention/KeyDiff on prose beyond budget parity
  (asserted on the first two instances per arm and budget).
- `null`/`random` used a 32-token generation cap (others 128); they score 0.000 everywhere, so the
  cap cannot have hidden a non-zero score, but it is a deviation from a uniform cap.
