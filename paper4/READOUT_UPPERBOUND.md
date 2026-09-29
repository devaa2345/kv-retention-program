# Attention read-out — diagnostic upper bound (extraction, not a method)

Query format confirmed first: `ledger_c.py`'s query template is fixed and always asks for the
FULL record (all fields, "Reproduce record Rxxx exactly, in full... Answer with the fields only"),
at every c_tag including c=40 — never a partial-field query. Read-out therefore = emit every
field of the top-attended record, in schema order, joined by `|`, matching the grader's format
exactly. If the top-attended record IS the queried one, this is definitionally correct
(identical content); if not, it is definitionally wrong. Read-out accuracy = pooled max-attention
match rate, weighted by each arm's actual wrong/correct case counts — no generation involved,
pure extraction, hence "diagnostic upper bound," not a decoding method.

## Item 1 — full_cache failure check (M2, c=40, n=50, 200 queries)

full_cache accuracy: **0.895** (179/200). Only 21 wrong.

- Max-attention = queried record on full_cache's wrong cases: **16/20 = 80%** (one case excluded,
  no payable span) — lower than floor's 98% and oracle's 89.5%, a real difference: full_cache
  has a meaningfully larger minority (20%) of failures where attention itself is misdirected.
- Invented (non-gold) 6-digit numbers among wrong cases: **7/21 = 33.3%.**

**This answers the framing question directly: the conversion failure exists WITHOUT
compression.** Even with the full, uncompressed context available, the model still gets 21/200
wrong, and in 80% of those it was already attending to the right record — so grounding/copy-
fidelity is not an artifact of compression, it is present in the base model's own decoding.

## Item 2 — read-out vs actual generation, all three arms

| arm | actual accuracy | wrong-case hit rate | correct-case hit rate | read-out (pooled) | recoverable gap |
|---|---|---|---|---|---|
| floor_pos | 0.175 (n=480q) | 49/50 = 98.0% | 79/81 = 97.5% | **0.291** | **+0.116** |
| oracle_causal | 0.695 (n=200q) | 51/57 = 89.5% | 29/29 = 100% (n=30 sample) | **0.968** | **+0.273** |
| full_cache | 0.895 (n=200q) | 16/20 = 80.0% | 47/47 = 100% (n=50 sample) | **0.979** | **+0.084** |

All three read-out figures are extraction-only upper bounds computed from the pooled
wrong/correct max-attention match rates and each arm's own actual wrong/correct case counts —
labeled explicitly as such, not a claim that a real decoding method achieves them.

## Stopping and reporting, per instruction

**The gap is large on all three arms (+0.084 to +0.273)** — this is the condition for moving to
the soft-intervention design (logit bonus λ, selection/held-out split, ceiling, confound check,
false-positive table). Not designed or run here — stopping to report per instruction, awaiting
confirmation to proceed to that design.
