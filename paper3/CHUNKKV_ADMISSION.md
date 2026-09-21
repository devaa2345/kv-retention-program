# ChunkKV admission gates (2026-09-21)

**Procedure.** Not in PREREG_P3.md (it contains no admission procedure). The gates are PREREG_P2_v2 §7, and the four admitted
methods were run through `paper2/n4a_admission_gates.py` (G2 and G3 only; a G1 reproduction was never run for them, see below).
`paper3/chunkkv_admission.py` reuses that file's instance construction (n=24 LEDGER instances, same seed key), random-arm reference,
paired bootstrap (20,000), thresholds and decision code unchanged. Budgets C in {32, 128, 512}, Qwen2.5-3B and Llama-3.2-3B, pinned WSL
environment, single process. ChunkKV: `ChunkKVBudget` over the floor-constrained SnapKV scorer, chunk length 20 (the Paper 3 grid setting).
Two ChunkKV-specific adaptations, forced by the method: the G2 permutation acts on the SnapKV token-score vector inside ChunkKV, before
the floors are pinned (as n4a permutes a method's own score before the floors); and the retained set is recomputed from the same scores
with the same chunk logic and recorded per (layer, head) (ChunkKV selects chunks per layer, so all heads of a layer share a set).

Results: `out/chunkkv_admission_Qwen2_5-3B-Instruct.json`, `out/chunkkv_admission_Llama-3_2-3B-Instruct.json`, log `out/chunkkv_admission.log`.

## G1 (reproduction): not run

Never run for SnapKV, AdaKV, ExpectedAttention or KeyDiff either (only LU-KV, on the second machine, `G1_RESULTS.md`). ChunkKV's published
setting (arXiv:2502.00299, Table 6) is LLaMA-3-8B-Instruct, Mistral-7B-Instruct-v0.3 and Qwen2-7B-Instruct at 10/20/30% compression,
chunk size 10, three runs averaged, no variance reported; the tables report gaps to FullKV. No published number exists for a model that fits
a 12 GB card in bf16, and per-task numbers are in a table not read. Not matched, so not reproduced.

## G2 (permutation): FAIL on both models

The n4a rule needs all three parts. (a) permuted accuracy at most the random arm + 0.02 at every budget; (b) |rho| < 0.10 between original and permuted retained sets at every budget; (c) original beats its own permutation (CI lower bound > 0) at some budget.

| model | C | original | permuted | random arm | original minus permuted [CI] | rho |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | 32 | 0.0417 | 0.0417 | 0.0000 | +0.0000 [0, 0] | 0.9735 |
| | 128 | 0.0625 | 0.0417 | 0.0000 | +0.0208 [−0.0208, +0.0625] | 0.1719 |
| | 512 | 0.2083 | 0.0625 | 0.0000 | +0.1458 [+0.0833, +0.2083] | −0.0030 |
| Llama-3.2-3B | 32 | 0.0104 | 0.0104 | 0.0000 | +0.0000 [0, 0] | 0.9735 |
| | 128 | 0.0312 | 0.0208 | 0.0000 | +0.0104 [0, +0.0312] | 0.1664 |
| | 512 | 0.1667 | 0.0625 | 0.0000 | +0.1042 [+0.0417, +0.1667] | 0.0170 |

(a) fails (permuted 0.0417 at C=32 and 0.0625 at C=512 on Qwen exceed 0.02; Llama 0.0625 at C=512). (b) fails (rho 0.97 at C=32, about 0.17 at C=128).
(c) holds at C=512 only. All four admitted methods passed (a), (b) and (c) (Paper 2 `gates/nvidia/n4a_admission_*.json`).
Two structural reasons, both properties of the method, not of the adaptation: at C=32 the floors at chunk granularity (sink chunk plus the chunks
overlapping the 64-token window) consume the whole budget, so the permuted and original keep-sets coincide; and whole-chunk selection keeps
contiguous spans, so even a randomly chosen set of chunks completes records that a random token set does not, which is why permuted
ChunkKV stays above the token-random arm. The criterion was not modified to accommodate this.

## G3 (oracle overlap): PASS on both models

Per-(layer, head) overlap with `oracle_causal` inside the compressible region, positive at every budget and increasing with C.

| model | C=32 | C=128 | C=512 |
|---|---|---|---|
| Qwen2.5-3B | 0.2871 | 0.3814 | 0.5607 |
| Llama-3.2-3B | 0.1881 | 0.4983 | 0.5785 |

For comparison (Paper 2 Appendix E, C=32/128/512): Qwen SnapKV 0.2600/0.3382/0.5156, AdaKV 0.2516/0.3283/0.5029, Llama SnapKV 0.1653/0.3595/0.4643.

## Verdict

G1 not run, G2 fail, G3 pass. ChunkKV is **not admitted**: it stays out of the admitted roster and is excluded with the failing gate (G2), as LU-KV is in Paper 2.
