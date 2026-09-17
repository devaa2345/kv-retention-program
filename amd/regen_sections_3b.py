"""Generate the second-model section of FINDINGS.md from stored 3B results."""
import json, os
J=lambda f: json.load(open(f)) if os.path.exists(f) else None
a=J("results3b/analysis.json"); sw=J("results3b/sweep_table.json")
g=J("results3b/calibration_gates.json"); ph=J("results3b/posthoc.json")
sw15=J("results/sweep_table.json")
if not a: raise SystemExit("no 3B analysis yet")
L=["## Second model — Qwen2.5-3B-Instruct\n",
"`meta-llama/Llama-3.2-3B-Instruct` was the first choice but is gated (HTTP 401,",
"`GatedRepoError`, no token configured), so the second model is Qwen2.5-3B-Instruct. Same family",
"as the 1.5B, which makes the comparison isolate **scale** rather than confounding it with",
"architecture and tokenizer — at the cost of not testing across architectures. 36 layers,",
"16 attention / 2 KV heads, head_dim 128, hidden 2048, 3.09B params, bfloat16.\n",
"**The KV geometry is identical to the 1.5B** (2 KV heads × 128 = 256-element quantizer groups),",
"so the quantizer is directly comparable and cannot itself explain any difference.\n",
"### Calibration: the competence gate cannot be matched\n",
"The brief asked that N and budget be tuned to a comparable competence gate. **They cannot be.**",
"Across 8 configurations the 3B never fell below ~0.94:\n",
"| lever | configurations tried | `full_cache_ref` |","|---|---|---|",
"| N (credentials) | 6 / 9 / 12 / 16 at ctx ~1030–1250 | 0.9861 / 1.0000 / 1.0000 / 0.9896 |",
"| context + distractors | ctx 2635 / 5086 / 8061 / 8155 | 0.9861 / 0.8194 / 1.0000 / 0.9583 |",
"| re-test of the one promising point | ctx 5086, 30 fresh seeds | **0.9833** (the 0.8194 was a 12-seed fluke; pooled n=42 → 0.9365) |","",
"The 1.5B's bottleneck was hex **copy fidelity** — 57–64% of its failures were near-misses. The",
"3B has largely solved that, so adding credentials only adds more of a task it can already do,",
"and lengthening context does not degrade it monotonically. **The ceiling difference is therefore",
"the scale effect itself, not a nuisance to be tuned away**, and it is reported as a measured",
"covariate. The matched-task configuration (identical to the 1.5B: N=6, 20 distractors, filler 7,",
"pad 26, budgets 154/257/514) was used so that scale is the only manipulated variable.\n"]
if g:
    L+=["### Gates, all re-run on the 3B\n",
        "| gate | 3B | 1.5B |","|---|---|---|",
        f"| G1 context length | median {g['G1']['median_len']}, range {g['G1']['range']} | median 1030.5, range [1017, 1041] |",
        f"| G2 competence | **{g['G2']['full_cache_mean_acc']:.4f}** | 0.7167 |",
        f"| G3 budget arithmetic | pass; budget 51 refused; `full_cache_ref` invariant at {list(g['G3']['full_cache_by_budget'].values())[0]:.4f} | pass; invariant at 0.8125 |",
        "| G4 quantizer integrity | pass | pass |",
        f"| G5 dormancy | {g['G5']['frac_weak']:.0%} weak, {g['G5']['frac_strict_8step']:.0%} strict | 100% / 0% |","",
        "3B quantizer error on real cached K/V: 8-bit 1.16%/0.89%, 4-bit 19.55%/14.87%,",
        "2-bit 96.68%/74.31% — within ~2% of the 1.5B and of the spec reference at every width.",
        "**Reconstruction error is essentially model-independent.**\n"]
L+=["### Phase 1 at 8-bit, iso-token\n",
    "| budget | arm1 | arm2 | arm3 | arm4 | arm5 | arm6 | n |","|---|---|---|---|---|---|---|---|"]
for b in ("154","257","514"):
    if b in a["phase1"]:
        e=a["phase1"][b]
        L.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %d |"%(b,e['arm1'],e['arm2'],
                 e['arm3'],e['arm4'],e['arm5'],e['arm6'],e['n']))
L+=["","| budget | protection | tiering | interaction | 95% CI | n differ |","|---|---|---|---|---|---|"]
for b in ("154","257","514"):
    if b in a.get("interaction",{}):
        e=a["interaction"][b]
        L.append("| %s | %+.4f | %+.4f | **%+.4f** | [%+.4f, %+.4f] | %d |"%(b,e['protection'],
                 e['tiering'],e['interaction'],e['ci'][0],e['ci'][1],e['n_differ']))
L+=["","**The 8-bit interaction is ≈0 on the second model too**, reproducing both the 1.5B's 8-bit",
    "result (−0.0011 at 257, −0.0056 at 514) and the spec's reference (+0.002, −0.002). The",
    "[GAP-W] finding — that the apparent interaction disagreement was an unstated bit-width, not a",
    "mechanism difference — therefore replicates across scale.\n"]
if sw:
    L+=["### Bit-width sweep — the headline result\n",
        "| bits | n | mean acc | empty/EOS | repetitive | degenerate | clean | mixed | recon err | levels/limit | viol |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in ("8","7","6","5","4","3"):
        if w not in sw: continue
        e=sw[w]; st="**" if e["mean_acc"]<0.15 else ""
        L.append("| %s%s%s | %d | %s%.4f%s | %.3f | %.3f | %d | %d | %d | %.2f%% | %s/%s | %s |"%(
            st,w,st,e['n'],st,e['mean_acc'],st,e['empty'],e['repetitive'],e['degenerate'],
            e['clean'],e['mixed'],e['recon_err_pct'] or float('nan'),e['max_levels'],e['limit'],e['violations']))
    L+=["","### The phenomenon reproduces; the widths do not\n",
        "| model | 8 | 7 | 6 | 5 | 4 | 3 |","|---|---|---|---|---|---|---|"]
    if sw15:
        L.append("| **1.5B** | %.4f | %.4f | %.4f | **%.4f** ↓ | **%.4f** ↑ | **%.4f** ↓ |"%tuple(
            sw15[w]["mean_acc"] for w in ("8","7","6","5","4","3")))
    L.append("| **3B** | %.4f | %.4f | %.4f | %.4f | **%.4f** ↓ | **%.4f** ↑ |"%tuple(
        sw[w]["mean_acc"] for w in ("8","7","6","5","4","3")))
    L+=["","**Non-monotone accuracy under strictly monotone error reproduces on an independent",
        "model.** The 3B is flat from 8 through 5 bits, drops 6.7× at 4-bit, then *recovers* at",
        "3-bit under twice the reconstruction error — the same qualitative signature as the 1.5B.",
        "",
        "**But the widths do not transfer at all.** The 1.5B collapses at 5-bit, where the 3B is",
        "unaffected (0.2467); the 1.5B *recovers* at 4-bit, which is precisely where the 3B fails.",
        "So non-monotonicity is a property of the phenomenon, while *which* widths collapse is",
        "model-specific. This strengthens §9's caution: the ordering is not merely non-monotone, it",
        "is not even stable across two models of the same family sharing a tokenizer and KV",
        "geometry.",
        "",
        "The quantizer is cleared independently on both models: identical group structure, error",
        "tables agreeing within ~2%, zero level violations at every width, and strictly monotone",
        "error on tensors intercepted during live generation.\n"]
if a.get("replication"):
    L+=["### Fresh-seed replication of the collapse widths\n",
        "| bits | seeds 0–49 | seeds 50–99 (unseen) | difference | n |","|---|---|---|---|---|"]
    for w in ("5","4","3"):
        if w in a["replication"]:
            e=a["replication"][w]
            L.append("| %s | %.4f | %.4f | %+.4f | %d |"%(w,e['orig'],e['fresh'],e['fresh']-e['orig'],e['n_fresh']))
    L+=["","The 4-bit collapse and the 3-bit recovery both hold on 50 unseen prompts, so the",
        "headline is not a seed artefact.\n"]
L+=["### The flat 8/7/6-bit region is not a stuck value\n",
    "Accuracy is 0.2500 at three consecutive widths, which warranted a check. Two independent",
    "verifications show it is a genuine aggregate:",
    "",
    "- **Per-seed distributions are non-degenerate and identical**: 30 seeds at 0.167, 15 at",
    "  0.333, 5 at 0.500 → mean exactly 0.2500 at 8, 7 and 6 bits (5-bit: 31/14/5 → 0.2467).",
    "- **The quantizer is demonstrably active**: generated *text* differs from the 8-bit run in",
    "  28/50 prompts at 7-bit, 30/50 at 6-bit and 43/50 at 5-bit, and retained-token counts differ",
    "  in 11/50.",
    "",
    "So quantization is perturbing the model throughout the flat region; the perturbations simply",
    "do not cross the exact-match scoring threshold until 4-bit. The plateau is *effect below",
    "threshold*, not *absence of effect*.\n"]
open("sections/90_second_model.md","w").write("\n".join(L)+"\n")
print("wrote sections/90_second_model.md")
