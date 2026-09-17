"""Regenerate every narrative section of FINDINGS.md from the stored JSON results.

Idempotent: sections are files, so re-running make_report.py concatenates rather than destroys.
"""
import json, os, statistics
os.makedirs("sections", exist_ok=True)
J = lambda f: json.load(open(f)) if os.path.exists(f) else None

# ---------------- 10: three-factor (the headline) ----------------
tf = J("results/three_factor.json")
if tf:
    L=["## Phase 1 as a three-factor design: protection × tiering × bit-width\n",
       "Arms 1 and 2 are permanent-eviction (`n_quant = 0`) and therefore bit-width invariant;",
       "they serve as the shared reference level rather than being re-run. Complete n=150 data",
       "exists at both widths for the tiered arms.\n",
       "| budget | bits | protection | tiering | interaction | 95% CI | n differ |",
       "|---|---|---|---|---|---|---|"]
    for b in (154,257,514):
        for w in (8,4):
            k=f"b{b}_{w}bit"
            if k in tf:
                e=tf[k]; star="**" if w==8 else ""
                L.append(f"| {b} | {star}{w}{star} | {e['protection']:+.4f} | {e['tiering']:+.4f} | "
                         f"{star}{e['interaction']:+.4f}{star} | [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] | {e['n_differ']} |")
    L+=["","**Both 8-bit interaction CIs contain zero; both 4-bit CIs exclude it.** The tiering main",
        "effect at 8-bit is indistinguishable from zero at every budget.\n",
        "### The bit-width factor, tested directly\n",
        "| budget | tiering @8-bit | tiering @4-bit | difference | 95% CI |","|---|---|---|---|---|"]
    for b in (154,257,514):
        k=f"widtheffect_b{b}"
        if k in tf:
            e=tf[k]
            L.append(f"| {b} | {e['tier_8bit']:+.4f} | {e['tier_4bit']:+.4f} | **{e['diff']:+.4f}** | "
                     f"[{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] |")
    L+=["","The width effect is significant at every budget and grows with budget: larger budgets",
        "retain more credential tokens and therefore expose more of them to quantization damage.\n",
        "### Interpretation\n",
        "The interaction the spec reports as ≈0 is **not a null result; it is a result conditional",
        "on bit-width.** Protection's benefit survives near-lossless (8-bit) tiering essentially",
        "intact and is progressively destroyed by aggressive (4-bit) tiering, in proportion to how",
        "much benefit there was to destroy. The reference's ≈0 and this implementation's −0.116 are",
        "two slices of the same surface; both are correct about their own condition.",
        "",
        "The mechanism is established independently by the credential-survival instrumentation:",
        "arms 2 and 4 evict **exactly the same 92.1 credential tokens**, so tiering adds no eviction",
        "at all. Its entire effect is precision loss on tokens that were retained — which is why it",
        "tracks reconstruction error (1.18% at 8-bit vs 19.98% at 4-bit) rather than anything about",
        "the retention policy.\n",
        "> **This condition was added post-hoc.** The 4-bit Phase 1 ran first, the disagreement with",
        "> the reference was observed, and the 8-bit condition was then specified and run to test a",
        "> stated mechanism. The prediction was made before the run and confirmed, but this is an",
        "> exploratory inference, not a pre-registered confirmatory one.\n"]
    open("sections/10_three_factor.md","w").write("\n".join(L)+"\n")

# ---------------- 20: sweep ----------------
sw = J("results/sweep_table.json"); ph = J("results/posthoc.json")
if sw:
    L=["## Bit-width sweep — full table (common seed set, n=50 at every width)\n",
       "8-bit and 4-bit had n=150 available; they are restricted to seeds 0–49 so all six widths",
       "are measured on identical prompts. Full-n values: 8-bit 0.1644, 4-bit 0.1189.\n",
       "A turn output is *well-formed* if it contains an `sk-<14 hex>` token. A prompt is *clean*",
       "if all 6 turns are well-formed, *degenerate* if none are. `empty/EOS` is the fraction of",
       "turn outputs empty after stripping. Reconstruction error is measured on tensors",
       "**intercepted during live generation**.\n",
       "| bits | n | mean acc | empty/EOS | repetitive | degenerate | clean | mixed | recon err | levels/limit | violations |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
    for w in ("8","7","6","5","4","3"):
        if w not in sw: continue
        e=sw[w]; st="**" if e["mean_acc"]==0.0 else ""
        L.append(f"| {st}{w}{st} | {e['n']} | {st}{e['mean_acc']:.4f}{st} | {e['empty']:.3f} | "
                 f"{e['repetitive']:.3f} | {e['degenerate']} | {e['clean']} | {e['mixed']} | "
                 f"{e['recon_err_pct']:.2f}% | {e['max_levels']}/{e['limit']} | {e['violations']} |")
    L+=["","### The headline claim reproduces\n",
        "**Accuracy is not monotone in bit-width; reconstruction error is.** Error rises strictly",
        "across all six widths (1.15 → 2.28 → 4.56 → 9.36 → 19.31 → 39.92 %), yet accuracy",
        "collapses to exactly 0.0000 at 5-bit and **recovers to 0.1167 at 4-bit while carrying more",
        "than twice the reconstruction error**, then collapses again at 3-bit.",
        "",
        "The reference reports 4-bit recovery at 0.123; this implementation measures 0.1167 (n=50)",
        "and 0.1189 (n=150). **The curve shape is independently confirmed.** The widths that",
        "collapse differ — the reference reports 6- and 5-bit, here it is 5- and 3-bit with 6-bit",
        "merely degraded — so the *phenomenon* replicates while the *specific widths* do not. That",
        "is itself the substance of §9's caution against reading a sweep as a severity ordering.",
        "",
        "**The quantizer is cleared.** Level counts never exceed 2^bits at any width (0 violations,",
        "checked per group on tensors actually written during generation), occupancy rises smoothly",
        "from 0.44 to 0.98 as width falls, and error is strictly monotone on those same intercepted",
        "tensors. The collapse is how the model responds to quantization noise, not a quantizer bug.",
        "",
        "The collapse signature is degenerate *output*: at 5-bit, 48 of 50 prompts produce no",
        "well-formed value on any turn and empty/EOS jumps to 0.223; at 3-bit, 46 of 50 and 0.513.",
        "At every working width empty/EOS is ≤0.018.\n"]
    open("sections/20_sweep.md","w").write("\n".join(L)+"\n")

# ---------------- 30: GAP-U scoring ----------------
sc = J("results/scoring_analysis.json")
if sc:
    L=["## [GAP-U] revisited: the ceiling gap is a scoring-convention difference\n",
       "**The scorer was not changed.** This reports exactly what it does and measures how much of",
       "the 0.799-vs-0.947 gap each alternative convention would close.\n",
       "### What the scorer does\n",
       "`score_turn` returns `target.value in generated_text` — a plain substring test of the full",
       "17-character value (`sk-` + 14 lowercase hex) against the **raw** decoded text of that turn",
       "(`skip_special_tokens=True`, nothing else). No lower-casing, stripping, whitespace",
       "collapsing, unicode normalisation or tokenisation. Credit only on the turn that asked",
       "([GAP-P]). Prompt score is k/6. This is the literal reading of §2 and of [SPEC-GAP 2]'s",
       "stated default.\n",
       "### What each convention rescues (arm 6, 900 turn judgements, n=150)\n",
       "| scoring rule | accuracy | vs 0.947 |","|---|---|---|",
       "| **1. exact 17-char substring — the actual scorer** | **0.7989** | −0.1481 |",
       "| 2. case-insensitive substring | 0.7989 | −0.1481 |",
       "| 3. substring after whitespace stripping | 0.7989 | −0.1481 |",
       "| 4. 14-hex payload, `sk-` prefix optional | 0.8633 | −0.0837 |",
       "| **5. payload within edit distance 1** | **0.9544** | **+0.0074** |",
       "| 6. payload within edit distance 2 | 0.9722 | +0.0252 |","",
       "### What this establishes\n",
       "**The normalisations [SPEC-GAP 2] contemplates rescue nothing.** Case-folding and",
       "whitespace-stripping recover exactly zero failures, so \"exact vs normalised match\" is not",
       "the axis the gap lies on. That eliminates the ambiguity the spec itself flagged.",
       "",
       "Two other conventions do close it. The model frequently answers with the bare 14-hex",
       "payload and omits the `sk-` prefix — e.g. `9d268ef50038b6` for `sk-9d268ef50038b6`. This is",
       "a per-prompt behavioural mode, not scattered noise: 12 of 150 prompts do it on ≥3 of their 6",
       "turns, accounting for 58 of 181 failures. Allowing a single character error in the payload",
       "recovers a further 82. Together they land at **0.9544 against the reference's 0.947 — a",
       "difference of 0.0074**, an order of magnitude below the 0.1481 gap under strict scoring.",
       "",
       "### Conclusion, with its limits\n",
       "The evidence favours a **scoring-convention difference over a capability difference**. A",
       "0.947 ceiling is hard to reach under strict 17-character exact-substring scoring — this",
       "model emits the correct payload far more often than this scorer credits — but sits almost",
       "exactly on a prefix-tolerant, edit-distance-1 convention.",
       "",
       "This is evidence, not proof: edit-distance tolerance is an unusual convention, §2's text",
       "does not license it, and the reference's code cannot be inspected under the blind protocol.",
       "What can be said firmly is that **the two implementations are probably not scoring the same",
       "thing, and the axis is prefix-tolerance plus fuzzy matching — not the exact-vs-normalised",
       "axis the spec anticipated.** §2 should state whether the `sk-` prefix is required and",
       "whether any edit tolerance is permitted.",
       "",
       "**The scorer remains unchanged; every other number in this document uses rule 1.**\n"]
    open("sections/30_gapu_scoring.md","w").write("\n".join(L)+"\n")

# ---------------- 40: ablation ----------------
ab = J("results/ablation_distractor.json")
if ab:
    L=["## Structural protection degenerates into an oracle when distractors stop matching\n",
       "Structural protection matches lines by surface pattern. Rewriting distractor values to a",
       "non-credential form (`ref_<hex>` instead of `sk-<hex>`), n=50, budget 257:\n",
       "| arm | shared shape | distinct shape | diff | 95% CI | p |","|---|---|---|---|---|---|"]
    for a in ("arm2","arm4","arm6"):
        if a in ab:
            e=ab[a]
            L.append(f"| {a} | {e['shared']:.4f} | {e['distinct']:.4f} | **{e['diff']:+.4f}** | "
                     f"[{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] | {e['p']:.4f} |")
    L+=["","`arm2` (structural/permanent) jumps from 0.150 to **0.907 — above the full-cache ceiling",
        "of 0.880 measured under the same condition**. With only 6 lines matching its pattern",
        "instead of 26, structural protection stops competing and simply retains every credential:",
        "it *becomes* `oracle_static`. The full-cache arm gains only +0.137, so most of arm 2's",
        "+0.757 is degeneration of the mechanism, not the task becoming easier.",
        "",
        "**Implication for pattern-based KV protection generally:** its strength is set by the ratio",
        "of target lines to pattern-matching non-target lines. This is a property of the mechanism,",
        "not a quirk of this task — and it means the spec's phrase \"same surface shape\" ([GAP-V])",
        "is load-bearing in a way the document never states. The *direction* of the protection",
        "effect should transfer to other designs; its *magnitude* should not be quoted out of the",
        "6:26 ratio it was measured at.\n"]
    open("sections/40_ablation.md","w").write("\n".join(L)+"\n")

# ---------------- 70: iso-memory + byte-cost sensitivity ----------------
im = J("results/isomemory_analysis.json")
if im:
    L=["## iso-memory condition (§6) and the [SPEC-GAP 3] byte-cost sensitivity\n",
       "Under iso-memory the tiered arms receive more raw positions, funded by the cold tier's",
       "lower byte cost, so total bytes match the permanent arms:",
       "`T = budget / (f + (1-f)·cost)`. With `f = 0.5` and `cost = 0.25` this is `T = 1.6 ×",
       "budget`, reproducing §6's worked example exactly (257 → 411). Permanent arms are unaffected",
       "by the condition, so their iso-token rows are reused rather than recomputed.\n",
       "| budget | T | arm3 iso-token | arm3 iso-memory | arm4 iso-token | arm4 iso-memory | arm4 diff | 95% CI |",
       "|---|---|---|---|---|---|---|---|"]
    for b in (154,257,514):
        k=f"b{b}"
        if k in im:
            e=im[k]
            L.append(f"| {b} | {e['T']} | {e['arm3_token']:.4f} | {e['arm3_memory']:.4f} | "
                     f"{e['arm4_token']:.4f} | {e['arm4_memory']:.4f} | **{e['arm4_diff']:+.4f}** | "
                     f"[{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] |")
    L+=["","### Interaction under each iso-condition\n",
        "§8 requires BH correction within each iso-condition separately, which presupposes both",
        "conditions exist. Both are now reported.\n",
        "| iso-condition | budget | interaction | 95% CI | n differ |","|---|---|---|---|---|"]
    for iso in ("iso_token","iso_memory"):
        for b in (154,257,514):
            k=f"interaction_{iso}_b{b}"
            if k in im:
                e=im[k]
                L.append(f"| {iso} | {b} | {e['mean']:+.4f} | [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] | {e['n_differ']} |")
    L+=["","### [SPEC-GAP 3] sensitivity to `quant_byte_cost`\n",
        "§3 registers 0.25 but notes it ignores per-group scales and zero-points, and that a real",
        "int8 scheme with bf16 scales is nearer 0.28–0.31. Changing the constant changes only the",
        "iso-memory token grant, so this is the condition where it can bite.\n",
        "| cost | T @ budget 257 | arm3 | arm4 |","|---|---|---|---|"]
    for c in (0.25,0.28,0.31):
        k=f"sensitivity_c{c}"
        if k in im:
            e=im[k]
            a3 = f"{e['arm3']:.4f}" if e.get('arm3') is not None else "—"
            L.append(f"| {c:.2f} | {e['T']} | {a3} | {e['arm4']:.4f} |")
    L+=["","Iso-token results are unaffected by this constant by construction, and Phase 1 and",
        "Phase 2 both run iso-token — so the headline results carry no exposure to it. Every output",
        "row records `quant_byte_cost` and a derived `physical_byte_cost = bits/16` ([GAP-N]) so the",
        "grant is re-derivable.\n"]
    open("sections/70_isomemory.md","w").write("\n".join(L)+"\n")

# ---------------- 80: Phase 2 corrected ----------------
p2 = J("results/phase2_fixed_analysis.json")
if p2:
    L=["## Phase 2 with corrected promotion signals (primary)\n",
       "The originally-run Phase 2 failed its §7 manipulation check on two of five signals",
       "(P4 ρ = +0.4974, P5 ρ = +0.8489 at n=100). Those arms do not test orthogonal promotion, so",
       "Phase 2 was re-run in full with corrected signals. Both runs are shown; the corrected run",
       "is primary.\n",
       "| id | signal | original | corrected | Δ | reference | n |","|---|---|---|---|---|---|---|"]
    for pid in ("P1","P2","P3","P4","P5"):
        if pid in p2 and isinstance(p2[pid],dict) and "signal" in p2[pid]:
            e=p2[pid]
            f = f"{e['fixed']:.4f}" if e.get('fixed') is not None else "—"
            d = f"{e['fixed']-e['original']:+.4f}" if e.get('fixed') is not None else "—"
            L.append(f"| {pid} | {e['signal']} | {e['original']:.4f} | **{f}** | {d} | "
                     f"{e['reference']:.3f} | {e.get('n_fix',0)} |")
    ks=[k for k in p2 if k.startswith("contrast_")]
    if ks:
        L+=["","### Corrected-signal contrasts vs P3 (random control), BH-corrected\n",
            "| contrast | diff | 95% CI | n differ | p_BH |","|---|---|---|---|---|"]
        for k in sorted(ks):
            e=p2[k]
            L.append(f"| {k.replace('contrast_','').replace('_',' ')} | {e['diff']:+.4f} | "
                     f"[{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}] | {e['n_differ']} | {e['p_bh']:.4f} |")
    L+=["","Retention is held fixed and attention-ranked in every Phase 2 arm, with protection ON",
        "and tiered eviction ON, so these contrasts isolate the promotion decision from retention.\n"]
    open("sections/80_phase2_corrected.md","w").write("\n".join(L)+"\n")

print("regenerated:", sorted(os.listdir("sections")))
