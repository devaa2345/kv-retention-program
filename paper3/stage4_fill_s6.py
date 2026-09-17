"""Fill STAGE4_REPORT.md section 6 from the determinism test results.

Reads out/stage4_determinism_M2.json and out/stage4_determinism_M3.json (written by
stage4_determinism.py), and replaces the PENDING section 6 in place. It does NOT hash; the filled
section is reviewed before the report is frozen.

The interpretation is chosen by the measured verdicts, with M2 as the control, using the three
outcomes stage4_determinism.py was written to separate:
  (A) run-to-run nondeterminism  -> a noise floor on every M3 arm
  (B) path dependence only       -> only the two oracle arms are affected
  deterministic at this sample   -> the Stage 4 mismatch rate is too low for n to catch, and stays
                                     unexplained at this n
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPORT = HERE / "STAGE4_REPORT.md"
NL = chr(10)


def load(tag):
    p = HERE / "out" / ("stage4_determinism_%s.json" % tag)
    if not p.exists():
        raise SystemExit("missing %s -- the determinism test has not finished" % p.name)
    return json.loads(p.read_text(encoding="utf-8"))


def kind(v):
    if v.startswith("(A)"):
        return "A"
    if v.startswith("(B)"):
        return "B"
    return "D"


def main() -> int:
    m2, m3 = load("M2"), load("M3")
    L = []
    W = L.append
    W("## 6. Generation determinism")
    W("")
    W("`stage4_determinism.py` ran after every Stage 4 package had finished, single-process. It took")
    W("12 c ≈ 19 instances at C = %d and the causal keep-set, then generated each query four ways:"
      % m3["C"])
    W("each path run twice (same-path reruns), both paths on the identical keep-set (cross-path), and")
    W("two independent prefills compared on their last-position logits. M2 is the control.")
    W("")
    W("| model | causal rerun identical | prescient rerun identical | cross-path identical | max \\|Δ logit\\|, two prefills | verdict |")
    W("|---|---|---|---|---|---|")
    for d in (m2, m3):
        c = d["counts"]
        t = c["total"]
        W("| %s | %d / %d | %d / %d | %d / %d | %.3e | %s |"
          % (d["model"], c["same_path_causal"], t, c["same_path_presc"], t,
             c["cross_path"], t, d["max_logit_diff"], d["verdict"]))
    W("")

    k2, k3 = kind(m2["verdict"]), kind(m3["verdict"])
    if k2 != "D":
        W("**The control is not clean.** M2 was expected to be deterministic, since the Stage 4")
        W("oracle identity check agreed 100 of 100 byte for byte, and here it is not. That undermines")
        W("the M2 identity result as well, and every between-arm comparison on both models needs the")
        W("measured nondeterminism rate as its noise floor. This is flagged ahead of every other")
        W("reading in this section.")
        W("")
    if k3 == "A":
        W("**M3 shows run-to-run nondeterminism.** Rerunning the *same* path on the *same* keep-set")
        W("gives different text. So the Stage 4 prescient/causal mismatch is not specific to the")
        W("oracles: every M3 arm carries a generation noise floor. Between-arm differences on M3 below")
        W("the observed disagreement rate are not resolvable, and M3's I(C) ≈ 0 at C ≥ 128 is zero")
        W("within that floor. M2 is %s, so the effect is model- or kernel-specific, not a harness"
          % ("deterministic" if k2 == "D" else "also affected"))
        W("defect. No Stage 4 verdict depends on an M3 difference smaller than this floor: the cost")
        W("axis, the crossover and the c = 1 results all move by far more.")
    elif k3 == "B":
        W("**M3 is self-consistent within each path but path-dependent.** Each path reproduces itself")
        W("exactly, while prefill-once-and-clone and prefill-per-query give different text on the")
        W("identical keep-set. So only the two oracle arms are affected, because they are the only")
        W("arms run through different paths. Every other Stage 4 arm used the clone path throughout")
        W("and is internally comparable. M3's I(C) at C ≥ 128 compares the two paths and is read as")
        W("zero within the measured path disagreement, not as an inversion.")
    else:
        W("**M3 is deterministic and path-independent at this sample.** Neither path reproduces the")
        W("Stage 4 mismatch on these 12 instances × 4 queries. That mismatch was 6 of 100 instance")
        W("scores and up to 14 of 100 generations at C = 512. At that rate a 48-generation sample can")
        W("plausibly miss it, so the test does not settle the cause. The Stage 4 discrepancy stays")
        W("**unexplained at this n**. Because the keep-sets were shown identical on CPU, it remains a")
        W("generation-level effect confined to the oracle comparison, and M3's I(C) ≈ 0 at C ≥ 128 is")
        W("still read within a noise floor of about 0.03.")
    W("")
    W("**Correction to the earlier determinism evidence.** Stage 2's 176 byte-identical duplicate")
    W("keys all came from M2 runs, so the programme's determinism claim had covered M2 only until")
    W("this test.")
    W("")

    text = REPORT.read_text(encoding="utf-8")
    start = text.index("## 6. Generation determinism")
    end = text.index(NL + "---" + NL, start)
    new = text[:start] + NL.join(L) + text[end:]
    new = new.replace("| M3 generation determinism | **§6, pending** |",
                      "| M3 generation determinism | %s (§6) |" % m3["verdict"])
    REPORT.write_text(new, encoding="utf-8", newline="\n")
    print("section 6 filled: M2 %s | M3 %s" % (m2["verdict"], m3["verdict"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
