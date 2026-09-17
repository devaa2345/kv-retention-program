"""Task 3: exactly what the scorer does, and how many failures each looser rule would rescue.

The scorer is NOT changed. This measures whether the [GAP-U] ceiling gap (0.799 vs the
reference's 0.947) could be a scoring-convention difference rather than a capability difference.
"""
import sys, json, glob, re, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.task import build_prompt, target_for

def edit_distance(a, b, cap=3):
    if abs(len(a)-len(b)) > cap: return cap+1
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(ca != cb)))
        prev = cur
        if min(prev) > cap: return cap+1
    return prev[-1]

CAND = re.compile(r"sk-[0-9a-zA-Z]{6,20}")

rows = {}
for p in glob.glob('results/bits*/*.jsonl'):
    if 'failures' in p: continue
    for l in open(p):
        try: r = json.loads(l)
        except: continue
        if r['arm'] == 6 and r['context_target'] == 1029:
            rows.setdefault(r['seed'], r)

tot = hit = 0
rescue = {"exact (current)": 0, "case-insensitive": 0, "whitespace-stripped": 0,
          "edit distance <= 1": 0, "edit distance <= 2": 0}
fail_kinds = {"near_miss": 0, "wrong_value": 0, "no_value": 0}
for seed, r in sorted(rows.items()):
    p = build_prompt(seed)
    allv = set(re.findall(r"sk-[0-9a-f]{14}", p.context))
    for t, o in enumerate(r['outputs']):
        tgt = target_for(p, t).value
        tot += 1
        if tgt in o:                                  # the actual scorer
            hit += 1
            continue
        # --- failure: classify and test looser rules ---
        rescue["exact (current)"] += 0
        if tgt.lower() in o.lower(): rescue["case-insensitive"] += 1
        if tgt in re.sub(r"\s+", "", o): rescue["whitespace-stripped"] += 1
        cands = CAND.findall(o)
        best = min((edit_distance(tgt, c) for c in cands), default=99)
        if best <= 1: rescue["edit distance <= 1"] += 1
        if best <= 2: rescue["edit distance <= 2"] += 1
        if not cands: fail_kinds["no_value"] += 1
        elif any(c in allv and c != tgt for c in cands): fail_kinds["wrong_value"] += 1
        else: fail_kinds["near_miss"] += 1

base = hit/tot
print(f"arm 6 (full_cache_ref), n={len(rows)} prompts, {tot} turn-level judgements")
print(f"  measured accuracy under the actual scorer: {base:.4f}")
print(f"  failures: {tot-hit}  (near-miss {fail_kinds['near_miss']}, "
      f"wrong-value {fail_kinds['wrong_value']}, no-value {fail_kinds['no_value']})")
print()
print(f"{'rule':<24}{'rescued':>9}{'accuracy':>11}{'vs 0.947':>10}")
for k in ["exact (current)", "case-insensitive", "whitespace-stripped",
          "edit distance <= 1", "edit distance <= 2"]:
    a = (hit + rescue[k]) / tot
    print(f"{k:<24}{rescue[k]:>9}{a:>11.4f}{a-0.947:>+10.4f}")
json.dump({"n_prompts": len(rows), "judgements": tot, "hits": hit, "accuracy": base,
           "rescued": rescue, "failure_kinds": fail_kinds},
          open("results/scoring_analysis.json", "w"), indent=2)
print("\nwrote results/scoring_analysis.json")
