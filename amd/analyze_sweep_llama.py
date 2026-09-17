"""Task 2: full bit-width sweep table on a COMMON seed set (0-49) for every width.

8-bit and 4-bit have n=150 available (Phase 1 / Phase 2 supplied identical cells); they are
restricted to seeds 0-49 so every width is measured on the same prompts. The n=150 value is
reported alongside where it exists.
"""
import sys, json, glob, re, statistics, collections; sys.path.insert(0,'/home/kxrx26/research test')

VALUE = re.compile(r"sk-[0-9a-f]{14}")
ANYSK = re.compile(r"sk-[0-9a-f]{4,}")

rows = {}
for p in glob.glob('results_llama/bits*/*.jsonl'):
    if 'failures' in p: continue
    for l in open(p):
        try: r = json.loads(l)
        except: continue
        if (r['arm'] == 4 and r['nominal_budget'] == 257 and r['promotion'] == 'attention'
                and r['context_target'] == 1029 and r['iso_condition'] == 'iso_token'):
            rows.setdefault((r['quant_bits'], r['seed']), r)

def repetitive(o: str) -> bool:
    """Degenerate repetition: some 3+ char substring repeated >=4 times back to back."""
    t = o.strip()
    if len(t) < 12: return False
    for L in range(3, 13):
        for i in range(0, min(len(t) - L * 4, 40)):
            u = t[i:i+L]
            if u * 4 in t: return True
    return False

# intercepted-generation reconstruction error and level audit
import os
ph = json.load(open('results_llama/posthoc.json')) if os.path.exists('results_llama/posthoc.json') else {}
err = {int(k): v for k, v in ph.get('intercepted_monotonicity',{}).get('error_pct_by_bits',{}).items()}
lvl = ph.get('level_occupancy', {})

print(f"{'bits':>4} {'n':>4} {'mean acc':>9} {'empty/EOS':>10} {'repetitive':>11} "
      f"{'degen':>6} {'clean':>6} {'mixed':>6} {'recon err':>10} {'levels':>12} {'viol':>5}")
table = {}
for w in [8, 7, 6, 5, 4, 3]:
    rs = [rows[(w, s)] for s in range(50) if (w, s) in rows]
    if not rs: continue
    acc = [r['accuracy'] for r in rs]
    turns = [o for r in rs for o in r['outputs']]
    empty = sum(1 for o in turns if o.strip() == '') / len(turns)
    rep = sum(1 for o in turns if repetitive(o)) / len(turns)
    wf = [sum(1 for o in r['outputs'] if VALUE.search(o)) for r in rs]   # well-formed turns
    degen = sum(1 for k in wf if k == 0)
    clean = sum(1 for k in wf if k == len(rs[0]['outputs']))
    mixed = len(rs) - degen - clean
    e = lvl.get(str(w), {})
    table[w] = dict(n=len(rs), mean_acc=statistics.mean(acc), empty=empty, repetitive=rep,
                    degenerate=degen, clean=clean, mixed=mixed,
                    recon_err_pct=err.get(w), max_levels=e.get('max_levels'),
                    limit=e.get('limit'), violations=e.get('violations'))
    print(f"{w:>4} {len(rs):>4} {statistics.mean(acc):>9.4f} {empty:>10.3f} {rep:>11.3f} "
          f"{degen:>6} {clean:>6} {mixed:>6} {(err.get(w) if err.get(w) is not None else float('nan')):>9.2f}% "
          f"{(str(e.get('max_levels'))+'/'+str(e.get('limit')) if e else '-'):>12} "
          f"{(e.get('violations') if e.get('violations') is not None else '-'):>5}")

print("\nfull-n values where available (8-bit and 4-bit have n=150):")
for w in [8, 4]:
    rs = [r for (b, s), r in rows.items() if b == w]
    print(f"  {w}-bit n={len(rs)}: mean acc {statistics.mean([r['accuracy'] for r in rs]):.4f}")
json.dump(table, open('results_llama/sweep_table.json', 'w'), indent=2)
print("\nwrote results_llama/sweep_table.json")
