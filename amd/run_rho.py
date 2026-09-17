"""Phase 2 manipulation check only (spec section 7): Spearman rho(eviction, promotion).

Runs immediately after Phase 2 so a broken manipulation surfaces before later stages consume
more time. P1 must be +1.000 (promotion score IS the eviction score); the rest near zero.
"""
import sys, json, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.task import build_prompt
from kvre.arms import make_cfg, PROMOTION_SIGNALS, PROMO_IDS
from kvre.cache_engine import QuantAudit
from run_posthoc import ProbeEngine, rho_for

model, tok = load()
SEEDS = list(range(12))
out = {}
print("=== Spearman rho(eviction, promotion) — section 7 manipulation check ===", flush=True)
for sig in PROMOTION_SIGNALS:
    r, n = rho_for(ProbeEngine, model, tok, sig, SEEDS)
    pid = PROMO_IDS[sig]
    tgt = "+1.000" if sig == "attention" else "~0"
    out[pid] = {"signal": sig, "rho": r, "n_prompts": n, "target": tgt}
    flag = ""
    if sig == "attention" and abs(r - 1.0) > 0.001: flag = "  <-- EXPECTED +1.000"
    if sig != "attention" and abs(r) > 0.3:         flag = "  <-- EXPECTED near zero"
    print(f"  {pid:>3} {sig:<12} rho = {r:+.4f}   (target {tgt}){flag}", flush=True)

# P5 genuine-ceiling check
over = 0
for s in SEEDS:
    r = Engine(model, tok).run_prompt(build_prompt(s), make_cfg(4, 257, quant_bits=4,
                                     promotion="oracle"), seed=s, audit=QuantAudit())
    over += 1 if r["oracle_oversubscribed"] else 0
print(f"\nP5 oversubscribed on {over}/{len(SEEDS)} prompts — "
      f"{'PASS (genuine ceiling)' if over==0 else 'FAIL'}", flush=True)
json.dump({"spearman_rho": out, "p5": {"n": len(SEEDS), "n_oversubscribed": over,
           "oversubscribed_rate": over/len(SEEDS)}},
          open("results/rho_check.json", "w"), indent=2)
print("wrote results/rho_check.json")
