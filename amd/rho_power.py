"""Spearman rho(eviction, promotion) at n>=100, for BOTH promotion definitions.

`old` = the definition in force when the completed Phase 2 data was generated (P5 fell back to
the eviction score for non-must positions; P4 rotated over raw position).
`new` = the corrected definition (P5 membership-only; P4 rotates over a fixed permutation).

Reporting both is what determines whether the completed Phase 2 arms are a valid test of
orthogonal promotion or a failed manipulation.
"""
import sys, json, random, statistics; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine, _stable_perm
from kvre.task import build_prompt
from kvre.arms import make_cfg, PROMOTION_SIGNALS, PROMO_IDS
from kvre.cache_engine import QuantAudit
from kvre.analysis import spearman
from kvre.policy import select_retained, assign_tiers

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100


class DualProbe(Engine):
    """Captures rho under both the old and the corrected promotion definitions."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.old, self.new = [], []

    def _apply_policy(self, cache, meta, scores, epi, cfg, audit, rng, rr, oracle_future,
                      diag_sink, counts=None):
        n = len(meta)
        if cfg.eviction != "none" and n > cfg.total_budget and len(self.old) < 30:
            sc = scores["v"][:n].tolist()
            if cfg.score_norm == "mean" and counts.get("v") is not None:
                cn = counts["v"][:n].tolist()
                ev = [sc[i]/max(cn[i],1.0) for i in range(n)]
            else:
                ev = sc
            sig = cfg.promotion
            r2 = random.Random(1234)
            if sig == "oracle":
                old = [1e9 if (meta.cred_idx[i] in oracle_future) else ev[i] for i in range(n)]
                new = [1.0 if (meta.cred_idx[i] in oracle_future) else 0.0 for i in range(n)]
            elif sig == "roundrobin":
                old = [float((i + rr) % n) for i in range(n)]
                perm = _stable_perm(n)
                new = [float((perm[i] + rr) % n) for i in range(n)]
            elif sig == "random":
                old = [r2.random() for _ in range(n)]
                new = [r2.random() for _ in range(n)]
            elif sig == "epiphany":
                old = new = (list(epi) if epi else list(ev))
            else:
                old = new = ev
            self.old.append(spearman(ev, old))
            self.new.append(spearman(ev, new))
        return super()._apply_policy(cache, meta, scores, epi, cfg, audit, rng, rr,
                                     oracle_future, diag_sink, counts)


def boot_ci(vals, n_boot=5000, seed=7):
    rng = random.Random(seed); n = len(vals)
    m = sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(n_boot))
    return m[int(.025*n_boot)], m[int(.975*n_boot)]


model, tok = load()
out = {}
print(f"=== Spearman rho(eviction, promotion), n={N} prompts ===", flush=True)
print(f"{'id':>3} {'signal':<12} {'rho_old':>9} {'95% CI':>20} {'rho_new':>9} {'95% CI':>20}")
for sig in PROMOTION_SIGNALS:
    po, pn = [], []
    for s in range(N):
        e = DualProbe(model, tok)
        e.run_prompt(build_prompt(s), make_cfg(4, 257, quant_bits=4, promotion=sig),
                     seed=s, audit=QuantAudit())
        if e.old:
            po.append(statistics.mean(e.old)); pn.append(statistics.mean(e.new))
    co, cn = boot_ci(po), boot_ci(pn)
    pid = PROMO_IDS[sig]
    out[pid] = {"signal": sig, "n_prompts": len(po),
                "rho_old": statistics.mean(po), "ci_old": list(co),
                "rho_new": statistics.mean(pn), "ci_new": list(cn),
                "target": "+1.000" if sig == "attention" else "~0"}
    print(f"{pid:>3} {sig:<12} {statistics.mean(po):>+9.4f} "
          f"[{co[0]:>+7.4f},{co[1]:>+7.4f}] {statistics.mean(pn):>+9.4f} "
          f"[{cn[0]:>+7.4f},{cn[1]:>+7.4f}]", flush=True)
json.dump(out, open("results/rho_power.json","w"), indent=2)
print("\nwrote results/rho_power.json")
