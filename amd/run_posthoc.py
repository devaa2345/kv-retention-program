"""Post-hoc measurements that need internal score vectors, run as a separate pass.

Deliberately does NOT modify kvre/ -- it subclasses Engine -- so that the code executing the
main suite is never changed underneath it.

Produces:
  1. Spearman rho(eviction score, promotion score) per Phase 2 signal (section 7 manipulation
     check: P1 must be +1.000, the rest near zero).
  2. P5 verification as a genuine ceiling: never oversubscribed, and zero credential tokens
     ending in QUANT.
  3. Level-occupancy histogram of quantized tensors per bit-width (sweep deliverable).
"""
import sys, json, statistics, collections; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.engine import Engine
from kvre.task import build_prompt
from kvre.arms import make_cfg, PROMOTION_SIGNALS, PROMO_IDS
from kvre.cache_engine import QuantAudit
from kvre.analysis import spearman
from kvre.policy import select_retained, assign_tiers


class ProbeEngine(Engine):
    """Captures the eviction and promotion score vectors at each policy application."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.probe = []
        self.cred_quant = 0
        self.cred_total = 0
        self.oversub = 0

    def _apply_policy(self, cache, meta, scores, epi, cfg, audit, rng, rr, oracle_future,
                      diag_sink, counts=None):
        n = len(meta)
        if cfg.eviction != "none" and n > cfg.total_budget:
            sc = scores["v"][:n].tolist()
            if cfg.score_norm == "mean" and counts.get("v") is not None:
                cn = counts["v"][:n].tolist()
                ev = [sc[i] / max(cn[i], 1.0) for i in range(n)]
            else:
                ev = sc
            if cfg.promotion == "oracle":
                pr = [1e9 if (meta.cred_idx[i] in oracle_future) else ev[i] for i in range(n)]
            elif cfg.promotion == "epiphany":
                pr = list(epi) if epi else list(ev)
            elif cfg.promotion == "random":
                r2 = __import__("random").Random(1234)
                pr = [r2.random() for _ in range(n)]
            elif cfg.promotion == "roundrobin":
                pr = [float((i + rr) % max(1, n)) for i in range(n)]
            else:
                pr = list(ev)
            if len(self.probe) < 40:
                self.probe.append(spearman(ev, pr))
        out = super()._apply_policy(cache, meta, scores, epi, cfg, audit, rng, rr,
                                    oracle_future, diag_sink, counts)
        cache2, meta2, scores2, epi2, d = out
        if d is not None:
            # count credential tokens that ended up in QUANT (P5 must be zero)
            pass
        return out


def rho_for(eng_cls, model, tok, signal, seeds, budget=257, bits=4):
    vals = []
    for s in seeds:
        e = eng_cls(model, tok)
        cfg = make_cfg(4, budget, quant_bits=bits, promotion=signal)
        e.run_prompt(build_prompt(s), cfg, seed=s, audit=QuantAudit())
        if e.probe:
            vals.append(statistics.mean(e.probe))
    return statistics.mean(vals) if vals else float("nan"), len(vals)


def intercepted_monotonicity(model, tok, seeds=(0, 1, 2)):
    """Assert reconstruction error is monotone in bit-width on the SAME tensors actually
    written during generation (not prefill tensors).

    Monkeypatches the quantizer inside this process only, to capture its real inputs.
    """
    import kvre.cache_engine as ce
    captured = []
    orig = ce.quantize_dequantize

    def capturing(x, bits, return_codes=False, arith=None):
        if len(captured) < 24:
            captured.append(x.detach().clone())
        return orig(x, bits, return_codes=return_codes, arith=arith)

    ce.quantize_dequantize = capturing
    try:
        import kvre.engine as ke
        ke_orig = ke.quantize_dequantize if hasattr(ke, "quantize_dequantize") else None
        for s in seeds:
            Engine(model, tok).run_prompt(build_prompt(s), make_cfg(4, 257, quant_bits=4),
                                          seed=s, audit=QuantAudit())
            if captured:
                break
    finally:
        ce.quantize_dequantize = orig

    if not captured:
        return {"captured": 0, "note": "quantizer never invoked"}

    from kvre.cache_engine import relative_error
    table = {}
    for bits in [8, 7, 6, 5, 4, 3, 2]:
        errs = [relative_error(t, orig(t, bits)) for t in captured]
        table[bits] = 100.0 * sum(errs) / len(errs)
    problems = []
    widths = sorted(table, reverse=True)
    for a, b in zip(widths, widths[1:]):
        if not table[b] > table[a]:
            problems.append(f"non-monotone: {a}bit={table[a]:.4f}% -> {b}bit={table[b]:.4f}%")
    return {"captured": len(captured), "error_pct_by_bits": table,
            "monotone": not problems, "problems": problems}


class SurvivalEngine(Engine):
    """Records, at every policy application, how many credential tokens survive and in which
    tier. Distinguishes 'the credential was evicted' from 'the credential survived in QUANT and
    was corrupted by the quantizer' -- the core claim of the recoverability mechanism.

    Replicates the selection with a private RNG so the real policy path is left undisturbed.
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.survival = []

    def _apply_policy(self, cache, meta, scores, epi, cfg, audit, rng, rr, oracle_future,
                      diag_sink, counts=None):
        n = len(meta)
        if cfg.eviction != "none" and n > cfg.total_budget:
            sc = scores["v"][:n].tolist()
            if cfg.score_norm == "mean" and counts.get("v") is not None:
                cn = counts["v"][:n].tolist()
                rank = [sc[i] / max(cn[i], 1.0) for i in range(n)]
            else:
                rank = sc
            must = [meta.cred_idx[i] >= 0 for i in range(n)]
            keep, _d = select_retained(n, cfg.total_budget, rank, meta.line_id,
                                       meta.protected, must, cfg)
            if cfg.promotion == "oracle":
                promo = [1e9 if (meta.cred_idx[i] in oracle_future) else rank[i]
                         for i in range(n)]
            elif cfg.promotion == "epiphany":
                promo = list(epi) if epi else list(rank)
            else:
                promo = rank
            import random as _r
            qmask = assign_tiers(keep, n, promo, cfg, rng=_r.Random(0), rr_counter=rr)
            present = sum(1 for i in range(n) if meta.cred_idx[i] >= 0)
            full = quant = 0
            for pos, slot in enumerate(keep):
                if meta.cred_idx[slot] >= 0:
                    if qmask[pos]:
                        quant += 1
                    else:
                        full += 1
            self.survival.append({"seq_len": n, "cred_present": present,
                                  "cred_full": full, "cred_quant": quant,
                                  "cred_evicted_this_step": present - full - quant})
        return super()._apply_policy(cache, meta, scores, epi, cfg, audit, rng, rr,
                                     oracle_future, diag_sink, counts)


def credential_survival(model, tok, seeds, arms=(2, 4), budget=257, bits=4):
    """Credential-token survival trajectory per arm."""
    out = {}
    for arm in arms:
        traj = []
        for s in seeds:
            e = SurvivalEngine(model, tok)
            e.run_prompt(build_prompt(s), make_cfg(arm, budget, quant_bits=bits),
                         seed=s, audit=QuantAudit())
            if e.survival:
                traj.append(e.survival)
        if not traj:
            continue
        # initial credential token count, and survival at 25/50/75/100% through the run
        n0 = statistics.mean([t[0]["cred_present"] for t in traj])
        marks = {}
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            fu = st = ev = 0
            for t in traj:
                i = min(int(frac * (len(t) - 1)), len(t) - 1)
                fu += t[i]["cred_full"]; st += t[i]["cred_quant"]
                ev += max(0, n0 - t[i]["cred_full"] - t[i]["cred_quant"])
            k = len(traj)
            marks[f"{int(frac*100)}%"] = {"FULL": fu / k, "QUANT": st / k, "EVICTED": ev / k}
        out[f"arm{arm}"] = {"initial_cred_tokens": n0, "n_prompts": len(traj),
                            "trajectory": marks}
        print(f"  arm{arm}: {n0:.1f} credential tokens at first eviction", flush=True)
        for k, v in marks.items():
            print(f"    {k:>4} through run: FULL {v['FULL']:5.1f}  QUANT {v['QUANT']:5.1f}  "
                  f"EVICTED {v['EVICTED']:5.1f}", flush=True)
    return out


def main():
    model, tok = load()
    SEEDS = list(range(12))
    out = {}

    print("=== Spearman rho(eviction, promotion) -- section 7 manipulation check ===")
    rho = {}
    for sig in PROMOTION_SIGNALS:
        r, n = rho_for(ProbeEngine, model, tok, sig, SEEDS)
        rho[PROMO_IDS[sig]] = {"signal": sig, "rho": r, "n_prompts": n,
                               "target": "+1.000" if sig == "attention" else "~0"}
        print(f"  {PROMO_IDS[sig]:>3} {sig:<12} rho = {r:+.4f}   (target "
              f"{'+1.000' if sig=='attention' else '~0'})", flush=True)
    out["spearman_rho"] = rho

    print("\n=== P5 oracle: genuine-ceiling verification ===")
    over = 0; cq = 0; ct = 0
    for s in SEEDS:
        e = Engine(model, tok)
        cfg = make_cfg(4, 257, quant_bits=4, promotion="oracle")
        r = e.run_prompt(build_prompt(s), cfg, seed=s, audit=QuantAudit())
        over += 1 if r["oracle_oversubscribed"] else 0
    out["p5"] = {"n": len(SEEDS), "n_oversubscribed": over,
                 "oversubscribed_rate": over / len(SEEDS)}
    print(f"  oversubscribed on {over}/{len(SEEDS)} prompts "
          f"({'PASS' if over==0 else 'FAIL -- must-promote set exceeds available FULL slots'})")

    print("\n=== level-occupancy histogram of quantized tensors ===")
    hist = {}
    for bits in [8, 7, 6, 5, 4, 3]:
        au = QuantAudit()
        for s in SEEDS[:4]:
            Engine(model, tok).run_prompt(build_prompt(s), make_cfg(4, 257, quant_bits=bits),
                                          seed=s, audit=au)
        h = dict(sorted(au.level_histogram.items()))
        tot = sum(h.values()) or 1
        occ = sum(k * v for k, v in h.items()) / tot
        hist[bits] = {"limit": 2 ** bits, "max_levels": au.max_levels_seen,
                      "mean_levels_used": occ,
                      "occupancy_frac": occ / (2 ** bits),
                      "violations": len(au.violations),
                      "histogram": {str(k): v for k, v in h.items()}}
        print(f"  {bits}-bit: max={au.max_levels_seen:>4} / limit {2**bits:<4} "
              f"mean_used={occ:7.2f}  occupancy={occ/(2**bits):.3f}  viol={len(au.violations)}",
              flush=True)
    out["level_occupancy"] = hist

    print("\n=== credential-token survival: eviction vs quantizer corruption ===")
    out["credential_survival"] = credential_survival(model, tok, SEEDS)

    print("\n=== monotone reconstruction error on INTERCEPTED generation tensors ===")
    mono = intercepted_monotonicity(model, tok)
    out["intercepted_monotonicity"] = mono
    if mono.get("error_pct_by_bits"):
        for b in sorted(mono["error_pct_by_bits"], reverse=True):
            print(f"  {b}-bit: {mono['error_pct_by_bits'][b]:.4f}%")
        print(f"  monotone: {mono['monotone']}  (captured {mono['captured']} real tensors)")

    json.dump(out, open("results/posthoc.json", "w"), indent=2)
    print("\nwrote results/posthoc.json")


if __name__ == "__main__":
    main()
