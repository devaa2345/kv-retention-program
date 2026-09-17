import sys, time, torch; sys.path.insert(0,'/home/kxrx26/research test')
from kvre.model import load
from kvre.task import build_prompt
from kvre.engine import Engine
from kvre.arms import make_cfg
from kvre.cache_engine import QuantAudit

model, tok = load(); eng = Engine(model, tok)
for arm in [6,1,4]:
    cfg = make_cfg(arm, 257)
    au = QuantAudit()
    t0=time.time()
    r = eng.run_prompt(build_prompt(0), cfg, seed=0, audit=au, collect_dormancy=True)
    dt=time.time()-t0
    print(f"arm{arm} {cfg.protection}/{cfg.eviction}: acc={r['accuracy']:.3f} k={r['k']}/6 "
          f"eff_tok={r['effective_tokens']} full={r['n_full']} quant={r['n_quant']} "
          f"ctx={r['context_length']} dorm={r['dormancy_events']} "
          f"qviol={len(au.violations)} time={dt:.1f}s")
    print("   turn0 out:", repr(r['outputs'][0][:60]))
