#!/bin/bash
# Waits for Llama Phase 1 to be running on a healthy card, then measures its real per-cell cost.
cd "/home/kxrx26/research test"
PY=/home/kxrx26/quant-rocm/bin/python
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
lg "waiting for Llama Phase 1 to start"
while true; do
  R=$($PY -c "
import glob
n=0
for d in glob.glob('/proc/[0-9]*'):
    try: c=open(d+'/cmdline','rb').read().decode('utf8','ignore')
    except Exception: continue
    if 'phase1_llama' in c and '/bin/python' in c: n+=1
print(n)")
  [ "$R" -ge 1 ] && break
  sleep 20
done
sleep 60   # let it get past model load
lg "measuring for 4 minutes"
$PY - <<'P'
import time, json, collections
def rows():
    try: return [json.loads(l) for l in open('results_llama/bits8/phase1_8bit.jsonl') if l.strip()]
    except FileNotFoundError: return []
a=rows(); t0=time.time(); time.sleep(240); b=rows(); dt=time.time()-t0
new=b[len(a):]
r=len(new)/dt
print('  measured %d cells in %.0fs -> %.3f cells/s (%.1f s/cell)'%(len(new),dt,r,1/max(r,1e-9)))
c=collections.Counter((x['nominal_budget'],x['arm']) for x in new)
print('  cells measured were:', dict(sorted(c.items())))
done=collections.Counter((x['nominal_budget'],x['arm']) for x in b)
rem=sum(max(0,150-done[(bu,ar)]) for bu in (154,257,514) for ar in range(1,7))
print('  remaining: %d cells'%rem)
print('  ETA at this rate: %.1f h  (n=150)'%(rem/max(r,1e-9)/3600))
print('  ETA if n reduced to 75: %.1f h'%(max(0,rem-1350)/max(r,1e-9)/3600 if rem>1350 else rem/2/max(r,1e-9)/3600))
P
lg "=== MEASUREMENT DONE ==="
