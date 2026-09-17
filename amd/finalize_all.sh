#!/bin/bash
# Final pass once Llama Phase 1 completes: analyses, sections, reports, and the zip rebuild.
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/kxrx26/quant-rocm/bin/python
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
while true; do
  N=$($PY -c "print(sum(1 for l in open('results_llama/bits8/phase1_8bit.jsonl') if l.strip()))" 2>/dev/null || echo 0)
  [ "$N" -ge 2700 ] && break
  L=$($PY - <<'P'
import glob
n=0
for d in glob.glob('/proc/[0-9]*'):
    try: c=open(d+'/cmdline','rb').read().decode('utf8','ignore')
    except Exception: continue
    if '/bin/python run_' in c and ' -c' not in c[:60]: n+=1
print(n)
P
)
  [ "$L" -eq 0 ] && { lg "no jobs alive at $N/2700; proceeding"; break; }
  sleep 30
done
lg "llama phase1 done: $($PY -c "print(sum(1 for l in open('results_llama/bits8/phase1_8bit.jsonl') if l.strip()))")/2700"
lg "=== LLAMA POSTHOC (slot now free) ==="
$PY run_posthoc_llama.py 2>&1 | grep -v "^Loading"
lg "=== ANALYSES ==="
$PY analyze_llama.py 2>&1 | grep -v "^Loading"
$PY analyze_sweep_llama.py 2>&1 | tail -10
$PY analyze_3b.py 2>&1 | tail -6
lg "=== SECTIONS + REPORTS ==="
$PY regen_sections_3b.py 2>&1 | tail -1
$PY make_report.py 2>&1 | tail -1
$PY make_agreement.py 2>&1 | head -1
$PY make_summary.py 2>&1 | tail -1
lg "=== ALL COMPLETE ==="
