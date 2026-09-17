#!/bin/bash
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/kxrx26/quant-rocm/bin/python
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
lg "waiting for 3B jobs"
while true; do
  P=$($PY -c "print(sum(1 for l in open('results3b/bits8/phase1_8bit.jsonl') if l.strip()))" 2>/dev/null || echo 0)
  F=$($PY -c "
import os
n=0
for w in (5,4,3):
    p='results3b/bits%d/fresh_bits%d.jsonl'%(w,w)
    n+= sum(1 for l in open(p) if l.strip()) if os.path.exists(p) else 0
print(n)" 2>/dev/null || echo 0)
  [ "$P" -ge 2693 ] && [ "$F" -ge 150 ] && break
  $PY -c "
import glob
n=0
for d in glob.glob('/proc/[0-9]*'):
    try: c=open(d+'/cmdline','rb').read().decode('utf8','ignore')
    except Exception: continue
    if 'quant-rocm/bin/python run_' in c: n+=1
import sys; sys.exit(0 if n>0 else 1)" || { lg "no jobs alive at phase1=$P fresh=$F; proceeding"; break; }
  sleep 30
done
lg "=== 3B ANALYSIS ==="; $PY analyze_3b.py 2>&1 | grep -v "^Loading"
lg "=== 3B SWEEP TABLE ==="; $PY analyze_sweep_3b.py 2>&1 | grep -v "^Loading"
lg "=== SECTIONS + REPORTS ==="; $PY regen_sections_3b.py 2>&1 | tail -2
$PY make_report.py 2>&1 | tail -2; $PY make_agreement.py 2>&1 | head -1; $PY make_summary.py 2>&1 | tail -1
lg "=== 3B COMPLETE ==="
