#!/bin/bash
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/kxrx26/quant-rocm/bin/python
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
lg "waiting for supervised jobs"
while true; do
  I=$(wc -l < results/bits4/isomemory.jsonl 2>/dev/null || echo 0)
  P=$(wc -l < results/p2fix/phase2_fixed.jsonl 2>/dev/null || echo 0)
  [ "$I" -ge 1500 ] && [ "$P" -ge 750 ] && break
  ps -eo cmd | grep -q "[s]upervise.sh" || { lg "supervisors gone at iso=$I p2=$P; proceeding"; break; }
  sleep 30
done
lg "measurement done: iso=$(wc -l < results/bits4/isomemory.jsonl 2>/dev/null||echo 0)/1500 p2fix=$(wc -l < results/p2fix/phase2_fixed.jsonl 2>/dev/null||echo 0)/750 failures=$(cat results/*/*.failures.jsonl 2>/dev/null|wc -l)"
for s in analyze_isomemory analyze_phase2fix analyze_3factor analyze_sweep analyze_scoring; do
  lg "=== $s ==="; $PY $s.py 2>&1 | grep -viE "^loading|warning"
done
lg "=== REGEN SECTIONS ==="; $PY regen_sections.py 2>&1 | tail -2
lg "=== REPORTS ==="; $PY make_report.py 2>&1 | tail -2; $PY make_agreement.py 2>&1 | head -1
lg "=== ALL COMPLETE ==="
