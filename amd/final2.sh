#!/bin/bash
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/kxrx26/quant-rocm/bin/python
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
while true; do
  A=$(wc -l < results/bits4/sens_c0.28.jsonl 2>/dev/null || echo 0)
  B=$(wc -l < results/bits4/sens_c0.31.jsonl 2>/dev/null || echo 0)
  [ "$A" -ge 300 ] && [ "$B" -ge 300 ] && break
  ps -eo cmd | grep -q "[s]upervise.sh" || { lg "supervisors gone at $A/$B; proceeding"; break; }
  sleep 30
done
lg "sensitivity done: c0.28=$(wc -l < results/bits4/sens_c0.28.jsonl 2>/dev/null||echo 0)/300 c0.31=$(wc -l < results/bits4/sens_c0.31.jsonl 2>/dev/null||echo 0)/300"
lg "=== iso-memory + sensitivity ==="; $PY analyze_isomemory.py 2>&1 | grep -viE "^loading|warning"
lg "=== regen sections ==="; $PY regen_sections.py 2>&1 | tail -1
lg "=== reports ==="; $PY make_report.py 2>&1 | tail -2; $PY make_agreement.py 2>&1 | head -1; $PY make_summary.py 2>&1 | tail -1
lg "=== EVERYTHING COMPLETE ==="
