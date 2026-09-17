#!/bin/bash
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/kxrx26/quant-rocm/bin/python
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
# Wait until Phase 1 reaches 2700 rows. Row count is the only condition -- the previous
# version also trusted a liveness probe, which returned 0 spuriously and fired early.
STALL=0; LAST=0
while true; do
  N=$($PY -c "
try: print(sum(1 for l in open('results_llama/bits8/phase1_8bit.jsonl') if l.strip()))
except Exception: print(0)")
  [ "$N" -ge 2700 ] && { lg "phase1 complete: $N/2700"; break; }
  if [ "$N" -eq "$LAST" ]; then STALL=$((STALL+1)); else STALL=0; fi
  LAST=$N
  [ "$STALL" -ge 30 ] && { lg "stalled at $N/2700 for 15min; proceeding"; break; }
  sleep 30
done
lg "=== ANALYSES ==="
$PY analyze_llama.py 2>&1 | grep -v "^Loading" | tail -18
$PY analyze_sweep_llama.py 2>&1 | tail -10
lg "=== SECTIONS + REPORTS ==="
$PY regen_sections_3b.py 2>&1 | tail -1
$PY make_report.py 2>&1 | tail -1
$PY make_agreement.py 2>&1 | head -1
$PY make_summary.py 2>&1 | tail -1
lg "=== FINAL COMPLETE ==="
