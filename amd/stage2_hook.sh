#!/bin/bash
# Fires when Phase 2 finishes. Triggered by the SWEEP stage banner, because Phase 2's own file
# no longer reaches 750 rows: its attention cells are reused from Phase 1 rather than rerun.
cd "/home/kxrx26/research test"
LOG=logs/run_all2.log
while true; do
  if grep -q "SWEEP START" "$LOG" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] phase2 complete - running rho manipulation check"
    PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 /home/kxrx26/quant-rocm/bin/python run_rho.py 2>&1 | grep -viE "^loading|warning"
    echo "[$(date +%H:%M:%S)] rho check done"
    break
  fi
  if ! pgrep -f "run_all" > /dev/null; then echo "suite ended before sweep banner"; break; fi
  sleep 20
done
