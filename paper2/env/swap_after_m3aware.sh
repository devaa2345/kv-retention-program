#!/usr/bin/env bash
# Wait for M3-aware to reach 4000 records, then hand off from chain5 to chain6.
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
AW=runs/nvidia/grid_M3_ledger_aware.jsonl
for i in $(seq 1 400); do
  n=$(wc -l < "$AW" 2>/dev/null || echo 0)
  [ "$n" -ge 4000 ] && { echo "M3-aware complete: $n"; break; }
  pgrep -f run_grid_aware.py >/dev/null || { echo "aware run exited at $n"; break; }
  sleep 60
done
sleep 45                      # let analyze_aware finish and write its json
pkill -f wsl_chain5.sh 2>/dev/null || true
sleep 2
pkill -f 'run_grid_aware|analyze_aware' 2>/dev/null || true
sleep 10
exec bash env/wsl_chain6.sh
