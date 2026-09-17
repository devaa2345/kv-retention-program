#!/bin/bash
# Regenerates CSV + FINDINGS.md checkpoints continuously while the suite runs.
# Read-only w.r.t. the raw JSONL; safe to run alongside run_all.sh.
cd "/home/kxrx26/research test"
PY=/home/kxrx26/quant-rocm/bin/python
while true; do
  if ! pgrep -f "run_all.sh" > /dev/null; then
    $PY make_report.py > logs/checkpoint.log 2>&1
    echo "[$(date +%H:%M:%S)] FINAL checkpoint written; suite finished"
    break
  fi
  $PY make_report.py > logs/checkpoint.log 2>&1
  N=$(cat results/bits*/[!f]*.jsonl 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] checkpoint: $N rows -> results/csv/, FINDINGS.md"
  sleep 300
done
