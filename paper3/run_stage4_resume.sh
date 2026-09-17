#!/bin/sh
# Stage 4 resume after power loss. Sequential, single-process, no pipes on the runner.
# Completed packages (capture M2/M3, plane M2) are verified complete and are NOT re-run.
# Every step re-runs the integrity repair first, so a second interruption self-heals.
set -e
cd /d/INNOCREW/Blockage/paper3
Q=Qwen/Qwen2.5-3B-Instruct
L=meta-llama/Llama-3.2-3B-Instruct

guard () {
  if MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -e bash -lc "pgrep -f stage4_run.py" >/dev/null 2>&1; then
    echo "ABORT: a stage4_run.py process is already running -- refusing to start a second writer"
    exit 1
  fi
  python stage4_repair.py
}

run () {   # package model n
  guard
  echo "=== $(date +%H:%M:%S)  $1  $2  n=$3 ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage4_run.py "$1" --model "$2" --n "$3" \
      > "out/_s4_$1_$(echo $2 | tr '/' '_').log" 2>&1
  tail -2 "out/_s4_$1_$(echo $2 | tr '/' '_').log"
}

run plane   "$L" 100
run anchors "$Q" 100
run anchors "$L" 100
run kaxis   "$Q" 100
run kaxis   "$L" 100
python stage4_repair.py
echo "=== STAGE 4 COMPLETE $(date +%H:%M:%S) ==="
