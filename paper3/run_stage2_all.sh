#!/bin/sh
# Stage 2, strictly sequential -- Machine N is single-process and stays that way.
set -e
cd /d/INNOCREW/Blockage/paper3
Q=Qwen/Qwen2.5-3B-Instruct
L=meta-llama/Llama-3.2-3B-Instruct
for step in \
  "p24 $L" \
  "p21 $Q" "p21 $L" \
  "p22 $Q" "p22 $L" \
  "p23 $Q" "p23 $L" ; do
  set -- $step
  echo "=== $(date +%H:%M:%S)  probe $1  model $2 ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage2_run.py "$1" --model "$2" --n 50 2>&1 | grep -v "^Loading" || exit 1
done
for m in "$Q" "$L"; do
  echo "=== $(date +%H:%M:%S)  capture  $m ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage2_capture.py --model "$m" --n 50 2>&1 | grep -v "^Loading" || exit 1
done
echo "=== ALL STAGE 2 RUNS COMPLETE $(date +%H:%M:%S) ==="
