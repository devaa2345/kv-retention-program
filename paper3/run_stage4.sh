#!/bin/sh
# Stage 4, sequential, single-process. No pipes on the runner (Stage 2 lesson).
set -e
cd /d/INNOCREW/Blockage/paper3
Q=Qwen/Qwen2.5-3B-Instruct
L=meta-llama/Llama-3.2-3B-Instruct
run () {   # package model n
  echo "=== $(date +%H:%M:%S)  $1  $2  n=$3 ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage4_run.py "$1" --model "$2" --n "$3" \
      > "out/_s4_$1_$(echo $2 | tr '/' '_').log" 2>&1
  tail -2 "out/_s4_$1_$(echo $2 | tr '/' '_').log"
}
# captures at the FULL registered n -- these carry the registered prediction scoring
run capture "$Q" 200
run capture "$L" 200
# the load-bearing plane
run plane "$Q" 100
run plane "$L" 100
# ladder anchors and tripwires
run anchors "$Q" 100
run anchors "$L" 100
# the k axis
run kaxis "$Q" 100
run kaxis "$L" 100
echo "=== STAGE 4 COMPLETE $(date +%H:%M:%S) ==="
