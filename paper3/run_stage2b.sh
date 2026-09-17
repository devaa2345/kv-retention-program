#!/bin/sh
# Stage 2B: corrected scatter layout. Sequential, single-process.
# NOTE: no pipe on the runner. The first version piped through `grep -v` and the
# pipeline exit code masked a crash -- the M2 anchor run died at instance 9 on a
# separation assertion and the driver walked on to the next model regardless.
set -e
cd /d/INNOCREW/Blockage/paper3
Q=Qwen/Qwen2.5-3B-Instruct
L=meta-llama/Llama-3.2-3B-Instruct
for step in "anchors $Q" "anchors $L" "p23b $Q" "p23b $L"; do
  set -- $step
  echo "=== $(date +%H:%M:%S)  $1  $2 ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage2_run.py "$1" --model "$2" --n 50 > out/_s2b_$1_$(echo $2 | tr '/' '_').log 2>&1
  echo "    exit=$?  $(tail -1 out/_s2b_$1_$(echo $2 | tr '/' '_').log)"
done
echo "=== STAGE 2B COMPLETE $(date +%H:%M:%S) ==="
