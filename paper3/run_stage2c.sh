#!/bin/sh
set -e
cd /d/INNOCREW/Blockage/paper3
for m in Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  echo "=== $(date +%H:%M:%S)  p23b C=1024  $m ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage2_run.py p23b --model "$m" --n 50 > out/_s2c_$(echo $m | tr '/' '_').log 2>&1
  echo "    exit=$?  $(tail -1 out/_s2c_$(echo $m | tr '/' '_').log)"
done
echo "=== STAGE 2C COMPLETE $(date +%H:%M:%S) ==="
