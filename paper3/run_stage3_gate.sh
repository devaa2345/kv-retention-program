#!/bin/sh
set -e
cd /d/INNOCREW/Blockage/paper3
Q=Qwen/Qwen2.5-3B-Instruct
L=meta-llama/Llama-3.2-3B-Instruct
echo "=== $(date +%H:%M:%S) shortcuts M3 ==="
MSYS_NO_PATHCONV=1 ./wsl.sh stage3_gate.py --model "$L" --n 200 --shortcuts > out/_gate_sc_M3.log 2>&1
tail -12 out/_gate_sc_M3.log
for m in "$Q" "$L"; do
  echo "=== $(date +%H:%M:%S) anchors $m ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage3_gate.py --model "$m" --n 24 --anchors > out/_gate_a_$(echo $m | tr '/' '_').log 2>&1
  tail -3 out/_gate_a_$(echo $m | tr '/' '_').log
done
echo "=== STAGE3 GATE RUNS COMPLETE $(date +%H:%M:%S) ==="
