#!/usr/bin/env bash
# Wait for N8/M2-aware to finish, SKIP M3-aware (protocol does not reproduce the
# literature -- see STATUS.md), then run N9 decomposition, M3 first.
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
AW=runs/nvidia/grid_M2_ledger_aware.jsonl

echo "### waiting for M2 aware to reach 4000 records ###"; date +%H:%M
for i in $(seq 1 240); do
  n=$(wc -l < "$AW" 2>/dev/null || echo 0)
  if [ "$n" -ge 4000 ]; then echo "M2 aware complete: $n"; break; fi
  if ! pgrep -f run_grid_aware.py >/dev/null; then echo "aware run exited at $n"; break; fi
  sleep 60
done

echo "### stopping N8 before M3-aware starts ###"; date +%H:%M
pkill -f wsl_n8.sh 2>/dev/null || true
sleep 3
pkill -f run_grid_aware.py 2>/dev/null || true
sleep 10

echo "### N8 M2 aware analysis ###"
"$PY" analyze_aware.py --agnostic runs/nvidia/grid_M2_ledger_agnostic.jsonl \
  --aware "$AW" --label "N8 M2 Qwen2.5-3B aware" \
  --out runs/nvidia/analysis_aware_M2.json 2>&1 | grep -v 'it/s' || true

echo "########## N9 M3 Llama-3.2-3B (8 KV heads) ##########"; date +%H:%M
"$PY" n9_decomposition.py --model meta-llama/Llama-3.2-3B-Instruct \
  --budgets 16,32,64,128,256,512 --n 200 \
  --out runs/nvidia/n9_M3_ledger.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true

echo "########## N9 M2 Qwen2.5-3B (2 KV heads) ##########"; date +%H:%M
"$PY" n9_decomposition.py --model Qwen/Qwen2.5-3B-Instruct \
  --budgets 16,32,64,128,256,512 --n 200 \
  --out runs/nvidia/n9_M2_ledger.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true
date +%H:%M
