#!/usr/bin/env bash
# N5-N7 main grid. M2 COMPLETE first, then M3 - an interruption leaves one model finished.
# Arms = 6 ladder + that model's admitted Tier-A methods (N4a).
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
LADDER=full_cache,null,random,floor_pos,oracle_causal,oracle_prescient

echo "########## N6  M2 Qwen2.5-3B  (C=32,64,128,256,512) ##########"
date +%H:%M
"$PY" run_grid.py --model Qwen/Qwen2.5-3B-Instruct --budgets 32,64,128,256,512 \
  --arms "$LADDER",snapkv,expected_attn,keydiff,adakv_snapkv \
  --n 200 --out runs/nvidia/grid_M2_ledger_agnostic.jsonl 2>&1 \
  | grep -vE 'Loading weights|it/s' || true

echo "########## N7  M3 Llama-3.2-3B  (C=16,32,64,128,256,512) ##########"
date +%H:%M
"$PY" run_grid.py --model meta-llama/Llama-3.2-3B-Instruct --budgets 16,32,64,128,256,512 \
  --arms "$LADDER",snapkv,tova,expected_attn,keydiff,adakv_snapkv \
  --n 200 --out runs/nvidia/grid_M3_ledger_agnostic.jsonl 2>&1 \
  | grep -vE 'Loading weights|it/s' || true
date +%H:%M
