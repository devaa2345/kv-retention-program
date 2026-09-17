#!/usr/bin/env bash
# Re-ordered GPU queue. N9 M3 resumes from disk (key_digest dedup), then the two
# requested items, then the low-priority N9 M2 (2 KV heads, structurally degenerate).
set -uo pipefail
cd /mnt/d/INNOCREW/Blockage/paper2
export HF_HOME=/mnt/d/hf_cache
PY=/opt/p2venv/bin/python
LADDER=full_cache,null,random,floor_pos,oracle_causal,oracle_prescient

echo "########## [1/5] N9 M3 Llama-3.2-3B (resume) ##########"; date +%H:%M
"$PY" n9_decomposition.py --model meta-llama/Llama-3.2-3B-Instruct \
  --budgets 16,32,64,128,256,512 --n 200 \
  --out runs/nvidia/n9_M3_ledger.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true

echo "########## [2/5] fragmentation units M2 (N=200) ##########"; date +%H:%M
"$PY" bench/fragmentation_units.py Qwen/Qwen2.5-3B-Instruct 200 16,32,64,128,256,512 \
  2>&1 | grep -vE 'Loading weights|it/s' || true

echo "########## [3/5] fragmentation units M3 (N=200) ##########"; date +%H:%M
"$PY" bench/fragmentation_units.py meta-llama/Llama-3.2-3B-Instruct 200 16,32,64,128,256,512 \
  2>&1 | grep -vE 'Loading weights|it/s' || true

echo "########## [4/5] N8 M3 Llama-3.2-3B aware (finish) ##########"; date +%H:%M
"$PY" run_grid_aware.py --model meta-llama/Llama-3.2-3B-Instruct --budgets 128,512 \
  --arms "$LADDER",snapkv,tova,expected_attn,keydiff,adakv_snapkv \
  --n 200 --out runs/nvidia/grid_M3_ledger_aware.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true
"$PY" analyze_aware.py --agnostic runs/nvidia/grid_M3_ledger_agnostic.jsonl \
  --aware runs/nvidia/grid_M3_ledger_aware.jsonl --label "N8 M3 Llama-3.2-3B aware" \
  --out runs/nvidia/analysis_aware_M3.json 2>&1 | grep -v 'it/s' || true

echo "########## [5/5] N9 M2 Qwen2.5-3B (2 KV heads) ##########"; date +%H:%M
"$PY" n9_decomposition.py --model Qwen/Qwen2.5-3B-Instruct \
  --budgets 16,32,64,128,256,512 --n 200 \
  --out runs/nvidia/n9_M2_ledger.jsonl 2>&1 | grep -vE 'Loading weights|it/s' || true
echo "########## chain complete ##########"; date +%H:%M
