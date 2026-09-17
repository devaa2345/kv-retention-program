#!/bin/bash
# Starts Llama Phase 1 once the Llama sweep and the 3B Phase 1 have both finished.
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/kxrx26/quant-rocm/bin/python
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
lg "waiting for llama sweep + 3B phase1"
while true; do
  SW=$($PY -c "
import glob
n=0
for p in glob.glob('results_llama/bits*/sweep_bits*.jsonl'):
    n+=sum(1 for l in open(p) if l.strip())
print(n)" 2>/dev/null || echo 0)
  P3=$($PY -c "print(sum(1 for l in open('results3b/bits8/phase1_8bit.jsonl') if l.strip()))" 2>/dev/null || echo 0)
  [ "$SW" -ge 300 ] && [ "$P3" -ge 2693 ] && break
  sleep 30
done
lg "prerequisites done (sweep=$SW, 3B phase1=$P3); starting Llama Phase 1"
$PY run_phase1_llama.py 150 2>&1 | grep -v "^Loading" | tail -20
lg "=== LLAMA PHASE 1 COMPLETE ==="
