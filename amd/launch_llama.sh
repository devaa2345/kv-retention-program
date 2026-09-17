#!/bin/bash
# Waits for the Llama weights to be fully downloaded, then runs the validity check + sweep.
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/kxrx26/quant-rocm/bin/python
LL=/home/kxrx26/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
lg "waiting for Llama weights"
while true; do
  INC=$(/usr/bin/find "$LL" -name "*.incomplete" 2>/dev/null | wc -l)
  ST=$(/usr/bin/find "$LL" -name "*.safetensors" 2>/dev/null | wc -l)
  SZ=$(du -sb "$LL" 2>/dev/null | cut -f1)
  if [ "$ST" -ge 1 ] && [ "$INC" -eq 0 ] && [ "${SZ:-0}" -gt 5000000000 ]; then
    lg "weights complete: $ST safetensors, $(echo "scale=2;$SZ/1000000000"|bc) GB"; break
  fi
  sleep 20
done
sleep 5
lg "=== LLAMA VALIDITY + SWEEP ==="
$PY run_llama.py 50 2>&1 | grep -v "^Loading"
lg "=== LLAMA COMPLETE ==="
