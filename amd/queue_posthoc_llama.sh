#!/bin/bash
# Runs the Llama post-hoc as soon as a slot frees. It occupies one of the two slots, so the
# scheduler holds Llama Phase 1 half B until it finishes -- no 3-job overlap is possible.
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kxrx26/quant-rocm/bin/python
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
while true; do
  N=$($PY - <<'P'
import glob
n=0
for d in glob.glob('/proc/[0-9]*'):
    try: c=open(d+'/cmdline','rb').read().decode('utf8','ignore')
    except Exception: continue
    if '/bin/python run_' in c and ' -c' not in c[:60]: n+=1
print(n)
P
)
  [ "$N" -lt 2 ] && break
  sleep 15
done
lg "slot free; running Llama post-hoc"
$PY run_posthoc_llama.py 2>&1 | grep -v "^Loading"
lg "=== LLAMA POSTHOC DONE ==="
