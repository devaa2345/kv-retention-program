#!/bin/bash
# Launches one Llama Phase 1 half as soon as a GPU slot frees (fewer than 2 harness jobs live).
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/kxrx26/quant-rocm/bin/python
S0=$1; S1=$2; TAG=$3
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
lg "half $TAG waiting for a slot (seeds $S0-$S1)"
while true; do
  N=$($PY -c "
import glob
n=0
for d in glob.glob('/proc/[0-9]*'):
    try: c=open(d+'/cmdline','rb').read().decode('utf8','ignore')
    except Exception: continue
    if 'bin/python run_' in c: n+=1
print(n)")
  [ "$N" -lt 2 ] && break
  sleep 20
done
lg "slot free (live=$N); starting Llama Phase 1 half $TAG"
$PY run_phase1_llama.py 150 $S0 $S1 2>&1 | grep -v "^Loading" | tail -12
lg "=== LLAMA PHASE1 HALF $TAG COMPLETE ==="
