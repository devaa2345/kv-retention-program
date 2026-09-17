#!/bin/bash
# Runs queued jobs at most MAXJOBS at a time, gated on measured free VRAM.
# Each 3B-class job needs ~8.4 GB; the card has ~25.7 GB, so 2 is safe and 3 OOMs.
# A single scheduler process owns the queue, so the double-claim race that killed
# half B (two independent waiters both seeing one free slot) cannot recur.
cd "/home/kxrx26/research test"
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/kxrx26/quant-rocm/bin/python
MAXJOBS=2
NEED_GB=9.0
lg(){ echo "[$(date +%H:%M:%S)] $*"; }

live(){ $PY - <<'P'
import glob
n=0
for d in glob.glob('/proc/[0-9]*'):
    try: c=open(d+'/cmdline','rb').read().decode('utf8','ignore')
    except Exception: continue
    if '/bin/python run_' in c and ' -c' not in c[:60]: n+=1
print(n)
P
}
freegb(){ $PY -c "
import subprocess,re
o=subprocess.run(['rocm-smi','--showmeminfo','vram'],capture_output=True,text=True).stdout
tot=used=0
for l in o.splitlines():
    if 'GPU[0]' in l and 'Total Memory' in l: tot=int(re.findall(r'(\d+)\s*$',l)[0])
    if 'GPU[0]' in l and 'Total Used' in l: used=int(re.findall(r'(\d+)\s*$',l)[0])
print('%.2f'%((tot-used)/1e9) if tot else '0')"; }

run_when_free(){   # $1=script $2=args $3=label
  while true; do
    N=$(live); F=$(freegb)
    if [ "$N" -lt "$MAXJOBS" ] && [ "$(echo "$F >= $NEED_GB" | bc)" -eq 1 ]; then break; fi
    sleep 20
  done
  lg "starting $3  (live=$N, free=${F}GB)"
  $PY $1 $2 2>&1 | grep -v "^Loading" | tail -8
  lg "finished $3"
}

lg "scheduler up: max $MAXJOBS jobs, need ${NEED_GB}GB free to start one"
run_when_free run_phase1_llama.py "150 0 75"   "llama-phase1-A"
run_when_free run_phase1_llama.py "150 75 150" "llama-phase1-B"
lg "=== SCHEDULER QUEUE COMPLETE ==="
