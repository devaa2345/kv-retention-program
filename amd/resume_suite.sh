#!/bin/bash
cd "/home/kxrx26/research test"
# wait for the determinism check to finish
while pgrep -f check_determinism.py > /dev/null; do sleep 5; done
sleep 3
if [ -s /tmp/detC.json ] && [ -s /tmp/detD.json ]; then
  python3 -c "
import json,sys
c=json.load(open('/tmp/detC.json')); d=json.load(open('/tmp/detD.json')); a=json.load(open('/tmp/detA.json'))
print('two-process determinism (new code):', c==d)
print('retained sets identical to PRE-OPTIMISATION code:', c==a)
sys.exit(0 if c==d else 1)
" || { echo "DETERMINISM FAILED - NOT RESUMING"; exit 1; }
fi
echo "[$(date +%H:%M:%S)] resuming suite with optimised engine"
nohup ./run_all.sh > logs/run_all2.log 2>&1 &
sleep 5
nohup ./checkpoint_loop.sh > logs/checkpoint_loop2.log 2>&1 &
echo "[$(date +%H:%M:%S)] suite + checkpoint loop relaunched"
