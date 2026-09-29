#!/bin/sh
# Paper 4 Stage 3 driver. Sequential, single process, no pipes on any runner.
# preflight(seeds, CPU) -> preflight(identity, prefill) -> c=1 both models -> CONTROL GATE
# -> c=8, c=19, c=40 both models -> final report.
cd /d/INNOCREW/Blockage/paper4
Q=Qwen/Qwen2.5-3B-Instruct
L=meta-llama/Llama-3.2-3B-Instruct

guard () {
  if MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -e bash -lc "pgrep -f 'stage[0-9]_run.py|stage3_preflight.py'" >/dev/null 2>&1; then
    echo "ABORT: a runner is already running -- refusing to start a second writer"; exit 1
  fi
}

pre () {  # check
  guard
  echo "=== $(date +%H:%M:%S) preflight $1 ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage3_preflight.py "$1" > "out/_p4_s3_pre_$1.log" 2>&1
  rc=$?; tail -3 "out/_p4_s3_pre_$1.log"
  if [ $rc -ne 0 ]; then echo "ABORT: preflight $1 failed"; exit $rc; fi
}

run () {  # model c
  guard
  log="out/_p4_s3_c$2_$(echo $1 | tr '/' '_').log"
  echo "=== $(date +%H:%M:%S) grid $1 c=$2 ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage3_run.py --model "$1" --c $2 --n 50 > "$log" 2>&1
  rc=$?; grep -E "CELL DONE|CELL INCOMPLETE|DONE " "$log" | tail -6
  if [ $rc -ne 0 ]; then echo "ABORT: grid $1 c=$2 exited $rc"; exit $rc; fi
}

pre seeds
pre identity

run "$Q" 1
run "$L" 1
MSYS_NO_PATHCONV=1 ./wsl.sh stage3_analyse.py control > out/_p4_s3_control.log 2>&1
if [ $? -ne 0 ]; then
  echo "=== c=1 CONTROL FAILED -- STOPPING, no further cells (prereg K1) ==="
  tail -20 out/_p4_s3_control.log
  exit 3
fi
echo "=== c=1 CONTROL PASS $(date +%H:%M:%S) ==="

for c in 8 19 40; do
  run "$Q" $c
  run "$L" $c
done

MSYS_NO_PATHCONV=1 ./wsl.sh stage3_analyse.py report > out/_p4_s3_report.log 2>&1
echo "=== STAGE 3 COMPLETE $(date +%H:%M:%S) ==="
