#!/bin/sh
# Paper 4 Stage 3, resume after amendment A3. c=1 data already exists and is NOT regenerated.
# Runs c=8, 19, 40 on both models, then the final report. Sequential, single process, no pipes.
cd /d/INNOCREW/Blockage/paper4
Q=Qwen/Qwen2.5-3B-Instruct
L=meta-llama/Llama-3.2-3B-Instruct

guard () {
  if MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -e bash -lc "pgrep -f 'stage[0-9]_run.py|stage3_preflight.py'" >/dev/null 2>&1; then
    echo "ABORT: a runner is already running -- refusing to start a second writer"; exit 1
  fi
}

run () {  # model c
  guard
  log="out/_p4_s3_c$2_$(echo $1 | tr '/' '_').log"
  echo "=== $(date +%H:%M:%S) grid $1 c=$2 ==="
  MSYS_NO_PATHCONV=1 ./wsl.sh stage3_run.py --model "$1" --c $2 --n 50 > "$log" 2>&1
  rc=$?; grep -E "CELL DONE|CELL INCOMPLETE|CONFOUND WATCH|DONE " "$log" | tail -8
  if [ $rc -ne 0 ]; then echo "ABORT: grid $1 c=$2 exited $rc"; exit $rc; fi
}

for c in 8 19 40; do
  run "$Q" $c
  run "$L" $c
done

MSYS_NO_PATHCONV=1 ./wsl.sh stage3_analyse.py control > out/_p4_s3_c1_diagnostic.log 2>&1
MSYS_NO_PATHCONV=1 ./wsl.sh stage3_analyse.py report > out/_p4_s3_report.log 2>&1
echo "=== STAGE 3 COMPLETE $(date +%H:%M:%S) ==="
