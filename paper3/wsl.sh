#!/bin/sh
# Run a paper3 script under the pinned Paper 2 toolchain in WSL, single-process.
CMD="cd /mnt/d/INNOCREW/Blockage/paper3 && HF_HOME=/mnt/d/hf_cache PYTHONPATH=/mnt/d/INNOCREW/Blockage/paper3:/mnt/d/INNOCREW/Blockage/paper2 TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 /opt/p2venv/bin/python $*"
exec wsl -d Ubuntu-24.04 -e bash -lc "$CMD"
