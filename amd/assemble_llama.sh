#!/bin/bash
# Assemble the browser-downloaded Llama files into a loadable directory using hardlinks
# (same filesystem, so no extra disk is consumed).
D=/home/kxrx26/Downloads
M=/home/kxrx26/llama32_3b_instruct
lg(){ echo "[$(date +%H:%M:%S)] $*"; }
lg "waiting for shard 1 to finish downloading"
while true; do
  if [ -f "$D/model-00001-of-00002.safetensors" ]; then
    SZ=$(stat -c%s "$D/model-00001-of-00002.safetensors" 2>/dev/null || echo 0)
    PARTS=$(/usr/bin/find "$D" -maxdepth 1 -name "model-00001-of-00002.*.part" 2>/dev/null | wc -l)
    if [ "$SZ" -gt 4900000000 ] && [ "$PARTS" -eq 0 ]; then
      lg "shard 1 complete: $(echo "scale=2;$SZ/1000000000"|bc) GB"; break
    fi
  fi
  sleep 15
done
mkdir -p "$M"
ln -f "$D/config(1).json"                    "$M/config.json"
ln -f "$D/generation_config(1).json"         "$M/generation_config.json"
ln -f "$D/model.safetensors.index(1).json"   "$M/model.safetensors.index.json"
ln -f "$D/tokenizer(1).json"                 "$M/tokenizer.json"
ln -f "$D/tokenizer_config(1).json"          "$M/tokenizer_config.json"
ln -f "$D/special_tokens_map.json"           "$M/special_tokens_map.json"
ln -f "$D/model-00001-of-00002.safetensors"  "$M/model-00001-of-00002.safetensors"
ln -f "$D/model-00002-of-00002.safetensors"  "$M/model-00002-of-00002.safetensors"
lg "assembled at $M"; ls -la "$M" | tail -9
export PYTHONHASHSEED=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "/home/kxrx26/research test"
lg "=== LLAMA VALIDITY + SWEEP ==="
/home/kxrx26/quant-rocm/bin/python run_llama.py 50 "$M" 2>&1 | grep -v "^Loading"
lg "=== LLAMA COMPLETE ==="
