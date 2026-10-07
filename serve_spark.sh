#!/usr/bin/env bash
# serve_spark.sh - start a vLLM OpenAI-compatible server on the DGX Spark (GB10 / Blackwell).
# Run in its own terminal and leave it up while you benchmark.
#
#   source ~/vllm-env/bin/activate
#   ./serve_spark.sh fp8           # profiles: bf16 | fp8 | nvfp4
#   EAGER=1 ./serve_spark.sh fp8   # if CUDA-graph / sm_121 errors appear (slower, but runs)
#
# Single GB10 = tensor-parallel 1. Use an 8B model for meaningful decode throughput;
# the Spark can hold far larger models but decode is bandwidth-bound (~273 GB/s).
# If you hit `libcudart.so.12: cannot open shared object file`, see the
# "Troubleshooting - DGX Spark" section of the README first.
set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
PROFILE="${1:-fp8}"

# DGX Spark ships CUDA 13; make sure the runtime is on the loader path.
export LD_LIBRARY_PATH="${CUDA_LIB:-/usr/local/cuda/lib64}:${LD_LIBRARY_PATH:-}"

case "$PROFILE" in
  bf16)  ARGS="" ;;
  fp8)   ARGS="--quantization fp8" ;;
  nvfp4)
    echo "[nvfp4] Point MODEL at a pre-quantized NVFP4 checkpoint and confirm the"
    echo "        --quantization value on its model card (commonly modelopt_fp4)."
    ARGS="--quantization modelopt_fp4" ;;
  *) echo "usage: $0 [bf16|fp8|nvfp4]"; exit 1 ;;
esac

# Bleeding-edge Blackwell builds can fail on CUDA graphs; EAGER=1 falls back (slower).
[ "${EAGER:-0}" = "1" ] && ARGS="$ARGS --enforce-eager"

echo "Serving: $MODEL"
echo "Profile: $PROFILE   Port: $PORT   TP: 1   Eager: ${EAGER:-0}"
# 'vllm serve' is equivalent to: python -m vllm.entrypoints.openai.api_server --model ...
exec vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 $ARGS
