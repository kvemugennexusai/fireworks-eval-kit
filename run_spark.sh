#!/usr/bin/env bash
# run_spark.sh - benchmark + eval against your local DGX Spark vLLM server AND Fireworks.
# Start the server first in another terminal:  ./serve_spark.sh fp8
#
#   export FIREWORKS_API_KEY=fw_xxx
#   source ~/vllm-env/bin/activate
#   ./run_spark.sh
set -euo pipefail

# ---- edit these ----
LOCAL_MODEL="${LOCAL_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LOCAL_URL="${LOCAL_URL:-http://localhost:8000/v1}"
FW_URL="https://api.fireworks.ai/inference/v1"
FW_MODEL="${FW_MODEL:-accounts/fireworks/models/llama-v3p1-8b-instruct}"
: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY in your environment}"
N="${N:-300}"; C="${C:-32}"; MAXTOK="${MAXTOK:-256}"
TAG="${TAG:-fp8}"   # label for output files, e.g. the precision you're serving
SYS="Classify the support ticket into exactly one of: billing, technical, account, other. Reply with ONLY the label."
LABELS="billing technical account other"
# --------------------

echo "### 1/4  Benchmark - DGX Spark (vLLM, local, $TAG)"
python bench.py --base-url "$LOCAL_URL" --api-key EMPTY --model "$LOCAL_MODEL" \
  --prompts data/sample_prompts.jsonl --num-requests "$N" --concurrency "$C" \
  --max-tokens "$MAXTOK" --output "results_spark_${TAG}.json"

echo "### 2/4  Benchmark - Fireworks (FireAttention)"
python bench.py --base-url "$FW_URL" --api-key "$FIREWORKS_API_KEY" --model "$FW_MODEL" \
  --prompts data/sample_prompts.jsonl --num-requests "$N" --concurrency "$C" \
  --max-tokens "$MAXTOK" --price-in 0.20 --price-out 0.20 --output results_fireworks.json

echo "### 3/4  Eval (accuracy) - DGX Spark base"
python eval.py --base-url "$LOCAL_URL" --api-key EMPTY --model "$LOCAL_MODEL" \
  --data data/sample_eval.jsonl --task classification --system "$SYS" \
  --labels $LABELS --output "eval_spark_${TAG}.json"

echo "### 4/4  Eval (accuracy) - Fireworks base"
python eval.py --base-url "$FW_URL" --api-key "$FIREWORKS_API_KEY" --model "$FW_MODEL" \
  --data data/sample_eval.jsonl --task classification --system "$SYS" \
  --labels $LABELS --output eval_fireworks_base.json

echo
echo "Done. Outputs: results_spark_${TAG}.json, results_fireworks.json, eval_spark_${TAG}.json, eval_fireworks_base.json"
echo "After fine-tuning (FireOptimizer), re-run with FW_MODEL=accounts/<you>/models/<your-ft>."
