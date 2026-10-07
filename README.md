# Fireworks vs. self-hosted evaluation kit

A small, honest harness to evaluate the **technicals** of the Fireworks platform
against a self-hosted baseline you control. Two scripts, one design principle:
vLLM (local), Fireworks, and frontier APIs all speak the **OpenAI-compatible**
protocol, so a single command runs against any of them by swapping
`--base-url` / `--api-key` / `--model`. No double implementation, no apples-to-oranges.

It produces three artifacts that answer the "plan, not proof" critique:
1. A serving benchmark you ran yourself (TTFT, p95, throughput, cost).
2. A fine-tune that beats the base model on a real eval.
3. A defensible "when an owned model / self-hosting is the *wrong* call" finding
   (on this hardware, it tends to fall out of the data on its own).

> **What can and can't run locally.** FireAttention and FireOptimizer are Fireworks'
> hosted engines - they do not install locally. Your box builds the **baseline**
> (vLLM serving + your own LoRA fine-tune); Fireworks runs FireOptimizer + FireAttention.
> The *delta between them* is the result.

---

## Your hardware: NVIDIA DGX Spark (GB10 Grace Blackwell)

Single GB10 superchip, Blackwell GPU (compute capability 12.1), 20-core Arm CPU,
**128 GB unified LPDDR5x memory** shared by CPU and GPU. Key implications:

- **`nvidia-smi` shows memory `[N/A]`** - that's expected; the memory is unified, not
  dedicated VRAM. Check the pool with `free -h` (~128 GB).
- **FP8 *and* NVFP4 both supported** (5th-gen Tensor Cores). The old "FP8 needs Hopper"
  caveat does not apply - benchmark both precisions, including NVFP4.
- **The limiter is memory bandwidth: ~273 GB/s** (vs. ~3.35 TB/s HBM on an H100). The
  Spark is compute-rich, bandwidth-poor, so **decode is slow** - public tests show
  Llama 3.1 70B (FP8) at ~803 tok/s prefill but only ~2.7 tok/s decode. It is a
  prototyping / fine-tuning / experimentation box, not a serving-throughput machine.
- **aarch64 (Arm) + Blackwell (sm_121).** Use Arm-native wheels/containers and a recent,
  Blackwell-aware vLLM (CUDA 12.8+). NVIDIA's DGX Spark playbooks / NIM containers are
  the smooth path if a build fights you. SGLang is also known-good on this box.
- **Cost framing:** the Spark is capex dev hardware, not a serving fleet. Compare
  **throughput/latency** (Spark dev box vs. Fireworks production serving); treat
  Fireworks cost as opex per-token. Don't try to force a $/token parity claim locally.

This is a strong story, not a weak one: *"I built and fine-tuned on a Blackwell DGX
Spark, validated FP8 and NVFP4 locally, then benchmarked production serving against
Fireworks' FireAttention - and the 273 GB/s bandwidth gap is exactly why you don't
self-host a latency-sensitive production loop."*

---

## The three-leg experiment

| Leg | Where | What you measure |
|-----|-------|------------------|
| 1. Self-hosted serving baseline | DGX Spark (vLLM) | TTFT, ITL, p95, throughput |
| 2. Hand-rolled fine-tune | DGX Spark (PEFT / TRL / Axolotl) | eval lift, effort |
| 3. Managed path | Fireworks (FireOptimizer + FireAttention) | eval lift, latency, cost/1M tok |

Plus a frontier model (e.g. GPT-4o) as the accuracy ceiling and a cost reference.

**Minimum viable version (1-2 days):** one 8B model, one public task with a clean
held-out set, vLLM baseline vs. one FireOptimizer SFT on FireAttention - latency +
accuracy (+ Fireworks cost). The hand-rolled fine-tune and RFT leg can come second.

---

## Install

```bash
pip install -r requirements.txt          # openai, numpy
```

## How to perform the kit adjustments (DGX Spark)

Two helper scripts do the work. Edit the variables at the top of each.

### 1. Start the local vLLM server (terminal A)

```bash
source ~/vllm-env/bin/activate     # your existing env
./serve_spark.sh fp8               # profiles: bf16 | fp8 | nvfp4
```

What the script runs under the hood (single-unit Spark = tensor-parallel 1):

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 \
     --tensor-parallel-size 1 --quantization fp8
```

- **bf16** - no quantization, the honest "full precision" point.
- **fp8** - dynamic FP8; works on Blackwell, good default.
- **nvfp4** - point `MODEL` at a pre-quantized NVFP4 checkpoint and use the matching
  `--quantization` value from its model card (commonly `modelopt_fp4`). NVFP4 paths are
  newer, so prefer a recent vLLM or an NVIDIA container.

Use an **8B** model for the throughput/latency runs (decode is bandwidth-bound, so 70B+
will crawl). The Spark can *hold* up to ~200B params at FP4 - save those for the
accuracy/eval leg where speed doesn't matter.

### 2. Run the benchmarks + evals (terminal B)

```bash
export FIREWORKS_API_KEY=fw_xxx
source ~/vllm-env/bin/activate
./run_spark.sh
```

`run_spark.sh` runs, against both the local Spark endpoint and Fireworks:
- `bench.py` - TTFT, ITL, p95, throughput (Fireworks run also reports cost/1M tok)
- `eval.py` - classification accuracy on the sample set

To compare precisions, restart `serve_spark.sh` with `bf16`, then `fp8`, then `nvfp4`,
re-running the local benchmark each time (rename the output files).

### 3. Build the SFT data, then fine-tune

First convert your labeled rows into Fireworks chat format with a clean, deduped split.
`to_sft.py` dedups on normalized input, then splits - so a training row can never leak
into the eval set:

```bash
python to_sft.py --in labeled.jsonl \
  --system "Classify the support ticket into exactly one of: billing, technical, account, other. Reply with ONLY the label." \
  --eval-frac 0.2 --seed 7
# -> train.jsonl (SFT) | eval_sft.jsonl (frozen held-out, SFT) | eval_grader.jsonl (for eval.py)
```

- **Fireworks (FireOptimizer):**
  ```bash
  firectl create dataset triage-train ./train.jsonl
  firectl create dataset triage-eval  ./eval_sft.jsonl
  firectl create sftj --base-model accounts/fireworks/models/llama-v3p1-8b-instruct \
    --dataset triage-train --evaluation-dataset triage-eval \
    --output-model triage-8b --epochs 2 --lora-rank 16
  firectl get sftj <JOB_ID>                                   # monitor
  firectl create deployment accounts/<you>/models/triage-8b   # or deploy serverless
  ```
  Then score base vs. fine-tuned on the SAME held-out rows:
  ```bash
  python eval.py --base-url https://api.fireworks.ai/inference/v1 --api-key "$FIREWORKS_API_KEY" \
    --model accounts/<you>/models/triage-8b --data eval_grader.jsonl \
    --task classification \
    --system "Classify the support ticket into exactly one of: billing, technical, account, other. Reply with ONLY the label." \
    --labels billing technical account other --output eval_fireworks_ft.json
  ```
  (Run `firectl create sftj --help` to confirm flag names on your CLI version.)
- **DGX Spark (hand-rolled):** LoRA/QLoRA with PEFT + TRL or Axolotl on `train.jsonl`;
  serve the adapter on vLLM and run the same eval against `eval_grader.jsonl`. This leg
  proves you can build an owned model, not just describe one. (Training is
  bandwidth-tolerant, so the Spark is well-suited to it even though serving throughput
  is modest.)

Both paths train on the **same `train.jsonl`** and are scored on the **same frozen
`eval_grader.jsonl`** - apples to apples, no contamination.

---

## Choosing a dataset, and keeping the eval set clean

Pick a task that mirrors the owned-model thesis - a narrow, high-volume classification
or extraction job where a small fine-tune can plausibly match a frontier model cheaply.
Good public starts: **Banking77**, **CLINC150**, or a redacted internal labeled set.

Contamination discipline (what a technical reviewer checks):
- **Never** fine-tune on the eval split. Split first, freeze the eval set, then train.
- De-duplicate across splits (exact and near-duplicate) so test items don't leak.
- Prefer a **recent or private** eval set the base model is unlikely to have memorized.
- Fix sampling: `temperature=0`, identical system prompt, identical `max_tokens` on
  every endpoint, so differences are the model/serving stack, not the harness.

---

## The comparison table to produce

| Setup | Precision | Accuracy | TTFT p95 (s) | Decode tok/s | $ / 1M out tok |
|-------|-----------|---------:|-------------:|-------------:|---------------:|
| Frontier API (GPT-4o)             | -    |  |  |  |  |
| Base 8B - DGX Spark / vLLM        | fp8  |  |  |  | capex |
| Base 8B - DGX Spark / vLLM        | nvfp4|  |  |  | capex |
| Base 8B - Fireworks               | -    |  |  |  |  |
| Fine-tuned - DGX Spark / vLLM     | fp8  |  |  |  | capex |
| Fine-tuned - Fireworks (FireAttention) | - |  |  |  |  |

Expect the Spark's decode tok/s to trail Fireworks substantially - that gap is the
point, not a flaw in your test.

## Fairness checklist

- Same prompts, same sampling params, same `max_tokens` on every endpoint.
- Warm up before timing (`--warmup`); report steady-state concurrency.
- Run each config 3x and report the median; state GPU (GB10), precision, vLLM version.
- Say plainly which numbers are measured vs. estimated, and that local cost is capex.

---

## Troubleshooting - DGX Spark: `ImportError: libcudart.so.12`

**Cause.** The DGX Spark ships **CUDA 13** (only `libcudart.so.13` exists), but
pip-installed vLLM's pre-compiled kernels are built against **CUDA 12** and look for
`libcudart.so.12`. Known GB10/sm_121 issue - not a kit problem.

Confirm:
```bash
nvcc --version | grep release                       # expect 13.x
find /usr/local -name 'libcudart.so*' 2>/dev/null    # expect only ...so.13
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

**Fix A - quick symlink (≈1 min; unblocks the import):**
```bash
CUDA_LIB=$(dirname "$(find /usr/local -name 'libcudart.so.13' 2>/dev/null | head -1)")
sudo ln -sf "$CUDA_LIB/libcudart.so.13" "$CUDA_LIB/libcudart.so.12"
export LD_LIBRARY_PATH="$CUDA_LIB:$LD_LIBRARY_PATH"
./serve_spark.sh fp8
```
If a follow-on `sm_121a` / CUDA-graph error appears, disable CUDA graphs:
`EAGER=1 ./serve_spark.sh fp8`  (~20-30% slower, but runs).

**Fix B - reliable (matched CUDA-13 stack; documented working path for GB10):**
```bash
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130
```
The "GPU capability 12.1 > max 12.0" warning is harmless - sm_120 binaries run on sm_121.

**Fix C - turnkey:** NVIDIA's NGC PyTorch container (the validated ARM64+CUDA PyTorch for
Blackwell), or a community one-command DGX Spark vLLM source installer, if A/B fight you.

> The kit's `serve_spark.sh` already exports `LD_LIBRARY_PATH=/usr/local/cuda/lib64` and
> supports `EAGER=1`. Override the lib path with `CUDA_LIB=/path/to/lib64 ./serve_spark.sh`.
