#!/usr/bin/env python3
"""
bench.py - OpenAI-compatible LLM serving benchmark.

Measures TTFT, inter-token latency (ITL), end-to-end latency, throughput, and
cost against ANY OpenAI-compatible endpoint - self-hosted vLLM / SGLang / TGI on
your DGX, or Fireworks - by swapping --base-url / --api-key / --model. Run the
same command against each and compare apples to apples.

Examples
--------
# vLLM on your DGX
#   python -m vllm.entrypoints.openai.api_server \
#       --model meta-llama/Llama-3.1-8B-Instruct --port 8000
python bench.py \
    --base-url http://localhost:8000/v1 --api-key EMPTY \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompts data/sample_prompts.jsonl \
    --num-requests 200 --concurrency 16 --max-tokens 256 \
    --gpu-hourly 2.50 --gpu-count 1 --output results_vllm.json

# Fireworks (serverless)
python bench.py \
    --base-url https://api.fireworks.ai/inference/v1 --api-key "$FIREWORKS_API_KEY" \
    --model accounts/fireworks/models/llama-v3p1-8b-instruct \
    --prompts data/sample_prompts.jsonl \
    --num-requests 200 --concurrency 16 --max-tokens 256 \
    --price-in 0.20 --price-out 0.20 --output results_fireworks.json
"""
import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from openai import AsyncOpenAI


@dataclass
class Rec:
    ok: bool
    ttft: float = 0.0
    latency: float = 0.0
    out_tokens: int = 0
    in_tokens: int = 0
    err: str = ""


def load_prompts(path):
    msgs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "messages" in o:
                msgs.append(o["messages"])
            elif "prompt" in o:
                msgs.append([{"role": "user", "content": o["prompt"]}])
            elif "input" in o:
                msgs.append([{"role": "user", "content": o["input"]}])
    if not msgs:
        sys.exit("No prompts loaded (expected JSONL with 'messages', 'prompt', or 'input').")
    return msgs


async def one(client, model, messages, max_tokens, temperature, use_usage):
    t0 = time.perf_counter()
    ttft = None
    chunk_count = 0
    in_tok = 0
    comp = 0
    try:
        kwargs = dict(model=model, messages=messages, max_tokens=max_tokens,
                      temperature=temperature, stream=True)
        if use_usage:
            kwargs["stream_options"] = {"include_usage": True}
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                in_tok = chunk.usage.prompt_tokens or 0
                comp = chunk.usage.completion_tokens or 0
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunk_count += 1
        lat = time.perf_counter() - t0
        out_tokens = comp if comp else chunk_count  # usage if available, else approx
        return Rec(True, ttft if ttft is not None else lat, lat, out_tokens, in_tok)
    except Exception as e:  # noqa: BLE001
        return Rec(False, err=f"{type(e).__name__}: {e}")


async def run(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY",
                         timeout=args.timeout)
    prompts = load_prompts(args.prompts)
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(messages):
        async with sem:
            return await one(client, args.model, messages, args.max_tokens,
                             args.temperature, not args.no_usage)

    if args.warmup > 0:
        await asyncio.gather(*[worker(prompts[i % len(prompts)]) for i in range(args.warmup)])

    sel = [prompts[i % len(prompts)] for i in range(args.num_requests)]
    wall0 = time.perf_counter()
    recs = await asyncio.gather(*[worker(m) for m in sel])
    wall = time.perf_counter() - wall0

    ok = [r for r in recs if r.ok]
    errs = [r for r in recs if not r.ok]
    if not ok:
        print("All requests failed. First error:", errs[0].err if errs else "n/a")
        sys.exit(1)

    ttft = np.array([r.ttft for r in ok])
    lat = np.array([r.latency for r in ok])
    out_tok = np.array([r.out_tokens for r in ok])
    in_tok = np.array([r.in_tokens for r in ok])
    itl = np.array([(r.latency - r.ttft) / max(r.out_tokens - 1, 1)
                    for r in ok if r.out_tokens > 1])
    tot_out = int(out_tok.sum())
    tot_in = int(in_tok.sum())
    sys_tput = tot_out / wall if wall else 0.0

    def pct(a, p):
        return float(np.percentile(a, p)) if len(a) else 0.0

    summary = {
        "endpoint": args.base_url, "model": args.model,
        "num_requests": args.num_requests, "concurrency": args.concurrency,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "ok": len(ok), "errors": len(errs), "wall_s": round(wall, 3),
        "requests_per_s": round(len(ok) / wall, 3) if wall else 0,
        "output_tokens_total": tot_out, "input_tokens_total": tot_in,
        "system_throughput_tok_s": round(sys_tput, 1),
        "ttft_s": {"p50": round(pct(ttft, 50), 4), "p95": round(pct(ttft, 95), 4),
                   "p99": round(pct(ttft, 99), 4), "mean": round(float(ttft.mean()), 4)},
        "itl_ms": {"p50": round(pct(itl, 50) * 1000, 2), "p95": round(pct(itl, 95) * 1000, 2),
                   "mean": round(float(itl.mean()) * 1000, 2) if len(itl) else 0},
        "latency_s": {"p50": round(pct(lat, 50), 3), "p95": round(pct(lat, 95), 3),
                      "p99": round(pct(lat, 99), 3), "mean": round(float(lat.mean()), 3)},
    }

    cost = None
    if args.price_in is not None and args.price_out is not None:
        c = tot_in / 1e6 * args.price_in + tot_out / 1e6 * args.price_out
        cost = {"mode": "per_token", "total_usd": round(c, 4),
                "usd_per_1k_req": round(c / len(ok) * 1000, 4),
                "usd_per_1M_output_tok": round(c / (tot_out / 1e6), 4) if tot_out else None}
    elif args.gpu_hourly is not None:
        c = args.gpu_hourly * args.gpu_count * (wall / 3600)
        cost = {"mode": "gpu_time", "gpu_hourly": args.gpu_hourly, "gpu_count": args.gpu_count,
                "total_usd": round(c, 4),
                "usd_per_1M_output_tok": round(c / (tot_out / 1e6), 4) if tot_out else None,
                "note": "Self-host cost is only fair at saturation - run high concurrency."}
    summary["cost"] = cost

    print("\n=== Benchmark summary ===")
    print(f"endpoint   : {args.base_url}")
    print(f"model      : {args.model}")
    print(f"requests   : {len(ok)} ok / {len(errs)} err   concurrency {args.concurrency}")
    print(f"wall       : {wall:.1f}s  |  {len(ok)/wall:.2f} req/s  |  {sys_tput:.0f} tok/s (system)")
    print(f"TTFT  (s)  : p50 {summary['ttft_s']['p50']:.3f}  p95 {summary['ttft_s']['p95']:.3f}  p99 {summary['ttft_s']['p99']:.3f}")
    print(f"ITL   (ms) : p50 {summary['itl_ms']['p50']:.1f}  p95 {summary['itl_ms']['p95']:.1f}")
    print(f"E2E   (s)  : p50 {summary['latency_s']['p50']:.2f}  p95 {summary['latency_s']['p95']:.2f}  p99 {summary['latency_s']['p99']:.2f}")
    if cost:
        if cost["mode"] == "per_token":
            print(f"cost       : ${cost['total_usd']:.4f}  |  ${cost['usd_per_1M_output_tok']}/1M out tok")
        else:
            print(f"cost       : ${cost['total_usd']:.4f}  ({args.gpu_count}x GPU @ ${args.gpu_hourly}/h)  |  ${cost['usd_per_1M_output_tok']}/1M out tok")
    if errs:
        print(f"\nfirst error: {errs[0].err}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "per_request": [r.__dict__ for r in recs]}, f, indent=2)
        print(f"\nwrote {args.output}")


def main():
    ap = argparse.ArgumentParser(
        description="OpenAI-compatible LLM serving benchmark (vLLM / Fireworks / any).")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key",
                    default=os.getenv("OPENAI_API_KEY") or os.getenv("FIREWORKS_API_KEY") or "EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True, help="JSONL with 'messages' | 'prompt' | 'input'")
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--no-usage", action="store_true",
                    help="server lacks streamed usage; count chunks (approx output tokens, no input tokens)")
    ap.add_argument("--price-in", type=float, help="USD per 1M input tokens (hosted endpoints)")
    ap.add_argument("--price-out", type=float, help="USD per 1M output tokens")
    ap.add_argument("--gpu-hourly", type=float, help="USD per GPU-hour (self-host cost)")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--output", help="write detailed JSON here")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
