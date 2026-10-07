#!/usr/bin/env python3
"""
eval.py - task accuracy / quality eval against an OpenAI-compatible endpoint.

Two modes:
  classification : exact / normalized match of model output to a gold label.
  judge          : LLM-as-judge scores answer quality 1-5 vs. a reference + rubric.

Same endpoint-swap trick as bench.py: point --base-url / --model at vLLM (DGX),
Fireworks (base or fine-tuned), or a frontier API, and compare on the same data.

Data format (JSONL):
  classification : {"input": "...", "label": "billing"}
  judge          : {"input": "...", "reference": "ideal answer ..."}   # reference optional

Examples
--------
# Base model on Fireworks, classification
python eval.py \
  --base-url https://api.fireworks.ai/inference/v1 --api-key "$FIREWORKS_API_KEY" \
  --model accounts/fireworks/models/llama-v3p1-8b-instruct \
  --data data/sample_eval.jsonl --task classification \
  --system "Classify the support ticket into exactly one of: billing, technical, account, other. Reply with ONLY the label." \
  --labels billing technical account other --output eval_fw_base.json

# Your fine-tuned model (swap the model id) - run the SAME command otherwise
python eval.py ... --model accounts/<you>/models/<your-ft-model> --output eval_fw_ft.json
"""
import argparse
import asyncio
import json
import os
import re
import sys

import numpy as np
from openai import AsyncOpenAI


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        sys.exit("No eval rows loaded.")
    return rows


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").strip().lower())


def match_label(text, labels):
    t = norm(text)
    for L in labels:                      # exact normalized match
        if t == norm(L):
            return L
    for L in labels:                      # else first label appearing in the text
        if norm(L) and norm(L) in t:
            return L
    return None


JUDGE_SYS = ("You are a strict grader. Given an INPUT, a MODEL ANSWER, and an optional "
             "REFERENCE, score the MODEL ANSWER from 1 (poor) to 5 (excellent) on correctness, "
             "completeness, and faithfulness. Respond with ONLY a JSON object: "
             '{"score": <1-5>, "reason": "<short>"}.')


async def call(client, model, system, user, max_tokens, temperature):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    r = await client.chat.completions.create(model=model, messages=msgs,
                                             max_tokens=max_tokens, temperature=temperature)
    return r.choices[0].message.content or ""


async def judge_one(jclient, jmodel, inp, ans, ref):
    user = f"INPUT:\n{inp}\n\nMODEL ANSWER:\n{ans}\n\nREFERENCE:\n{ref or '(none)'}"
    out = await call(jclient, jmodel, JUDGE_SYS, user, 200, 0.0)
    m = re.search(r"\{.*\}", out, re.S)
    try:
        o = json.loads(m.group(0)) if m else {}
        return float(o.get("score", 0)), o.get("reason", "")
    except Exception:  # noqa: BLE001
        return 0.0, "unparseable judge output"


async def run(args):
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY",
                         timeout=args.timeout)
    jclient, jmodel = client, args.model
    if args.task == "judge" and args.judge_base_url:
        jclient = AsyncOpenAI(base_url=args.judge_base_url,
                              api_key=args.judge_api_key or args.api_key or "EMPTY",
                              timeout=args.timeout)
        jmodel = args.judge_model or args.model

    rows = load(args.data)
    sem = asyncio.Semaphore(args.concurrency)

    async def work(row):
        async with sem:
            inp = row.get("input") or row.get("prompt") or ""
            try:
                ans = await call(client, args.model, args.system, inp,
                                 args.max_tokens, args.temperature)
            except Exception as e:  # noqa: BLE001
                return {"input": inp, "error": f"{type(e).__name__}: {e}"}
            rec = {"input": inp, "output": ans}
            if args.task == "classification":
                gold = row.get("label")
                pred = match_label(ans, args.labels) if args.labels else norm(ans)
                rec.update({"gold": gold, "pred": pred,
                            "correct": norm(pred or "") == norm(gold or "")})
            else:
                score, reason = await judge_one(jclient, jmodel, inp, ans, row.get("reference"))
                rec.update({"reference": row.get("reference"), "score": score, "reason": reason})
            return rec

    results = await asyncio.gather(*[work(r) for r in rows])
    errs = [r for r in results if "error" in r]
    good = [r for r in results if "error" not in r]

    print("\n=== Eval summary ===")
    print(f"endpoint : {args.base_url}")
    print(f"model    : {args.model}")
    print(f"task     : {args.task}   examples: {len(good)} ok / {len(errs)} err")
    summary = {"endpoint": args.base_url, "model": args.model, "task": args.task,
               "n": len(good), "errors": len(errs)}
    if args.task == "classification" and good:
        acc = sum(1 for r in good if r["correct"]) / len(good)
        summary["accuracy"] = round(acc, 4)
        print(f"accuracy : {acc*100:.2f}%")
    elif good:
        scores = [r["score"] for r in good if r["score"] > 0]
        mean = float(np.mean(scores)) if scores else 0
        summary["mean_score"] = round(mean, 3)
        summary["scored"] = len(scores)
        print(f"mean judge score (1-5): {mean:.2f}  over {len(scores)} graded")
    if errs:
        print(f"errors   : {len(errs)} (first: {errs[0]['error']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print(f"wrote {args.output}")


def main():
    ap = argparse.ArgumentParser(
        description="Task eval (classification / LLM-judge) over an OpenAI-compatible endpoint.")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key",
                    default=os.getenv("OPENAI_API_KEY") or os.getenv("FIREWORKS_API_KEY") or "EMPTY")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="JSONL: {'input','label'} or {'input','reference'}")
    ap.add_argument("--task", choices=["classification", "judge"], default="classification")
    ap.add_argument("--system", default="", help="system prompt string (use $(cat file) for a file)")
    ap.add_argument("--labels", nargs="*", help="allowed labels (classification)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--judge-base-url")
    ap.add_argument("--judge-api-key")
    ap.add_argument("--judge-model")
    ap.add_argument("--output")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
