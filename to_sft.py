#!/usr/bin/env python3
"""
to_sft.py - convert {"input","label"} rows into Fireworks SFT chat-format JSONL,
with a reproducible train/eval split and a dedup / contamination check.

Outputs (all in OpenAI-compatible chat format that Fireworks accepts unchanged):
  train.jsonl       - SFT training data        -> firectl create dataset <id> train.jsonl
  eval_sft.jsonl    - FROZEN held-out eval set  -> firectl create dataset <id> eval_sft.jsonl
                                                   (use as --evaluation-dataset)
  eval_grader.jsonl - same held-out rows as {"input","label"} for this kit's eval.py

Guarantee: train and eval are disjoint. Inputs are de-duplicated (normalized) first,
then split - so a training row can never leak into the eval set.

Example
-------
python to_sft.py --in labeled.jsonl \
  --system "Classify the support ticket into exactly one of: billing, technical, account, other. Reply with ONLY the label." \
  --eval-frac 0.2 --seed 7
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load(path):
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"skip line {i}: {e}", file=sys.stderr)
    return rows


def pick(row, fields):
    for k in fields:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Convert labeled rows to Fireworks SFT JSONL with a clean train/eval split.")
    ap.add_argument("--in", dest="inp", required=True, help="input JSONL with input + label/target")
    ap.add_argument("--system", default="", help="system prompt to embed (match your eval-time prompt)")
    ap.add_argument("--input-field", default=None, help="input field name (default: input|prompt)")
    ap.add_argument("--target-field", default=None, help="target field name (default: label|output|reference)")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-prefix", default="", help="prefix for output filenames")
    ap.add_argument("--no-stratify", action="store_true", help="random split instead of per-label stratified")
    args = ap.parse_args()

    rows = load(args.inp)
    if not rows:
        sys.exit("no rows loaded")
    in_fields = [args.input_field] if args.input_field else ["input", "prompt"]
    tgt_fields = [args.target_field] if args.target_field else ["label", "output", "reference"]

    seen = set()
    items = []
    dropped_incomplete = 0
    dropped_dup = 0
    for r in rows:
        inp = pick(r, in_fields)
        tgt = pick(r, tgt_fields)
        if inp is None or tgt is None:
            dropped_incomplete += 1
            continue
        key = norm(str(inp))
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        items.append((str(inp), str(tgt)))

    if len(items) < 4:
        sys.exit(f"only {len(items)} usable rows after dedup; need more to split")

    rnd = random.Random(args.seed)
    eval_set, train_set = [], []
    if args.no_stratify:
        idx = list(range(len(items)))
        rnd.shuffle(idx)
        n_eval = max(1, round(len(items) * args.eval_frac))
        ev = set(idx[:n_eval])
        for i, it in enumerate(items):
            (eval_set if i in ev else train_set).append(it)
    else:
        by = defaultdict(list)
        for it in items:
            by[it[1]].append(it)
        for _, group in by.items():
            rnd.shuffle(group)
            n_eval = max(1, round(len(group) * args.eval_frac)) if len(group) > 1 else 0
            eval_set += group[:n_eval]
            train_set += group[n_eval:]

    tr_keys = {norm(i) for i, _ in train_set}
    ev_keys = {norm(i) for i, _ in eval_set}
    overlap = tr_keys & ev_keys
    assert not overlap, f"CONTAMINATION: {len(overlap)} inputs in both train and eval"

    def sft(inp, tgt):
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.append({"role": "user", "content": inp})
        msgs.append({"role": "assistant", "content": tgt})
        return {"messages": msgs}

    p = args.out_prefix
    with open(f"{p}train.jsonl", "w") as f:
        for i, t in train_set:
            f.write(json.dumps(sft(i, t)) + "\n")
    with open(f"{p}eval_sft.jsonl", "w") as f:
        for i, t in eval_set:
            f.write(json.dumps(sft(i, t)) + "\n")
    with open(f"{p}eval_grader.jsonl", "w") as f:
        for i, t in eval_set:
            f.write(json.dumps({"input": i, "label": t}) + "\n")

    print("=== to_sft ===")
    print(f"loaded {len(rows)}  | dropped incomplete {dropped_incomplete}  dup {dropped_dup}")
    print(f"usable {len(items)}  -> train {len(train_set)}  eval {len(eval_set)}  "
          f"(eval_frac~{args.eval_frac}, seed {args.seed})")
    print(f"train/eval disjoint: OK ({len(overlap)} overlap)")
    print("train label dist :", dict(Counter(t for _, t in train_set)))
    print("eval  label dist :", dict(Counter(t for _, t in eval_set)))
    print(f"wrote {p}train.jsonl, {p}eval_sft.jsonl, {p}eval_grader.jsonl")


if __name__ == "__main__":
    main()
