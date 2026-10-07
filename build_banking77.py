#!/usr/bin/env python3
"""Build labeled.jsonl from Banking77 train split: {"input": query, "label": intent_name}."""
import json
import random
import sys

from datasets import load_dataset

CANDIDATES = ["PolyAI/banking77", "banking77", "legacy-datasets/banking77"]

ds = None
for ds_id in CANDIDATES:
    try:
        ds = load_dataset(ds_id, split="train")
        print(f"loaded: {ds_id}")
        break
    except Exception as e:  # noqa: BLE001
        print(f"  {ds_id} failed: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
if ds is None:
    sys.exit("could not load Banking77 from any known namespace")

# Resolve ClassLabel int -> intent name
label_feat = ds.features["label"]
names = label_feat.names
print(f"intents: {len(names)}  | train rows available: {len(ds)}")

N = 400
SEED = 7
idx = list(range(len(ds)))
random.Random(SEED).shuffle(idx)
idx = idx[:N]

rows = []
for i in idx:
    r = ds[i]
    rows.append({"input": r["text"], "label": names[r["label"]]})

with open("labeled.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

from collections import Counter
dist = Counter(r["label"] for r in rows)
print(f"wrote labeled.jsonl: {len(rows)} rows, {len(dist)} distinct intents")
print("sample row:", json.dumps(rows[0]))
print(f"label int example -> name: 0 = {names[0]!r}")
