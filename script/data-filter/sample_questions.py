#!/usr/bin/env python3
"""Sample random questions (user turns only) from TESSY-Code-80K.

Streams every part_*.jsonl file and uses reservoir sampling, so the 4 GB
dataset is never fully loaded into memory. Questions are deduplicated by exact
text first (80K rows -> ~37K unique). Writes one JSON object per line:
    {"id": <id_ddm>, "question": <first user message content>}

Usage:
    python3 script/data-filter/sample_questions.py \
        [--input-dir data/tessy-code-80k] \
        [--output data/tessy-rnd-2k.jsonl] \
        [--n 2000] [--seed 42]
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys


def iter_questions(input_dir):
    files = sorted(
        glob.glob(os.path.join(input_dir, "part_*.jsonl")),
        key=lambda p: int(os.path.basename(p)[len("part_"):-len(".jsonl")]),
    )
    if not files:
        sys.exit(f"No part_*.jsonl files found in {input_dir}")
    seen = set()
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                question = next(
                    (t["content"] for t in row["dialogs"] if t.get("role") == "user"),
                    None,
                )
                if not question:
                    continue
                # The 80K rows hold only ~37K unique questions; keep the first
                # occurrence of each so the sample has no repeated prompts.
                key = hashlib.sha256(question.strip().encode("utf-8")).digest()
                if key in seen:
                    continue
                seen.add(key)
                yield {"id": row["id_ddm"], "question": question}


def reservoir_sample(items, n, rng):
    sample = []
    for i, item in enumerate(items):
        if i < n:
            sample.append(item)
        else:
            j = rng.randint(0, i)
            if j < n:
                sample[j] = item
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", default="data/tessy-code-80k")
    parser.add_argument("--output", default="data/tessy-rnd-2k.jsonl")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sample = reservoir_sample(iter_questions(args.input_dir), args.n, rng)
    rng.shuffle(sample)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in sample:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} questions to {args.output}")


if __name__ == "__main__":
    main()
