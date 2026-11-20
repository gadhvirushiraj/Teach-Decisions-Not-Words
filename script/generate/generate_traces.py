#!/usr/bin/env python3
"""Step 2: generate thinking traces from Qwen3-4B with vLLM.

For every question in the input JSONL ({"id", "question"}) this samples
--n-samples completions (default 3 -> 2K questions x 3 = 6K traces) using
Qwen3's recommended thinking settings, and writes one line per trace.

Nothing is filtered: truncated traces (hit max_tokens) and traces that never
close </think> are kept and flagged so later steps can decide what to do.

The exact prompt string and token ids are saved with every trace. Step 4
(prefill + hidden-state extraction) should feed `prompt_token_ids +
output_token_ids` straight back into the model instead of re-tokenising text,
so the tokenisation is guaranteed identical to generation.

Resumable: re-running with the same --output skips questions that already have
all their samples, and drops any half-written question from a crashed run.

Usage (on the GPU server):
    # quick smoke test
    python script/generate/generate_traces.py --limit 10 --output data/smoke.jsonl
    # full run
    python script/generate/generate_traces.py
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

DEFAULTS = dict(
    model="Qwen/Qwen3-4B",
    input="data/tessy-rnd-2k.jsonl",
    output="data/qwen3-4b-traces.jsonl",
    n_samples=3,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    max_tokens=32768,
    max_model_len=40960,  # Qwen3-4B's native context (max_position_embeddings)
    seed=1234,
    chunk_size=256,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--input", default=DEFAULTS["input"])
    p.add_argument("--output", default=DEFAULTS["output"])
    p.add_argument("--n-samples", type=int, default=DEFAULTS["n_samples"])
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--top-p", type=float, default=DEFAULTS["top_p"])
    p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    p.add_argument("--min-p", type=float, default=DEFAULTS["min_p"])
    p.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"],
                   help="Max generated tokens per trace (thinking + answer).")
    p.add_argument("--max-model-len", type=int, default=DEFAULTS["max_model_len"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                   help="Base seed; question i uses seed + i for reproducibility.")
    p.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"],
                   help="Questions per vLLM call; output is flushed after each chunk.")
    p.add_argument("--limit", type=int, default=None, help="Only use the first N questions.")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def load_questions(path, limit):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        sys.exit(f"Duplicate ids in {path}")
    return rows[:limit] if limit else rows


def recover_existing(path, n_samples):
    """Return question ids already complete; rewrite file without partial questions."""
    if not os.path.exists(path):
        return set()
    by_q = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written last line from a crash
            by_q[rec["question_id"]].append(line if line.endswith("\n") else line + "\n")
    done = {q for q, lines in by_q.items() if len(lines) == n_samples}
    dropped = len(by_q) - len(done)
    if dropped:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for q in done:
                f.writelines(by_q[q])
        os.replace(tmp, path)
        print(f"Resume: dropped {dropped} partially written question(s)")
    print(f"Resume: {len(done)} question(s) already complete in {path}")
    return done


def split_thinking(token_ids, think_start_id, think_end_id, tokenizer):
    """Split generated ids into thinking / answer using the </think> token."""
    try:
        end = token_ids.index(think_end_id)
    except ValueError:
        end = None
    think_ids = token_ids if end is None else token_ids[:end]
    start = 1 if think_ids[:1] == [think_start_id] else 0
    thinking = tokenizer.decode(think_ids[start:], skip_special_tokens=False).strip("\n")
    answer = None if end is None else tokenizer.decode(
        token_ids[end + 1:], skip_special_tokens=True).strip()
    return thinking, answer, end


def main():
    args = parse_args()

    # Imported here so --help works without a GPU / vLLM install.
    import transformers
    import vllm
    from vllm import LLM, SamplingParams

    questions = load_questions(args.input, args.limit)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    done = recover_existing(args.output, args.n_samples)
    todo = [(i, q) for i, q in enumerate(questions) if q["id"] not in done]
    print(f"{len(questions)} questions, {len(todo)} to generate, "
          f"{args.n_samples} samples each")
    if not todo:
        return

    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        enable_prefix_caching=True,  # the n samples share one prompt
    )
    tok = llm.get_tokenizer()
    think_start_id = tok.convert_tokens_to_ids("<think>")
    think_end_id = tok.convert_tokens_to_ids("</think>")

    sampling_meta = dict(
        n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, min_p=args.min_p, max_tokens=args.max_tokens,
    )
    meta = dict(
        model=args.model, max_model_len=args.max_model_len, dtype=args.dtype,
        sampling=sampling_meta, base_seed=args.seed, enable_thinking=True,
        vllm_version=vllm.__version__, transformers_version=transformers.__version__,
        input=args.input, started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(args.output + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    t0 = time.time()
    n_written = 0
    with open(args.output, "a", encoding="utf-8") as out:
        for c in range(0, len(todo), args.chunk_size):
            chunk = todo[c:c + args.chunk_size]
            prompts, params, prompt_texts = [], [], []
            for idx, q in chunk:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": q["question"]}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=True,
                )
                ids = tok(text, add_special_tokens=False)["input_ids"]
                budget = min(args.max_tokens, args.max_model_len - len(ids))
                prompts.append({"prompt_token_ids": ids})
                prompt_texts.append(text)
                params.append(SamplingParams(
                    n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k, min_p=args.min_p, max_tokens=budget,
                    seed=args.seed + idx, skip_special_tokens=False,
                ))

            results = llm.generate(prompts, params, use_tqdm=True)

            lines = []
            for (idx, q), text, sp, res in zip(chunk, prompt_texts, params, results):
                for k, o in enumerate(res.outputs):
                    out_ids = list(o.token_ids)
                    thinking, answer, think_end = split_thinking(
                        out_ids, think_start_id, think_end_id, tok)
                    lines.append(json.dumps({
                        "trace_id": f"{q['id']}_{k}",
                        "question_id": q["id"],
                        "sample_idx": k,
                        "question": q["question"],
                        "prompt_text": text,
                        "prompt_token_ids": list(res.prompt_token_ids),
                        "output_token_ids": out_ids,
                        "output_text": o.text,
                        "thinking": thinking,
                        "answer": answer,
                        "think_end_token_idx": think_end,
                        "finish_reason": o.finish_reason,
                        "truncated": o.finish_reason == "length",
                        "thinking_closed": think_end is not None,
                        "num_prompt_tokens": len(res.prompt_token_ids),
                        "num_output_tokens": len(out_ids),
                        "max_tokens": sp.max_tokens,
                        "seed": args.seed + idx,
                        "model": args.model,
                        "correct": None,  # no tests in TESSY; filled later if mapped
                    }, ensure_ascii=False))
            out.write("\n".join(lines) + "\n")
            out.flush()
            n_written += len(chunk)
            rate = n_written / (time.time() - t0)
            print(f"[{n_written}/{len(todo)} questions] "
                  f"{rate * 60:.1f} q/min, ETA {(len(todo) - n_written) / rate / 60:.1f} min")

    summarize(args.output)


def summarize(path):
    lens, trunc, unclosed, total = [], 0, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total += 1
            lens.append(r["num_output_tokens"])
            trunc += r["truncated"]
            unclosed += not r["thinking_closed"]
    lens.sort()
    stats = dict(
        traces=total,
        truncated=trunc, truncated_pct=round(100 * trunc / total, 2),
        thinking_not_closed=unclosed,
        output_tokens_mean=round(statistics.mean(lens)),
        output_tokens_p50=lens[len(lens) // 2],
        output_tokens_p90=lens[int(len(lens) * 0.9)],
        output_tokens_max=lens[-1],
        total_output_tokens=sum(lens),
    )
    with open(path + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
