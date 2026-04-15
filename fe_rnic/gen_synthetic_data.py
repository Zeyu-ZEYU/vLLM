#!/usr/bin/env python3
"""
Synthetic data generator for PD disagg serving benchmark.

Generates synthetic prompts of specified token lengths for testing.
Supports multiple input lengths and saves to JSONL for replay.

Usage:
  # Generate 10 prompts each for 10K, 20K, 30K tokens
  python gen_synthetic_data.py --input-lens 10000,20000,30000 --num-per-len 10

  # Generate with specific output length
  python gen_synthetic_data.py --input-lens 10000 --num-per-len 5 --max-tokens 100

  # Save to file
  python gen_synthetic_data.py --input-lens 10000,20000 --num-per-len 3 -o data.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path


# Vocabulary for synthetic prompt generation
VOCAB = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "and", "cat", "sat", "on", "mat", "in", "big", "red", "house",
    "with", "from", "that", "this", "for", "are", "but", "not",
    "you", "all", "can", "had", "her", "was", "one", "our", "out",
    "day", "get", "has", "him", "his", "how", "its", "may", "new",
    "now", "old", "see", "way", "who", "boy", "did", "let", "put",
    "say", "she", "too", "use", "man", "run", "set", "try", "ask",
    "own", "any", "off", "end", "why", "each", "just", "know",
    "take", "come", "make", "like", "long", "look", "many", "some",
    "time", "very", "when", "word", "work", "year", "also", "back",
    "been", "call", "first", "give", "good", "great", "hand", "help",
    "high", "home", "keep", "last", "life", "line", "live", "long",
    "much", "name", "need", "next", "only", "open", "part", "place",
]


def generate_prompt(target_tokens: int, seed: int = None) -> str:
    """
    Generate a synthetic prompt of approximately target_tokens tokens.

    Approximation: ~1.3 tokens per word for common English words.
    We generate slightly fewer words than target_tokens to account for
    subword tokenization.
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    num_words = int(target_tokens * 0.8)  # conservative: 1 word ≈ 1.25 tokens
    words = [rng.choice(VOCAB) for _ in range(num_words)]
    return " ".join(words)


def generate_dataset(
    input_lens: list[int],
    num_per_len: int,
    max_tokens: int,
    model: str,
    seed: int = 42,
) -> list[dict]:
    """Generate a list of benchmark requests."""
    requests = []
    req_id = 0

    for input_len in input_lens:
        for i in range(num_per_len):
            prompt = generate_prompt(input_len, seed=seed + req_id)
            requests.append({
                "req_id": req_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "model": model,
                "target_input_len": input_len,
                "approx_word_count": len(prompt.split()),
            })
            req_id += 1

    return requests


def main():
    parser = argparse.ArgumentParser(description="Synthetic data generator")
    parser.add_argument(
        "--input-lens", type=str, required=True,
        help="Comma-separated target input lengths in tokens "
             "(e.g., 10000,20000,30000)",
    )
    parser.add_argument(
        "--num-per-len", type=int, default=10,
        help="Number of prompts per input length",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Max output tokens for each request",
    )
    parser.add_argument(
        "--model", type=str, default="Qwen3-235B",
        help="Model name",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file path (JSONL format). Default: print to stdout",
    )

    args = parser.parse_args()
    input_lens = [int(x) for x in args.input_lens.split(",")]

    requests = generate_dataset(
        input_lens=input_lens,
        num_per_len=args.num_per_len,
        max_tokens=args.max_tokens,
        model=args.model,
        seed=args.seed,
    )

    print(f"Generated {len(requests)} requests:", file=sys.stderr)
    for il in input_lens:
        count = sum(1 for r in requests if r["target_input_len"] == il)
        sample = next(r for r in requests if r["target_input_len"] == il)
        print(f"  {il} tokens: {count} requests "
              f"(~{sample['approx_word_count']} words each)", file=sys.stderr)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for req in requests:
                f.write(json.dumps(req) + "\n")
        print(f"Saved to: {args.output}", file=sys.stderr)
    else:
        for req in requests:
            print(json.dumps(req))


if __name__ == "__main__":
    main()
