"""
BILM — train.py
Streaming training loop with checkpointing, BPC reporting, and sleep triggers.

Run:
    py -m bilm.train --text data/wiki.txt
    py -m bilm.train --text data/wiki.txt --resume checkpoints/bilm.npz
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

from bilm.model import BILM
from bilm.config import (
    CHECKPOINT_EVERY_N_TOKENS,
    REPORT_EVERY_N_TOKENS,
    SLEEP_EVERY_N_TOKENS,
)


def compute_bpc(surprise_sum: float, n: int) -> float:
    """Approximate bits-per-character from rolling surprise."""
    if n == 0:
        return float("inf")
    avg_surprise = surprise_sum / n
    avg_surprise = max(1e-9, min(1.0 - 1e-9, avg_surprise))
    return -math.log2(1.0 - avg_surprise)


def train(
    text_path: str,
    resume_path: str | None = None,
    max_tokens: int | None = None,
) -> None:
    model = BILM()

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path} ...")
        model.load(resume_path)
        print(f"  Loaded. Tokens seen so far: {model.tokens_seen:,}")

    print(f"Training on: {text_path}")
    print(f"  Reporting every {REPORT_EVERY_N_TOKENS:,} tokens")
    print(f"  Checkpointing every {CHECKPOINT_EVERY_N_TOKENS:,} tokens")
    print(f"  Sleep consolidation every {SLEEP_EVERY_N_TOKENS:,} tokens")
    print()

    with open(text_path, "rb") as f:
        raw = f.read()

    if max_tokens is not None:
        raw = raw[:max_tokens]

    total = len(raw)
    print(f"  Corpus size: {total:,} bytes")

    surprise_sum = 0.0
    window_count = 0
    t0 = time.time()

    for i, b in enumerate(raw):
        model.tick(int(b), learn=True)
        ach = model.neuromod.ach
        surprise = 1.0 - (ach - 0.10) / 0.90
        surprise_sum += max(0.0, min(1.0, surprise))
        window_count += 1

        local_tokens = model.tokens_seen

        # Periodic BPC report
        if local_tokens > 0 and local_tokens % REPORT_EVERY_N_TOKENS == 0:
            bpc = compute_bpc(surprise_sum, window_count)
            elapsed = time.time() - t0
            tps = local_tokens / max(elapsed, 1e-9)
            pct = (i + 1) / total * 100
            print(
                f"  [{pct:5.1f}%] tokens={local_tokens:>9,}  "
                f"BPC={bpc:.4f}  "
                f"ACh={ach:.3f}  "
                f"TPS={tps:.0f}"
            )
            surprise_sum = 0.0
            window_count = 0

        # Sleep consolidation
        if local_tokens > 0 and local_tokens % SLEEP_EVERY_N_TOKENS == 0:
            print(f"  [SLEEP] Consolidating at token {local_tokens:,} ...")
            model.sleep()

        # Checkpoint
        if local_tokens > 0 and local_tokens % CHECKPOINT_EVERY_N_TOKENS == 0:
            path = model.save()
            print(f"  [CHECKPOINT] Saved to {path}")

    # Final checkpoint
    path = model.save()
    total_time = time.time() - t0
    print()
    print(f"Training complete. {model.tokens_seen:,} tokens in {total_time:.1f}s")
    print(f"Final checkpoint: {path}")
    stats = model.get_stats()
    print(f"Synapses per layer: {stats['synapses_per_layer']}")
    print(f"Hippocampus binds: {stats['hippocampus']['binds']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BILM streaming training loop")
    parser.add_argument("--text", required=True, help="Path to training text file")
    parser.add_argument("--resume", default=None, help="Path to .npz checkpoint to resume from")
    parser.add_argument("--max-tokens", type=int, default=None, help="Limit tokens for testing")
    args = parser.parse_args()

    if not os.path.exists(args.text):
        print(f"Error: file not found: {args.text}", file=sys.stderr)
        sys.exit(1)

    train(args.text, resume_path=args.resume, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
