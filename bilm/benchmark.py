from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
import tracemalloc
import urllib.request

from bilm.model import BILM


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

WIKITEXT2_TRAIN_URL = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
WIKITEXT2_TEST_URL  = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"  Saved to {path}")


def _load_text(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def eval_accuracy(model: BILM, data: bytes, label: str = "") -> float:
    """Evaluate next-byte prediction accuracy on held-out data (no learning)."""
    correct = 0
    total = len(data)
    if total == 0:
        return 0.0

    for b in data:
        pred = model.tick(int(b), learn=False)
        if pred == int(b):
            correct += 1

    accuracy = correct / total
    if label:
        print(f"  {label}: accuracy={accuracy:.4f}")
    return accuracy


# ---------------------------------------------------------------------------
# Benchmark A: Accuracy
# ---------------------------------------------------------------------------

def benchmark_accuracy(train_tokens: int = 500_000) -> None:
    print("\n=== Benchmark A: WikiText-2 Next-Byte Accuracy ===")

    train_path = os.path.join(DATA_DIR, "wikitext2_train.txt")
    test_path  = os.path.join(DATA_DIR, "wikitext2_test.txt")

    _download(WIKITEXT2_TRAIN_URL, train_path)
    _download(WIKITEXT2_TEST_URL, test_path)

    train_data = _load_text(train_path)[:train_tokens]
    test_data  = _load_text(test_path)[:50_000]

    print(f"  Training on {len(train_data):,} tokens ...")
    model = BILM()
    t0 = time.time()

    for b in train_data:
        model.tick(int(b), learn=True)

    elapsed = time.time() - t0
    tps = len(train_data) / max(elapsed, 1e-9)
    print(f"  Training done in {elapsed:.1f}s  ({tps:.0f} TPS)")

    model.cortex.reset_context()
    acc = eval_accuracy(model, test_data, label="WikiText-2 Test")
    print(f"\n  Result: Next-Byte Accuracy = {acc:.4f}")
    print("  (Note: BILM utilizes sparse representations without softmax distributions,")
    print("   making direct BPC comparisons to dense models invalid.)")


# ---------------------------------------------------------------------------
# Benchmark B: Catastrophic Forgetting
# ---------------------------------------------------------------------------

def benchmark_forgetting() -> None:
    print("\n=== Benchmark B: Catastrophic Forgetting ===")

    # Use real, non-repeating data for honest testing
    train_path = os.path.join(DATA_DIR, "wikitext2_train.txt")
    _download(WIKITEXT2_TRAIN_URL, train_path)
    wiki_data = _load_text(train_path)

    domain_a = wiki_data[:20000]
    # Held-out evaluation set from Domain A (never seen during training)
    eval_a   = wiki_data[20000:25000]

    # Domain B: python code (use this file itself)
    with open(__file__, "rb") as f:
        domain_b = f.read()

    model = BILM()

    # Train on Domain A
    print(f"  Training on Domain A (natural language, {len(domain_a):,} bytes) ...")
    for b in domain_a:
        model.tick(int(b), learn=True)

    model.cortex.reset_context()
    acc_before = eval_accuracy(model, eval_a, "Domain A accuracy BEFORE domain B training")

    # Train on Domain B
    print(f"  Training on Domain B (Python code, {len(domain_b):,} bytes) ...")
    for b in domain_b:
        model.tick(int(b), learn=True)

    model.cortex.reset_context()
    acc_after = eval_accuracy(model, eval_a, "Domain A accuracy AFTER domain B training")

    degradation = ((acc_after - acc_before) / max(acc_before, 1e-9)) * 100
    print(f"\n  Accuracy Change: {degradation:+.1f}%")
    if degradation > -10:
        print("  [PASS] Catastrophic forgetting prevented (<10% degradation)")
    else:
        print("  [WARN] Degradation present. Interference occurred.")


# ---------------------------------------------------------------------------
# Benchmark C: Efficiency
# ---------------------------------------------------------------------------

def benchmark_efficiency() -> None:
    print("\n=== Benchmark C: Efficiency ===")

    gc.collect()
    tracemalloc.start()

    model = BILM()
    sample = b"The quick brown fox jumps over the lazy dog. " * 1000

    t0 = time.time()
    for b in sample:
        model.tick(int(b), learn=True)
    elapsed = time.time() - t0

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # NumPy C-allocations (like the 8192x8192 float32 Hippocampus W matrix) 
    # are often missed by tracemalloc. Add manual estimate:
    # 8192 * 8192 * 4 bytes = 268.4 MB
    estimated_np_mb = 268.4
    peak_mb = (peak / 1024 / 1024) + estimated_np_mb

    tps = len(sample) / max(elapsed, 1e-9)

    print(f"  Tokens processed: {len(sample):,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens per second: {tps:.0f} TPS")
    print(f"  Peak RAM (traced + numpy est): ~{peak_mb:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BILM benchmark suite")
    parser.add_argument(
        "--test",
        choices=["accuracy", "forgetting", "efficiency", "all"],
        default="all",
    )
    parser.add_argument("--train-tokens", type=int, default=500_000)
    args = parser.parse_args()

    print("BILM Benchmark Suite")
    print("=" * 50)

    if args.test in ("accuracy", "all"):
        benchmark_accuracy(train_tokens=args.train_tokens)

    if args.test in ("forgetting", "all"):
        benchmark_forgetting()

    if args.test in ("efficiency", "all"):
        benchmark_efficiency()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
