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
    report = model.evaluate(data, warmup=min(256, len(data)))
    accuracy = report.accuracy
    if label:
        print(
            f"  {label}: accuracy={accuracy:.4f} "
            f"BPB={report.bits_per_byte:.4f} PPL={report.perplexity:.2f}"
        )
    return accuracy


# ---------------------------------------------------------------------------
# Benchmark A: Accuracy
# ---------------------------------------------------------------------------

def benchmark_accuracy(train_tokens: int = 100_000) -> None:
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
    report = model.evaluate(test_data, warmup=256)
    acc = report.accuracy
    print(
        f"  WikiText-2 Test: accuracy={acc:.4f} "
        f"BPB={report.bits_per_byte:.4f} PPL={report.perplexity:.2f}"
    )
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

    model.cortex.reset_context()
    domain_b_before = model.evaluate(domain_b, warmup=min(256, len(domain_b)))

    # Train on Domain B
    print(f"  Training on Domain B (Python code, {len(domain_b):,} bytes) ...")
    for b in domain_b:
        model.tick(int(b), learn=True)

    model.cortex.reset_context()
    domain_b_after = model.evaluate(domain_b, warmup=min(256, len(domain_b)))

    model.cortex.reset_context()
    acc_after = eval_accuracy(model, eval_a, "Domain A accuracy AFTER domain B training")

    degradation = ((acc_after - acc_before) / max(acc_before, 1e-9)) * 100
    acquisition = domain_b_before.bits_per_byte - domain_b_after.bits_per_byte
    print(f"\n  Accuracy Change: {degradation:+.1f}%")
    print(f"  Domain B acquisition: {acquisition:+.4f} BPB improvement")
    if degradation > -10 and acquisition > 0.0:
        print("  [PASS] Acquisition observed with <10% relative accuracy degradation")
    elif acquisition <= 0.0:
        print("  [WARN] No Domain B acquisition; retention alone is not evidence")
    else:
        print("  [WARN] Degradation present. Interference occurred.")


# ---------------------------------------------------------------------------
# Benchmark C: Efficiency
# ---------------------------------------------------------------------------

def benchmark_efficiency() -> None:
    print("\n=== Benchmark C: Efficiency ===")

    gc.collect()
    tracemalloc.start()
    proc_start_rss = _get_rss_mb()

    model = BILM()
    sample = b"The quick brown fox jumps over the lazy dog. " * 1000

    t0 = time.time()
    for b in sample:
        model.tick(int(b), learn=True)
    elapsed = time.time() - t0

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    proc_peak_rss = _get_rss_mb()

    peak_mb = peak / 1024 / 1024
    rss_mb = proc_peak_rss - proc_start_rss

    tps = len(sample) / max(elapsed, 1e-9)

    print(f"  Tokens processed: {len(sample):,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens per second: {tps:.0f} TPS")
    print(f"  Peak RSS (delta): {rss_mb:.1f} MB")
    print(f"  Peak tracemalloc: {peak_mb:.1f} MB")


def _get_rss_mb() -> float:
    """Get current process RSS in MB (Windows-compatible via tracemalloc fallback)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    try:
        import ctypes
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
    except Exception:
        pass
    gc.collect()
    _, peak = tracemalloc.get_traced_memory()
    return peak / 1024 / 1024


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
    parser.add_argument("--train-tokens", type=int, default=100_000)
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
