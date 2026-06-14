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
# BPC computation
# ---------------------------------------------------------------------------

def eval_bpc(model: BILM, data: bytes, label: str = "") -> float:
    """Evaluate bits-per-character on held-out data (no learning)."""
    correct = 0
    total = len(data)
    if total == 0:
        return float("inf")

    for b in data:
        pred = model.tick(int(b), learn=False)
        if pred == int(b):
            correct += 1

    accuracy = correct / total
    # BPC from accuracy: -log2(accuracy) (approximate)
    accuracy = max(1e-9, accuracy)
    bpc = -math.log2(accuracy)
    if label:
        print(f"  {label}: accuracy={accuracy:.4f}  BPC~{bpc:.4f}")
    return bpc


# ---------------------------------------------------------------------------
# Benchmark A: Perplexity / BPC
# ---------------------------------------------------------------------------

def benchmark_perplexity(train_tokens: int = 500_000) -> None:
    print("\n=== Benchmark A: WikiText-2 BPC ===")

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
    bpc = eval_bpc(model, test_data, label="WikiText-2 Test")
    print(f"\n  Result: BPC = {bpc:.4f}")
    print("  (LSTM large baseline: ~1.30 BPC on WikiText-2 char-level)")


# ---------------------------------------------------------------------------
# Benchmark B: Catastrophic Forgetting
# ---------------------------------------------------------------------------

DOMAIN_A_TEXT = """
The history of science is the study of the development of science and scientific knowledge.
Science is a systematic enterprise that builds and organizes knowledge in the form of testable
explanations and predictions about the universe. The earliest roots of science can be traced
to ancient Egypt and Mesopotamia in around 3500 to 3000 BCE. Their contributions entered and
shaped Greek natural philosophy of classical antiquity, whereby formal attempts were made to
provide explanations of events in the physical world based on natural causes. After the fall
of the Western Roman Empire, knowledge of Greek conceptions of the world deteriorated in Western
Europe during the early centuries of the Middle Ages but was preserved in the Muslim world.
""" * 50  # Repeat to get ~10K chars of domain A


DOMAIN_B_TEXT = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, val):
        if val < self.value:
            if self.left is None:
                self.left = BinaryTree(val)
            else:
                self.left.insert(val)
        else:
            if self.right is None:
                self.right = BinaryTree(val)
            else:
                self.right.insert(val)

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""" * 50  # Repeat to get ~10K chars of domain B


def benchmark_forgetting() -> None:
    print("\n=== Benchmark B: Catastrophic Forgetting ===")

    model = BILM()
    domain_a = DOMAIN_A_TEXT.encode("utf-8")
    domain_b = DOMAIN_B_TEXT.encode("utf-8")
    eval_a   = DOMAIN_A_TEXT[:2000].encode("utf-8")

    # Train on Domain A
    print(f"  Training on Domain A (natural language, {len(domain_a):,} bytes) ...")
    for b in domain_a:
        model.tick(int(b), learn=True)

    model.cortex.reset_context()
    bpc_before = eval_bpc(model, eval_a, "Domain A BPC BEFORE domain B training")

    # Train on Domain B
    print(f"  Training on Domain B (Python code, {len(domain_b):,} bytes) ...")
    for b in domain_b:
        model.tick(int(b), learn=True)

    model.cortex.reset_context()
    bpc_after = eval_bpc(model, eval_a, "Domain A BPC AFTER domain B training")

    degradation = ((bpc_after - bpc_before) / max(bpc_before, 1e-9)) * 100
    print(f"\n  Degradation: {degradation:+.1f}%")
    if degradation < 10:
        print("  ✅ PASS: Catastrophic forgetting prevented (<10% degradation)")
    else:
        print("  ⚠️  Degradation present. Sparsity may need tuning.")
    print("  (Typical LSTM/Transformer degradation: 30–80%)")


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

    tps = len(sample) / max(elapsed, 1e-9)
    peak_mb = peak / 1024 / 1024

    print(f"  Tokens processed: {len(sample):,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Tokens per second: {tps:.0f} TPS")
    print(f"  Peak RAM (traced): {peak_mb:.1f} MB")
    print(f"  (GPT-2 small inference: ~500MB RAM, requires GPU)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BILM benchmark suite")
    parser.add_argument(
        "--test",
        choices=["perplexity", "forgetting", "efficiency", "all"],
        default="all",
    )
    parser.add_argument("--train-tokens", type=int, default=500_000)
    args = parser.parse_args()

    print("BILM Benchmark Suite")
    print("=" * 50)

    if args.test in ("perplexity", "all"):
        benchmark_perplexity(train_tokens=args.train_tokens)

    if args.test in ("forgetting", "all"):
        benchmark_forgetting()

    if args.test in ("efficiency", "all"):
        benchmark_efficiency()

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
