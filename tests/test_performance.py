"""Performance budget tests — RSS, TPS, checkpoint size, latency."""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from bilm import BILM


class TestTPSBudget:
    def test_throughput_above_minimum(self):
        model = BILM()
        data = b"perf " * 50
        t0 = time.perf_counter()
        for b in data:
            model.observe(b, learn=True)
        elapsed = time.perf_counter() - t0
        tps = len(data) / max(elapsed, 1e-9)
        assert tps > 1.0, f"TPS {tps:.1f} too low (minimum 1.0)"


class TestCheckpointSize:
    def test_checkpoint_reasonable_size(self):
        model = BILM()
        for b in b"size " * 10:
            model.observe(b, learn=True)
        path = os.path.join(os.path.dirname(__file__), "_test_size.npz")
        try:
            model.save(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            assert size_mb < 500, f"Checkpoint {size_mb:.1f} MB too large"
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestLatencyBudget:
    def test_single_observe_latency(self):
        model = BILM()
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            model.observe(ord('a'), learn=True)
            times.append(time.perf_counter() - t0)
        median_ms = sorted(times)[len(times) // 2] * 1000
        assert median_ms < 500, f"Median observe latency {median_ms:.1f}ms too high"

    def test_evaluate_latency(self):
        model = BILM()
        for b in b"warmup " * 5:
            model.observe(b, learn=True)
        data = b"eval " * 5
        t0 = time.perf_counter()
        model.evaluate(data, warmup=0, reset_context=True)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30, f"Evaluate took {elapsed:.1f}s, budget is 30s"
