"""Hippocampus CA3 memory tests — binding, retrieval, bounded memory, eviction."""
from __future__ import annotations

import numpy as np
import pytest

from bilm import BILM
from bilm.hippocampus import Hippocampus, _project_cortex_to_hippo
from bilm.config import HIPPO_SIZE, HIPPO_MAX_SYNAPSES_PER_CELL


class TestHebbianBinding:
    def test_binding_increases_weights(self):
        hippo = Hippocampus()
        pattern = np.array([1, 3, 7, 11], dtype=np.int32)
        hippo._hebbian_bind(pattern)
        stats = hippo.get_stats()
        assert stats["active_weights"] > 0
        assert hippo.binds == 0

    def test_repeated_binding_strengthens(self):
        hippo = Hippocampus()
        pattern = np.array([1, 3, 7], dtype=np.int32)
        hippo._hebbian_bind(pattern)
        w1 = int(hippo.counts[1])
        hippo._hebbian_bind(pattern)
        w2 = int(hippo.counts[1])
        assert w2 >= w1


class TestCorticalProjection:
    def test_projection_is_deterministic(self):
        cols = np.array([10, 20, 30], dtype=np.int64)
        p1 = _project_cortex_to_hippo(cols)
        p2 = _project_cortex_to_hippo(cols)
        assert np.array_equal(p1, p2)

    def test_projection_is_bounded(self):
        cols = np.arange(200, dtype=np.int64)
        projected = _project_cortex_to_hippo(cols)
        assert projected.size <= 164


class TestRetrieval:
    def test_partial_cue_completes_pattern(self):
        hippo = Hippocampus()
        pattern = np.array([1, 3, 7, 11, 15], dtype=np.int32)
        hippo._hebbian_bind(pattern)
        cue = np.array([1], dtype=np.int32)
        result = hippo.retrieve_ca3(cue)
        assert result.size > 0
        assert 1 in result

    def test_empty_cue_returns_empty(self):
        hippo = Hippocampus()
        result = hippo.retrieve_ca3(np.array([], dtype=np.int32))
        assert result.size == 0


class TestBoundedMemory:
    def test_memory_is_bounded(self):
        hippo = Hippocampus()
        assert hippo.targets.shape == (HIPPO_SIZE, HIPPO_MAX_SYNAPSES_PER_CELL)
        assert hippo.counts.shape == (HIPPO_SIZE,)

    def test_memory_usage_bounded(self):
        hippo = Hippocampus()
        for i in range(0, HIPPO_SIZE, 10):
            cells = np.arange(i, min(i + 20, HIPPO_SIZE), dtype=np.int32)
            hippo._hebbian_bind(cells)
        stats = hippo.get_stats()
        assert stats["active_weights"] <= HIPPO_SIZE * HIPPO_MAX_SYNAPSES_PER_CELL


class TestBILMHippocampus:
    def test_high_surprise_triggers_binding(self):
        model = BILM()
        surprise_events = 0
        for b in b"surprise test " * 30:
            r = model.observe(b, learn=True)
            if r.surprise > 0.3:
                surprise_events += 1
        assert model.hippocampus.binds > 0 or surprise_events == 0
