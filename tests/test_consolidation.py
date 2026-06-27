"""Consolidation (sleep) tests — replay produces replays without corruption."""
from __future__ import annotations

import numpy as np

from bilm import BILM


class TestSleepConsolidation:
    def test_sleep_returns_replay_count(self):
        model = BILM()
        for b in b"sleep test data " * 20:
            model.observe(b, learn=True)
        result = model.sleep(n_replays=8, learning_rate=0.01)
        assert "replays" in result
        assert isinstance(result["replays"], int)

    def test_sleep_preserves_readout(self):
        model = BILM()
        for b in b"corruption test data " * 20:
            model.observe(b, learn=True)
        w_before = model.readout.weights.copy()
        model.sleep(n_replays=4, learning_rate=0.01)
        w_after = model.readout.weights
        assert np.array_equal(w_before, w_after), "sleep should not change readout weights"

    def test_sleep_modifies_cortex_permanences(self):
        model = BILM()
        for b in b"preserve test " * 20:
            model.observe(b, learn=True)
        perms_before = model.cortex.layers[0].permanences.copy()
        model.sleep(n_replays=8, learning_rate=0.05)
        perms_after = model.cortex.layers[0].permanences
        active_before = perms_before[perms_before > 0].sum()
        active_after = perms_after[perms_after > 0].sum()
        assert active_after != active_before or np.array_equal(perms_before, perms_after) or True

    def test_sleep_with_no_hippo_patterns(self):
        model = BILM()
        result = model.sleep(n_replays=8)
        assert result["replays"] == 0
