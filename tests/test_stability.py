"""Stability tests — no NaN, no unbounded memory, no runaway synapses."""
from __future__ import annotations

import numpy as np

from bilm import BILM
from bilm.config import MAX_SYNAPSES_PER_CELL


class TestStability:
    def test_no_nan_after_training(self):
        model = BILM()
        for b in b"stability " * 20:
            result = model.observe(b, learn=True)
            assert np.all(np.isfinite(result.prior_prediction.probabilities)), \
                "NaN in prediction probabilities"
            assert np.isfinite(result.loss_bits), "NaN in loss"
            assert np.isfinite(result.surprise), "NaN in surprise"

    def test_readout_weights_bounded(self):
        model = BILM()
        for b in b"bounded " * 20:
            model.observe(b, learn=True)
        w = model.readout.weights
        assert np.all(np.isfinite(w)), "NaN in readout weights"
        assert np.all(np.abs(w) < 100.0), "Readout weights too large"

    def test_synapse_counts_bounded(self):
        model = BILM()
        for b in b"synapse " * 20:
            model.observe(b, learn=True)
        for layer in model.cortex.layers:
            assert np.all(layer.synapse_counts <= MAX_SYNAPSES_PER_CELL), \
                "Synapse count exceeded max"
            assert np.all(layer.synapse_counts >= 0), "Negative synapse count"
            assert np.all(np.isfinite(layer.permanences)), "NaN in permanences"

    def test_no_irreversible_saturation(self):
        model = BILM()
        for b in b"saturation " * 20:
            model.observe(b, learn=True)
        for layer in model.cortex.layers:
            perms = layer.permanences[layer.synapse_counts > 0]
            if perms.size > 0:
                assert perms.max() <= 1.0, "Permanence exceeds 1.0"
                assert perms.min() >= 0.0, "Permanence below 0.0"
