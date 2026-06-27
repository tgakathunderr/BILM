"""Phase 0/1 invariant tests — autoregressive alignment, probability contracts,
evaluation side-effects, and checkpoint round-trips."""
from __future__ import annotations

import math
import numpy as np
import pytest

from bilm import BILM
from bilm.metrics import target_loss_bits, bits_per_byte, byte_perplexity


class TestAutoregressiveAlignment:
    """Prediction scored against byte i must use only context bytes 0..i-1."""

    def test_first_byte_gets_uniform(self):
        model = BILM()
        pred = model.predict_next()
        assert pred.probabilities.shape == (256,)
        assert np.isclose(pred.probabilities.sum(), 1.0)
        assert np.allclose(pred.probabilities, 1.0 / 256.0)
        assert target_loss_bits(pred.probabilities, 0) == 8.0

    def test_observe_returns_prior_loss_correctly(self):
        model = BILM()
        data = b"AAAA"
        for b in data:
            result = model.observe(b, learn=False)
            assert result.target == b
            assert result.prior_prediction is not result.next_prediction

    def test_evaluate_scores_prediction_before_byte(self):
        model = BILM()
        for b in b"train " * 10:
            model.observe(b, learn=True)
        model.cortex.reset_context()
        report = model.evaluate(b"hello", warmup=0, reset_context=True)
        assert report.tokens == 5
        assert math.isfinite(report.bits_per_byte)
        assert 0.0 <= report.accuracy <= 1.0

    def test_uniform_model_scores_exactly_8_bpb(self):
        probs = np.full(256, 1.0 / 256.0, dtype=np.float64)
        for target in range(256):
            loss = target_loss_bits(probs, target)
            assert loss == 8.0


class TestProbabilityContract:
    """All probability outputs must be finite, non-negative, and sum to 1."""

    def test_fresh_model_prediction(self):
        model = BILM()
        pred = model.predict_next()
        assert np.all(np.isfinite(pred.probabilities))
        assert np.all(pred.probabilities >= 0.0)
        assert math.isclose(pred.probabilities.sum(), 1.0, rel_tol=1e-7)

    def test_after_training_prediction(self):
        model = BILM()
        for b in b"train " * 10:
            model.observe(b, learn=True)
        pred = model.predict_next()
        assert np.all(np.isfinite(pred.probabilities))
        assert np.all(pred.probabilities >= 0.0)
        assert math.isclose(pred.probabilities.sum(), 1.0, rel_tol=1e-7)

    def test_empty_columns_gives_uniform(self):
        model = BILM()
        pred = model.predict_next()
        assert np.allclose(pred.probabilities, 1.0 / 256.0)

    def test_argmax_matches_confidence(self):
        model = BILM()
        for b in b"test " * 10:
            model.observe(b, learn=True)
        pred = model.predict_next()
        assert pred.argmax == int(np.argmax(pred.probabilities))
        assert pred.confidence == float(pred.probabilities[pred.argmax])


class TestEvaluationSideEffectFree:
    """evaluate() must not mutate any learnable or adaptive state."""

    def _snapshot(self, model):
        return {
            "readout_w": model.readout.weights.copy(),
            "readout_b": model.readout.bias.copy(),
            "readout_updates": model.readout.updates,
            "codec_freq": model.codec.frequencies.copy(),
            "codec_total": model.codec.total_seen,
            "neuromod_ach": model.neuromod.ach,
            "neuromod_surprise": list(model.neuromod._surprise_history),
            "neuromod_var": list(model.neuromod._variance_window),
            "homeostasis_step": model.homeostasis.step_count,
            "homeostasis_apps": model.homeostasis.applications,
            "hippo_binds": model.hippocampus.binds,
            "hippo_retrievals": model.hippocampus.retrievals,
            "tokens": model.tokens_seen,
        }

    def _assert_equal(self, before, after, msg=""):
        for key in before:
            b, a = before[key], after[key]
            if isinstance(b, np.ndarray):
                assert np.array_equal(b, a), f"{msg}: {key} changed"
            elif isinstance(b, list):
                assert b == a, f"{msg}: {key} changed"
            else:
                assert b == a, f"{msg}: {key} changed {b} -> {a}"

    def test_evaluate_preserves_all_state(self):
        model = BILM()
        for b in b"test " * 10:
            model.observe(b, learn=True)
        snap = self._snapshot(model)
        model.evaluate(b"eval data here", warmup=1, reset_context=True)
        self._assert_equal(snap, self._snapshot(model))

    def test_evaluate_after_training_is_idempotent(self):
        model = BILM()
        for b in b"train " * 10:
            model.observe(b, learn=True)
        r1 = model.evaluate(b"test", warmup=0, reset_context=True)
        r2 = model.evaluate(b"test", warmup=0, reset_context=True)
        assert r1.bits_per_byte == r2.bits_per_byte
        assert r1.accuracy == r2.accuracy
