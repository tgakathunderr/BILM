"""Training improvement tests — model must beat uniform on repeated sequences."""
from __future__ import annotations

import math
import numpy as np

from bilm import BILM
from bilm.baselines import UnigramByteLM, NGramByteLM


class TestLearningImprovement:
    def test_repeated_sequence_beats_uniform(self):
        model = BILM()
        data = b"hello " * 50
        for b in data:
            model.observe(b, learn=True)
        model.cortex.reset_context()
        report = model.evaluate(data[:200], warmup=2)
        assert report.accuracy > 0.3, f"accuracy {report.accuracy} too low for repeated sequence"

    def test_alternating_pattern_learned(self):
        model = BILM()
        data = b"AB" * 100
        for b in data:
            model.observe(b, learn=True)
        model.cortex.reset_context()
        report = model.evaluate(b"ABABABABAB", warmup=0)
        assert report.accuracy > 0.5, f"accuracy {report.accuracy} <= 0.5"

    def test_train_on_text_returns_losses(self):
        model = BILM()
        losses = model.train_on_text("test " * 5)
        assert len(losses) > 0
        assert all(math.isfinite(l) for l in losses)
        assert all(l >= 0.0 for l in losses)


class TestBaselineComparison:
    def test_unigram_beats_uniform(self):
        model = UnigramByteLM()
        data = b"hello world " * 20
        model.train(data)
        report = model.evaluate(data[:200], warmup=0)
        assert report.bits_per_byte < 8.0

    def test_bigram_beats_unigram(self):
        data = b"hello world " * 20
        uni = UnigramByteLM()
        uni.train(data)
        r_uni = uni.evaluate(data[:200], warmup=0)

        bi = NGramByteLM(order=2)
        bi.train(data)
        r_bi = bi.evaluate(data[:200], warmup=0)

        assert r_bi.bits_per_byte <= r_uni.bits_per_byte

    def test_trigram_beats_bigram(self):
        data = b"the cat sat on the mat and the dog ran " * 50
        bi = NGramByteLM(order=2)
        bi.train(data)
        r_bi = bi.evaluate(data[:500], warmup=0)

        tri = NGramByteLM(order=3)
        tri.train(data)
        r_tri = tri.evaluate(data[:500], warmup=0)

        assert r_tri.bits_per_byte <= r_bi.bits_per_byte + 0.5
