"""Determinism and seed reproducibility tests."""
from __future__ import annotations

import numpy as np
import pytest

from bilm import BILM
from bilm.codec import ByteCodec


class TestCodecDeterminism:
    def test_same_seed_same_sdrs(self):
        sdrs1 = ByteCodec().byte_sdrs
        sdrs2 = ByteCodec().byte_sdrs
        assert np.array_equal(sdrs1, sdrs2)

    def test_all_bytes_have_unique_sdrs(self):
        codec = ByteCodec()
        sdrs = codec.byte_sdrs
        for i in range(256):
            for j in range(i + 1, 256):
                overlap = len(set(sdrs[i]) & set(sdrs[j]))
                assert overlap < 10, f"bytes {i},{j} have {overlap} overlap bits"


class TestModelDeterminism:
    def test_two_models_same_data_same_metrics(self):
        results = []
        for _ in range(2):
            model = BILM()
            losses = []
            for b in b"deterministic test sequence " * 20:
                r = model.observe(b, learn=True)
                losses.append(r.loss_bits)
            results.append(losses)
        assert results[0] == results[1]

    def test_same_data_same_readout_weights(self):
        w1, w2 = None, None
        for i in range(2):
            model = BILM()
            for b in b"weight test " * 15:
                model.observe(b, learn=True)
            w = model.readout.weights.copy()
            if i == 0:
                w1 = w
            else:
                w2 = w
        assert np.array_equal(w1, w2)
