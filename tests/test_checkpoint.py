"""Checkpoint version and roundtrip tests."""
from __future__ import annotations

import numpy as np
import os

from bilm import BILM


class TestCheckpointRoundTrip:
    def test_save_load_evaluates_identically(self):
        model = BILM()
        for b in b"data " * 20:
            model.observe(b, learn=True)

        path = os.path.join(os.path.dirname(__file__), "_test_ckpt.npz")
        try:
            model.save(path)
            model2 = BILM.from_checkpoint(path)

            test_data = b"roundtrip eval"
            model.cortex.reset_context()
            model2.cortex.reset_context()
            r1 = model.evaluate(test_data, warmup=0, reset_context=True)
            r2 = model2.evaluate(test_data, warmup=0, reset_context=True)

            assert r1.bits_per_byte == r2.bits_per_byte
            assert r1.accuracy == r2.accuracy
            assert model.tokens_seen == model2.tokens_seen
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_checkpoint_contains_version(self):
        model = BILM()
        for b in b"v" * 5:
            model.observe(b, learn=True)

        path = os.path.join(os.path.dirname(__file__), "_test_ver.npz")
        try:
            model.save(path)
            data = np.load(path, allow_pickle=False)
            assert "checkpoint_version" in data
            assert int(data["checkpoint_version"][0]) == 2
            data.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_all_state_roundtrips(self):
        model = BILM()
        for b in b"state " * 10:
            model.observe(b, learn=True)

        path = os.path.join(os.path.dirname(__file__), "_test_state.npz")
        try:
            model.save(path)
            model2 = BILM.from_checkpoint(path)

            assert model.tokens_seen == model2.tokens_seen
            assert model.neuromod.ach == model2.neuromod.ach
            assert model.hippocampus.binds == model2.hippocampus.binds
            assert model.hippocampus.retrievals == model2.hippocampus.retrievals
            assert np.array_equal(model.codec.frequencies, model2.codec.frequencies)
            assert model.codec.total_seen == model2.codec.total_seen
        finally:
            if os.path.exists(path):
                os.unlink(path)
