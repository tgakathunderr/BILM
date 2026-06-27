from __future__ import annotations

import numpy as np

from bilm.codec import ByteCodec
from bilm.bilm_config import BILMConfig


class LocalByteReadout:
    """Online output-local decoder; no error is propagated into the cortex."""

    def __init__(self, cfg: BILMConfig | None = None) -> None:
        self.cfg = cfg or BILMConfig()
        self.weights = np.zeros((self.cfg.sdr_size, 256), dtype=np.float32)
        self.bias = np.zeros(256, dtype=np.float32)
        self.updates = 0

    def predict(
        self,
        predictive_columns: np.ndarray,
        codec: ByteCodec,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")
        columns = np.asarray(predictive_columns, dtype=np.int64)
        logits = self.bias.astype(np.float64)
        if columns.size:
            logits += self.weights[columns].mean(axis=0)
            logits += 1.0 * codec.overlap_scores(columns)
        logits /= float(temperature)
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return probabilities

    def learn(
        self,
        predictive_columns: np.ndarray,
        probabilities: np.ndarray,
        target: int,
    ) -> None:
        """Apply a local delta update at active cortical-output synapses."""
        columns = np.unique(np.asarray(predictive_columns, dtype=np.int64))
        error = -np.asarray(probabilities, dtype=np.float32)
        error[int(target) & 0xFF] += 1.0
        self.bias += np.float32(0.01) * error
        if columns.size:
            update = np.float32(0.05) * error
            self.weights[columns] = self.weights[columns] + update[None, :]
        self.updates += 1
