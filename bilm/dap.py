"""Dense Associative Projection — enriches SDR encoding before cortex sees it."""
from __future__ import annotations
import numpy as np
from bilm.bilm_config import BILMConfig

class DenseAssociativeProjection:
    def __init__(self, cfg: BILMConfig) -> None:
        self.cfg = cfg
        self.W = np.random.default_rng(42).normal(
            0, cfg.sdr_size**-0.5, (cfg.sdr_size, cfg.dap_dim)
        ).astype(np.float32)

    def project(self, cols: np.ndarray) -> np.ndarray:
        return self.W[cols].mean(0) if cols.size else np.zeros(self.cfg.dap_dim, np.float32)

    def learn(self, cols: np.ndarray, error: np.ndarray) -> None:
        if cols.size:
            self.W[cols] += self.cfg.dap_lr * error[None, :]
