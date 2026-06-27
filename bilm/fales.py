"""
Feedback Alignment Local Error Signal (FALES).
Fixed random matrices project output error back to each cortex layer.
NOT backprop — matrices are random and never updated. Local learning rule.
Based on: Lillicrap et al. (2016) "Random synaptic feedback weights support
error backpropagation for deep learning."
"""
from __future__ import annotations
import numpy as np
from bilm.bilm_config import BILMConfig

class FeedbackAlignment:
    def __init__(self, cfg: BILMConfig) -> None:
        self.cfg = cfg
        total = cfg.sdr_size * cfg.cells_per_column
        rng   = np.random.default_rng(99)
        # Fixed, never updated
        self.B = [rng.normal(0, total**-0.5, (256, total)).astype(np.float32)
                  for _ in range(cfg.n_layers)]

    def apply(self, layer, idx: int, error_256: np.ndarray, lr: float) -> None:
        if not layer.winner_cells.any():
            return
        active = np.where(layer.winner_cells)[0]
        delta_active = error_256 @ self.B[idx][:, active]  # project only to active cells
        layer.permanences[active] += lr * delta_active[:, None]
        layer.permanences[active] = np.clip(layer.permanences[active], 0.0, 1.0)
