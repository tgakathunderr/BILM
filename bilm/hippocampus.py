"""
BILM — hippocampus.py
CA3 sparse attractor network for episodic memory and long-context retrieval.

Ported from BIM 3 core/hippocampus.py.
Binds high-surprise cortical states into Hopfield-style attractor weights.
Retrieves via iterative dynamics (pattern completion from partial cues).
BILM-specific: applies retrieved attractor as apical bias on L1 for long-range context.
"""
from __future__ import annotations

import numpy as np

from bilm.config import (
    HIPPO_BIND_SURPRISE_THRESHOLD,
    HIPPO_LEARNING_RATE,
    HIPPO_RETRIEVE_ITER,
    HIPPO_SIZE,
    HIPPO_SPARSITY,
    HIPPO_WEIGHT_MAX,
    SDR_SIZE,
    SDR_SPARSITY,
)


def _project_cortex_to_hippo(cortex_cols: np.ndarray) -> np.ndarray:
    """Deterministic fixed hash projection from SDR column space to HIPPO_SIZE cells."""
    if len(cortex_cols) == 0:
        return np.array([], dtype=np.int32)
    hashed = (cortex_cols.astype(np.int64) * np.int64(2654435769)) & 0xFFFFFFFF
    hashed = (hashed % HIPPO_SIZE).astype(np.int32)
    uniq = np.unique(hashed)
    if uniq.size >= HIPPO_SPARSITY:
        return uniq[:HIPPO_SPARSITY]
    pad = np.setdiff1d(np.arange(HIPPO_SIZE, dtype=np.int32), uniq)
    need = HIPPO_SPARSITY - uniq.size
    return np.concatenate([uniq, pad[:need]])


class Hippocampus:
    """
    Sparse CA3 attractor memory.
    Binds high-surprise cortical states. Retrieves via iterative dynamics.
    Proven 100% retrieval on 50%-corrupted inputs (BIM 3 tests).
    """

    def __init__(self) -> None:
        self.W: np.ndarray = np.zeros((HIPPO_SIZE, HIPPO_SIZE), dtype=np.float32)
        self.last_active: np.ndarray = np.array([], dtype=np.int32)
        self.binds: int = 0
        self.retrievals: int = 0

    def maybe_bind(self, cortex_cols: np.ndarray, surprise: float) -> bool:
        """Bind the cortical state if surprise is above threshold."""
        if float(surprise) < HIPPO_BIND_SURPRISE_THRESHOLD:
            return False
        if len(cortex_cols) == 0:
            return False
        hippo_cells = _project_cortex_to_hippo(cortex_cols)
        if hippo_cells.size == 0:
            return False
        self._hebbian_bind(hippo_cells)
        self.binds += 1
        self.last_active = hippo_cells
        return True

    def _hebbian_bind(self, active_cells: np.ndarray) -> None:
        idx = active_cells.astype(np.int32)
        ii, jj = np.meshgrid(idx, idx, indexing="ij")
        w_update = self.W[ii, jj] + np.float32(HIPPO_LEARNING_RATE)
        np.clip(w_update, -HIPPO_WEIGHT_MAX, HIPPO_WEIGHT_MAX, out=self.W[ii, jj])

    def retrieve(self, cue_cortex_cols: np.ndarray) -> np.ndarray:
        """Iterate attractor dynamics from a cortical cue. Returns settled hippocampal cells."""
        cue = _project_cortex_to_hippo(cue_cortex_cols)
        if cue.size == 0:
            return np.array([], dtype=np.int32)
        state = np.zeros(HIPPO_SIZE, dtype=np.float32)
        state[cue] = 1.0
        for _ in range(int(HIPPO_RETRIEVE_ITER)):
            energy = self.W @ state
            if not energy.any():
                break
            k = min(HIPPO_SPARSITY, HIPPO_SIZE)
            top = np.argpartition(energy, -k)[-k:]
            new_state = np.zeros(HIPPO_SIZE, dtype=np.float32)
            new_state[top] = 1.0
            if np.array_equal(np.where(new_state > 0)[0], np.where(state > 0)[0]):
                state = new_state
                break
            state = new_state
        self.retrievals += 1
        active = np.where(state > 0)[0].astype(np.int32)
        self.last_active = active
        return active

    def apply_to_cortex(self, retrieved: np.ndarray, hierarchy, gain: float = 0.2) -> None:
        """Inject retrieved attractor as apical bias on L1 (long-context signal)."""
        if retrieved.size == 0:
            return
        retrieved_set = set(int(x) for x in retrieved)
        cortex_cols = np.arange(SDR_SIZE, dtype=np.int64)
        hashed = (cortex_cols * np.int64(2654435769)) & 0xFFFFFFFF
        hashed = (hashed % HIPPO_SIZE).astype(np.int32)
        mask = np.isin(hashed, list(retrieved_set))
        candidates = np.where(mask)[0]
        if candidates.size == 0:
            return
        if candidates.size > SDR_SPARSITY:
            candidates = candidates[:SDR_SPARSITY]
        bias = np.zeros(SDR_SIZE, dtype=np.float32)
        bias[candidates] = float(gain)
        np.clip(bias, 0.0, 0.3, out=bias)
        hierarchy.layers[0].apply_apical_bias(bias)

    def get_stats(self) -> dict:
        return {
            "binds": self.binds,
            "retrievals": self.retrievals,
            "active_weights": int(np.count_nonzero(self.W)),
        }
