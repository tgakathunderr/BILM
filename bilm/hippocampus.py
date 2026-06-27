from __future__ import annotations

import numpy as np

from bilm.bilm_config import BILMConfig


def _project_cortex_to_hippo(cortex_cols: np.ndarray, cfg: BILMConfig | None = None) -> np.ndarray:
    """Deterministic sparse projection from cortical columns into CA3 cells."""
    c = cfg or BILMConfig()
    if len(cortex_cols) == 0:
        return np.array([], dtype=np.int32)
    hashed = (np.asarray(cortex_cols, dtype=np.int64) * np.int64(2654435769)) & 0xFFFFFFFF
    unique = np.unique((hashed % c.hippo_size).astype(np.int32))
    return unique[:c.hippo_sparsity]


class Hippocampus:
    """Bounded sparse CA3 recurrent attractor with local Hebbian binding."""

    def __init__(self, cfg: BILMConfig | None = None) -> None:
        self.cfg = cfg or BILMConfig()
        self.targets = np.full(
            (self.cfg.hippo_size, self.cfg.hippo_max_synapses), -1, dtype=np.int32
        )
        self.weights = np.zeros(
            (self.cfg.hippo_size, self.cfg.hippo_max_synapses), dtype=np.float32
        )
        self.counts = np.zeros(self.cfg.hippo_size, dtype=np.int16)
        self.last_active = np.array([], dtype=np.int32)
        self.binds = 0
        self.retrievals = 0

    def _project_cortex_to_hippo(self, cortex_cols: np.ndarray) -> np.ndarray:
        return _project_cortex_to_hippo(cortex_cols, self.cfg)

    def maybe_bind(self, cortex_cols: np.ndarray, surprise: float) -> bool:
        if float(surprise) < self.cfg.hippo_bind_threshold or len(cortex_cols) == 0:
            return False
        cells = self._project_cortex_to_hippo(cortex_cols)
        if cells.size < 2:
            return False
        self._hebbian_bind(cells)
        self.binds += 1
        self.last_active = cells
        return True

    def _hebbian_bind(self, active_cells: np.ndarray) -> None:
        cells = np.unique(np.asarray(active_cells, dtype=np.int32))
        for source in cells:
            candidates = cells[cells != source]
            if candidates.size > self.cfg.hippo_max_synapses:
                offset = int(source) % candidates.size
                candidates = np.roll(candidates, -offset)[:self.cfg.hippo_max_synapses]
            count = int(self.counts[source])
            for target in candidates:
                existing = np.where(self.targets[source, :count] == target)[0]
                if existing.size:
                    slot = int(existing[0])
                    self.weights[source, slot] = min(
                        1.0,  # HIPPO_WEIGHT_MAX
                        float(self.weights[source, slot]) + self.cfg.hippo_lr,
                    )
                elif count < self.cfg.hippo_max_synapses:
                    self.targets[source, count] = int(target)
                    self.weights[source, count] = 0.50  # HIPPO_INITIAL_PERMANENCE
                    count += 1
                else:
                    slot = int(np.argmin(self.weights[source, :count]))
                    if self.weights[source, slot] <= 0.50:
                        self.targets[source, slot] = int(target)
                        self.weights[source, slot] = 0.50
            self.counts[source] = count

    def retrieve(self, cue_cortex_cols: np.ndarray) -> np.ndarray:
        cue = self._project_cortex_to_hippo(cue_cortex_cols)
        return self.retrieve_ca3(cue)

    def retrieve_ca3(self, cue: np.ndarray) -> np.ndarray:
        """Settle recurrent dynamics from an already projected CA3 cue."""
        cue = np.unique(np.asarray(cue, dtype=np.int32))
        if cue.size == 0:
            return np.array([], dtype=np.int32)
        state = cue
        for _ in range(int(self.cfg.hippo_retrieve_iter)):
            energy = np.zeros(self.cfg.hippo_size, dtype=np.float32)
            for source in state:
                count = int(self.counts[source])
                if count:
                    targets = self.targets[source, :count]
                    np.add.at(energy, targets, self.weights[source, :count])
            positive = np.flatnonzero(energy > 0.0)
            if positive.size == 0:
                break
            k = min(self.cfg.hippo_sparsity, positive.size)
            selected = positive[np.argpartition(energy[positive], -k)[-k:]]
            selected = np.sort(selected.astype(np.int32))
            if np.array_equal(selected, np.sort(state)):
                state = selected
                break
            state = selected
        self.retrievals += 1
        self.last_active = state.copy()
        return self.last_active

    def replay_patterns(self, n: int = 32) -> list[np.ndarray]:
        """Reactivate strong internally stored attractors without an episode database."""
        active_sources = np.flatnonzero(self.counts > 0)
        if active_sources.size == 0 or n <= 0:
            return []
        strength = self.counts[active_sources].astype(np.int64)
        seeds = active_sources[np.argsort(-strength)[: min(n, active_sources.size)]]
        patterns: list[np.ndarray] = []
        for seed in seeds:
            pattern = self.retrieve_ca3(np.array([seed], dtype=np.int32))
            if pattern.size:
                patterns.append(pattern.copy())
        return patterns

    def cortical_columns_for(self, retrieved: np.ndarray) -> np.ndarray:
        if retrieved.size == 0:
            return np.array([], dtype=np.int64)
        cortex_cols = np.arange(self.cfg.sdr_size, dtype=np.int64)
        hashed = (cortex_cols * np.int64(2654435769)) & 0xFFFFFFFF
        hashed = (hashed % self.cfg.hippo_size).astype(np.int32)
        return np.where(np.isin(hashed, retrieved))[0][:self.cfg.sdr_sparsity].astype(np.int64)

    def apply_to_cortex(self, retrieved: np.ndarray, hierarchy, gain: float = 0.2) -> None:
        import warnings
        warnings.warn(
            "apply_to_cortex is deprecated and will be replaced in Phase 6.",
            DeprecationWarning,
            stacklevel=2,
        )
        if retrieved.size == 0:
            return
        candidates = self.cortical_columns_for(retrieved)
        if candidates.size == 0:
            return
        bias = np.zeros(self.cfg.sdr_size, dtype=np.float32)
        bias[candidates] = float(gain)
        hierarchy.layers[0].apply_apical_bias(bias)

    def get_stats(self) -> dict:
        return {
            "binds": self.binds,
            "retrievals": self.retrievals,
            "active_weights": int(self.counts.astype(np.int64).sum()),
            "capacity": int(self.cfg.hippo_size * self.cfg.hippo_max_synapses),
            "memory_bytes": int(
                self.targets.nbytes + self.weights.nbytes + self.counts.nbytes
            ),
        }
