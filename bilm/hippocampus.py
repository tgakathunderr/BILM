import numba
import numpy as np
from bilm.bilm_config import BILMConfig


@numba.njit
def hebbian_bind_jit(
    active_cells: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    hippo_max_synapses: int,
    hippo_lr: float,
) -> None:
    cells = np.unique(active_cells)
    for source in cells:
        candidates = cells[cells != source]
        if candidates.size > hippo_max_synapses:
            offset = int(source) % candidates.size
            n = candidates.size
            rolled = np.empty_like(candidates)
            for idx in range(n):
                rolled[idx] = candidates[(idx + offset) % n]
            candidates = rolled[:hippo_max_synapses]
            
        count = int(counts[source])
        for target in candidates:
            existing_idx = -1
            for idx in range(count):
                if targets[source, idx] == target:
                    existing_idx = idx
                    break
            
            if existing_idx != -1:
                weights[source, existing_idx] = min(
                    1.0,
                    float(weights[source, existing_idx]) + hippo_lr,
                )
            elif count < hippo_max_synapses:
                targets[source, count] = int(target)
                weights[source, count] = 0.50
                count += 1
            else:
                min_idx = 0
                min_val = weights[source, 0]
                for idx in range(1, count):
                    if weights[source, idx] < min_val:
                        min_val = weights[source, idx]
                        min_idx = idx
                if min_val <= 0.50:
                    targets[source, min_idx] = int(target)
                    weights[source, min_idx] = 0.50
        counts[source] = count


@numba.njit
def retrieve_ca3_jit(
    cue: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    counts: np.ndarray,
    hippo_size: int,
    hippo_retrieve_iter: int,
    hippo_sparsity: int,
) -> np.ndarray:
    cue = np.unique(cue)
    if cue.size == 0:
        return np.empty(0, dtype=np.int32)
    state = cue
    for _ in range(hippo_retrieve_iter):
        energy = np.zeros(hippo_size, dtype=np.float32)
        for source in state:
            count = int(counts[source])
            if count > 0:
                for idx in range(count):
                    t = targets[source, idx]
                    energy[t] += weights[source, idx]
        
        pos_count = 0
        positive = np.empty(hippo_size, dtype=np.int32)
        for idx in range(hippo_size):
            if energy[idx] > 0.0:
                positive[pos_count] = idx
                pos_count += 1
        
        if pos_count == 0:
            break
            
        k = min(hippo_sparsity, pos_count)
        pos_energy = np.empty(pos_count, dtype=np.float32)
        pos_indices = np.empty(pos_count, dtype=np.int32)
        for idx in range(pos_count):
            pos_indices[idx] = positive[idx]
            pos_energy[idx] = energy[positive[idx]]
            
        args = np.argsort(pos_energy)
        selected = pos_indices[args[-k:]]
        selected = np.sort(selected)
        
        if selected.size == state.size:
            same = True
            for idx in range(selected.size):
                if selected[idx] != state[idx]:
                    same = False
                    break
            if same:
                state = selected
                break
        state = selected
        
    return state


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
        hebbian_bind_jit(
            active_cells.astype(np.int32),
            self.targets,
            self.weights,
            self.counts,
            self.cfg.hippo_max_synapses,
            self.cfg.hippo_lr,
        )

    def retrieve(self, cue_cortex_cols: np.ndarray) -> np.ndarray:
        cue = self._project_cortex_to_hippo(cue_cortex_cols)
        return self.retrieve_ca3(cue)

    def retrieve_ca3(self, cue: np.ndarray) -> np.ndarray:
        """Settle recurrent dynamics from an already projected CA3 cue."""
        ret = retrieve_ca3_jit(
            cue.astype(np.int32),
            self.targets,
            self.weights,
            self.counts,
            self.cfg.hippo_size,
            int(self.cfg.hippo_retrieve_iter),
            self.cfg.hippo_sparsity,
        )
        self.retrievals += 1
        self.last_active = ret.copy()
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
