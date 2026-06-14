from __future__ import annotations

import numpy as np

from bilm.config import (
    APICAL_BIAS_THRESHOLD,
    APICAL_SOURCE_LAYERS,
    CELLS_PER_COLUMN,
    LAYER_DECAY_RATES,
    LEARNING_RATE_HEBB,
    MAX_SYNAPSES_PER_CELL,
    N_CORTICAL_LAYERS,
    SDR_SIZE,
    SDR_SPARSITY,
    SYNAPSE_CONNECTION_THRESHOLD,
)
from bilm.kernels import (
    apply_predictive_rules_jit,
    generate_predictions_jit,
    select_winner_cell_jit,
)

TOTAL_CELLS: int = SDR_SIZE * CELLS_PER_COLUMN


class SparseCortex:
    """One cortical layer. All arrays kept flat for full @njit coverage."""

    def __init__(self) -> None:
        self.connected_targets = np.full(
            (TOTAL_CELLS, MAX_SYNAPSES_PER_CELL), -1, dtype=np.int32
        )
        self.permanences = np.zeros(
            (TOTAL_CELLS, MAX_SYNAPSES_PER_CELL), dtype=np.float32
        )
        self.synapse_counts = np.zeros(TOTAL_CELLS, dtype=np.int32)
        self.active_cells = np.zeros(TOTAL_CELLS, dtype=bool)
        self.winner_cells = np.zeros(TOTAL_CELLS, dtype=bool)
        self.predictive_cells = np.zeros(TOTAL_CELLS, dtype=bool)
        self.prev_winner_cells = np.zeros(TOTAL_CELLS, dtype=bool)
        self.cell_usage = np.zeros(TOTAL_CELLS, dtype=np.int32)

    def step(
        self,
        active_columns_idx: np.ndarray,
        learn: bool = True,
        learn_rate_override: float = -1.0,
    ) -> float:
        """
        Process one SDR step. Returns Jaccard surprise (0=perfect, 1=total mismatch).
        BIM 2 optimization: if surprise == 0.0, skip Hebbian kernel entirely.
        """
        self.prev_winner_cells[:] = self.winner_cells
        self.active_cells.fill(False)
        self.winner_cells.fill(False)

        # Measure surprise: predicted vs actual column overlap
        predicted_cols = set(self.get_predictive_columns().tolist())
        actual_cols = set(int(c) for c in active_columns_idx)
        if predicted_cols or actual_cols:
            intersection = len(predicted_cols & actual_cols)
            union = len(predicted_cols | actual_cols)
            surprise = 1.0 - (intersection / union) if union > 0 else 1.0
        else:
            surprise = 0.0

        active_pws = np.where(self.prev_winner_cells)[0].astype(np.int32)
        for col_idx in active_columns_idx:
            start = int(col_idx) * CELLS_PER_COLUMN
            end = start + CELLS_PER_COLUMN
            predicted = self.predictive_cells[start:end]
            if predicted.any():
                for k in range(CELLS_PER_COLUMN):
                    if predicted[k]:
                        self.active_cells[start + k] = True
                        self.winner_cells[start + k] = True
            else:
                winner = select_winner_cell_jit(
                    start,
                    end,
                    active_pws,
                    self.connected_targets,
                    self.permanences,
                    self.synapse_counts,
                    self.cell_usage,
                    SYNAPSE_CONNECTION_THRESHOLD,
                )
                self.active_cells[winner] = True
                self.winner_cells[winner] = True
                self.cell_usage[winner] += 1

        # BIM 2 optimization: skip learning kernel on perfect prediction
        if learn and surprise > 0.0:
            lr = learn_rate_override if learn_rate_override >= 0.0 else LEARNING_RATE_HEBB
            active_cws = np.where(self.winner_cells)[0].astype(np.int32)
            apply_predictive_rules_jit(
                active_pws,
                active_cws,
                self.predictive_cells,
                self.active_cells,
                self.connected_targets,
                self.permanences,
                self.synapse_counts,
                CELLS_PER_COLUMN,
                MAX_SYNAPSES_PER_CELL,
                lr,
            )

        self._generate_predictions()
        return float(surprise)

    def _generate_predictions(self) -> None:
        active_sources = np.where(self.winner_cells)[0]
        if len(active_sources) == 0:
            self.predictive_cells.fill(False)
            return
        self.predictive_cells = generate_predictions_jit(
            active_sources,
            self.connected_targets,
            self.permanences,
            self.synapse_counts,
            TOTAL_CELLS,
            SYNAPSE_CONNECTION_THRESHOLD,
        )

    def get_predictive_columns(self) -> np.ndarray:
        predictive_indices = np.where(self.predictive_cells)[0]
        if len(predictive_indices) == 0:
            return np.array([], dtype=int)
        return np.unique(predictive_indices // CELLS_PER_COLUMN)

    def apply_apical_bias(self, bias: np.ndarray) -> None:
        """Top-down apical feedback. Threshold fixed at 0.05 (BIM 3 bug fix)."""
        high_bias_cols = np.where(bias > APICAL_BIAS_THRESHOLD)[0]
        for col_idx in high_bias_cols:
            start = col_idx * CELLS_PER_COLUMN
            for k in range(CELLS_PER_COLUMN):
                if self.winner_cells[start + k]:
                    self.predictive_cells[start + k] = True

    def reset_context(self) -> None:
        self.active_cells.fill(False)
        self.winner_cells.fill(False)
        self.predictive_cells.fill(False)
        self.prev_winner_cells.fill(False)

    def total_synapses(self) -> int:
        return int(self.synapse_counts.sum())


class HierarchicalCortex:
    """
    3-layer stacked cortex with graduated temporal pooling and bidirectional
    apical feedback.

    L1: character/byte patterns  (decay 0.80 — short horizon)
    L2: word/phrase patterns     (decay 0.95 — medium horizon)
    L3: sentence structure       (decay 0.99 — long horizon)
    """

    def __init__(self) -> None:
        if len(LAYER_DECAY_RATES) != N_CORTICAL_LAYERS:
            raise ValueError(
                f"LAYER_DECAY_RATES has {len(LAYER_DECAY_RATES)} entries; "
                f"expected {N_CORTICAL_LAYERS}"
            )
        self.layers: list[SparseCortex] = [
            SparseCortex() for _ in range(N_CORTICAL_LAYERS)
        ]
        self.temporal_pools: np.ndarray = np.zeros(
            (N_CORTICAL_LAYERS, SDR_SIZE), dtype=np.float32
        )
        self.decay_rates: np.ndarray = np.asarray(LAYER_DECAY_RATES, dtype=np.float32)

    def step(
        self,
        L1_active_columns: np.ndarray,
        *,
        learn: bool = True,
        learn_rate_override: float = -1.0,
    ) -> float:
        """
        Full cortical step. Returns L1 surprise (used by neuromodulator).
        """
        active_cols = np.asarray(L1_active_columns, dtype=np.int64)
        l1_surprise = 0.0

        # Bottom-up pass
        for i in range(N_CORTICAL_LAYERS):
            surprise = self.layers[i].step(
                active_cols,
                learn=learn,
                learn_rate_override=learn_rate_override,
            )
            if i == 0:
                l1_surprise = surprise
            self.temporal_pools[i] *= self.decay_rates[i]
            if len(active_cols) > 0:
                self.temporal_pools[i, active_cols] += 1.0
            if i + 1 < N_CORTICAL_LAYERS:
                active_cols = self._top_k_of_pool(self.temporal_pools[i])

        # Top-down apical pass
        for src in APICAL_SOURCE_LAYERS:
            dst = src - 1
            if src >= N_CORTICAL_LAYERS or dst < 0:
                continue
            pred_cols = self.layers[src].get_predictive_columns()
            if len(pred_cols) == 0:
                continue
            bias = self._project_apical(src, pred_cols)
            self.layers[dst].apply_apical_bias(bias)

        # Subnormal guard (prevents IEEE 754 subnormal slowdown)
        self.temporal_pools[self.temporal_pools < 1e-30] = 0.0

        return l1_surprise

    def get_predictive_columns(self) -> np.ndarray:
        """L1 predictive columns — used by the Generator to decode next byte."""
        return self.layers[0].get_predictive_columns()

    def get_layer_sdr(self, layer_idx: int) -> np.ndarray:
        return self._top_k_of_pool(self.temporal_pools[layer_idx]).copy()

    def reset_context(self) -> None:
        for layer in self.layers:
            layer.reset_context()
        self.temporal_pools.fill(0.0)

    def total_synapses_per_layer(self) -> list[int]:
        return [layer.total_synapses() for layer in self.layers]

    @staticmethod
    def _top_k_of_pool(pool: np.ndarray) -> np.ndarray:
        if pool.max() <= 0.0:
            return np.array([], dtype=np.int64)
        return np.argpartition(pool, -SDR_SPARSITY)[-SDR_SPARSITY:]

    def _project_apical(self, src_layer: int, src_pred_cols: np.ndarray) -> np.ndarray:
        apical = np.zeros(SDR_SIZE, dtype=np.float32)
        pool = self.temporal_pools[src_layer]
        apical[src_pred_cols] = pool[src_pred_cols]
        max_val = float(apical.max())
        if max_val > 1e-8:
            apical /= max_val
        np.clip(apical, 0.0, 0.3, out=apical)
        return apical
