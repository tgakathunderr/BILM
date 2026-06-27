from __future__ import annotations

import numpy as np
from numba import njit

from bilm.config import (
    LEARNING_RATE_HEBB,
    LEARNING_RATE_LTD,
    SYNAPSE_INITIAL_PERMANENCE,
)


@njit(fastmath=True, cache=True)
def apply_predictive_rules_jit(
    prev_winners: np.ndarray,
    current_winners: np.ndarray,
    predictive_cells: np.ndarray,
    active_cells: np.ndarray,
    connected_targets: np.ndarray,
    permanences: np.ndarray,
    synapse_counts: np.ndarray,
    CELLS_PER_COLUMN: int,
    MAX_SYNAPSES_PER_CELL: int,
    learning_rate_hebb: float = LEARNING_RATE_HEBB,
    learning_rate_ltd: float = LEARNING_RATE_LTD,
) -> None:
    """Hebbian growth + potentiation, plus sequential LTD. Local rule only."""
    if len(prev_winners) == 0 or len(current_winners) == 0:
        return

    for w_idx in current_winners:
        was_predicted = predictive_cells[w_idx]
        if was_predicted:
            for pw in prev_winners:
                count = synapse_counts[pw]
                for i in range(count):
                    if connected_targets[pw, i] == w_idx:
                        permanences[pw, i] += learning_rate_hebb
                        if permanences[pw, i] > 1.0:
                            permanences[pw, i] = 1.0
                        break
        else:
            for pw in prev_winners:
                count = synapse_counts[pw]
                exists = False
                for i in range(count):
                    if connected_targets[pw, i] == w_idx:
                        exists = True
                        break
                if not exists and count < MAX_SYNAPSES_PER_CELL:
                    connected_targets[pw, count] = w_idx
                    permanences[pw, count] = SYNAPSE_INITIAL_PERMANENCE
                    synapse_counts[pw] += 1

    predicted_indices = np.where(predictive_cells)[0]
    for p_cell in predicted_indices:
        col_idx = p_cell // CELLS_PER_COLUMN
        col_start = col_idx * CELLS_PER_COLUMN
        is_col_active = False
        for c in range(col_start, col_start + CELLS_PER_COLUMN):
            if active_cells[c]:
                is_col_active = True
                break
        if not is_col_active:
            for pw in prev_winners:
                count = synapse_counts[pw]
                for i in range(count):
                    if connected_targets[pw, i] == p_cell:
                        permanences[pw, i] += learning_rate_ltd
                        if permanences[pw, i] < 0.0:
                            permanences[pw, i] = 0.0
                        break


@njit(fastmath=True, cache=True)
def generate_predictions_jit(
    active_winners: np.ndarray,
    connected_targets: np.ndarray,
    permanences: np.ndarray,
    synapse_counts: np.ndarray,
    TOTAL_CELLS: int,
    SYNAPSE_CONNECTION_THRESHOLD: float,
) -> np.ndarray:
    """Forward pass: active winners cast predictions onto connected targets."""
    predictive_flags = np.zeros(TOTAL_CELLS, dtype=np.bool_)
    for w_idx in active_winners:
        count = synapse_counts[w_idx]
        for i in range(count):
            target = connected_targets[w_idx, i]
            if permanences[w_idx, i] >= SYNAPSE_CONNECTION_THRESHOLD:
                predictive_flags[target] = True
    return predictive_flags


@njit(fastmath=True, cache=True)
def select_winner_cell_jit(
    col_start: int,
    col_end: int,
    active_pws: np.ndarray,
    connected_targets: np.ndarray,
    permanences: np.ndarray,
    synapse_counts: np.ndarray,
    cell_usage: np.ndarray,
    SYNAPSE_CONNECTION_THRESHOLD: float,
) -> int:
    """Winner-cell selection using INCOMING synapses from prev_winners.
    Fixed direction from BIM 3 (old code scored outgoing — always zero)."""
    has_prev = len(active_pws) > 0

    if not has_prev:
        best_cell = col_start
        for cell_idx in range(col_start + 1, col_end):
            if cell_usage[cell_idx] < cell_usage[best_cell]:
                best_cell = cell_idx
        return best_cell

    active_count = min(len(active_pws), 512)

    best_cell = col_start
    best_score = -1
    for cell_idx in range(col_start, col_end):
        score = 0
        for idx in range(active_count):
            pw = active_pws[idx]
            count = synapse_counts[pw]
            for i in range(count):
                if (
                    connected_targets[pw, i] == cell_idx
                    and permanences[pw, i] >= SYNAPSE_CONNECTION_THRESHOLD
                ):
                    score += 1
                    break
        if score > best_score:
            best_score = score
            best_cell = cell_idx
        elif score == best_score and cell_usage[cell_idx] < cell_usage[best_cell]:
            best_cell = cell_idx

    return best_cell


@njit(fastmath=True, cache=True)
def prune_low_permanence_jit(
    permanences: np.ndarray,
    connected_targets: np.ndarray,
    synapse_counts: np.ndarray,
    threshold: float,
) -> int:
    """Compact synapse lists, removing entries below threshold. Returns count pruned."""
    n_cells = permanences.shape[0]
    removed = 0
    for cell in range(n_cells):
        count = synapse_counts[cell]
        if count == 0:
            continue
        write = 0
        for read in range(count):
            if permanences[cell, read] >= threshold:
                if write != read:
                    permanences[cell, write] = permanences[cell, read]
                    connected_targets[cell, write] = connected_targets[cell, read]
                write += 1
            else:
                removed += 1
        for k in range(write, count):
            permanences[cell, k] = 0.0
            connected_targets[cell, k] = -1
        synapse_counts[cell] = write
    return removed
