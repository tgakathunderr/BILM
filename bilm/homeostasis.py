"""
BILM — homeostasis.py
Synaptic rescaling and pruning to prevent saturation.

Ported from BIM 3 learning/homeostasis.py.
Fires every HOMEOSTASIS_EVERY_N tokens. Two operations:
  1. Rescale: cells whose permanence sum exceeds target are scaled down.
  2. Prune:   synapses below SYNAPSE_PRUNE_THRESHOLD are deleted entirely.
"""
from __future__ import annotations

import numpy as np

from bilm.config import (
    HOMEOSTASIS_EVERY_N,
    HOMEOSTASIS_FLOOR_RATIO,
    HOMEOSTASIS_TARGET_SUM_PER_LAYER,
    MAX_SYNAPSES_PER_CELL,
    SYNAPSE_CONNECTION_THRESHOLD,
    SYNAPSE_PRUNE_THRESHOLD,
)
from bilm.kernels import prune_low_permanence_jit


class Homeostasis:
    """Per-BILM synaptic rescale + prune. One instance per model."""

    def __init__(self) -> None:
        self.target_sums = np.asarray(HOMEOSTASIS_TARGET_SUM_PER_LAYER, dtype=np.float32)
        self.every_n = int(HOMEOSTASIS_EVERY_N)
        self.prune_threshold = float(SYNAPSE_PRUNE_THRESHOLD)
        self.step_count = 0
        self.applications = 0
        self.total_pruned = 0

        # Safety check: target_sum must keep scaled permanences above connection threshold
        min_target = MAX_SYNAPSES_PER_CELL * SYNAPSE_CONNECTION_THRESHOLD
        for idx, target in enumerate(self.target_sums):
            if float(target) < min_target:
                raise ValueError(
                    f"HOMEOSTASIS_TARGET_SUM_PER_LAYER[{idx}]={target} is below "
                    f"the safety floor {min_target:.2f}."
                )

    def maybe_apply(self, cortex) -> bool:
        """Call once per token. Applies homeostasis every `every_n` steps."""
        self.step_count += 1
        if self.step_count % self.every_n != 0:
            return False
        for layer_idx, layer in enumerate(cortex.layers):
            target = float(self.target_sums[min(layer_idx, len(self.target_sums) - 1)])
            self._rescale(layer, target)
            self.total_pruned += int(
                prune_low_permanence_jit(
                    layer.permanences,
                    layer.connected_targets,
                    layer.synapse_counts,
                    self.prune_threshold,
                )
            )
        self.applications += 1
        return True

    @staticmethod
    def _rescale(layer, target_sum: float) -> None:
        floor = SYNAPSE_CONNECTION_THRESHOLD * HOMEOSTASIS_FLOOR_RATIO
        counts = layer.synapse_counts
        perms = layer.permanences
        for i in range(len(counts)):
            n = int(counts[i])
            if n == 0:
                continue
            cell_sum = float(perms[i, :n].sum())
            if cell_sum > target_sum * 1.2:
                scale = target_sum / cell_sum
                perms[i, :n] *= scale
                np.maximum(perms[i, :n], floor, out=perms[i, :n])

    def get_stats(self) -> dict:
        return {
            "step_count": self.step_count,
            "applications": self.applications,
            "total_pruned": self.total_pruned,
        }
