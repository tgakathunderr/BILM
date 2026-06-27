from __future__ import annotations

import numpy as np

from bilm.bilm_config import BILMConfig
from bilm.kernels import prune_low_permanence_jit


class Homeostasis:
    """Per-BILM synaptic rescale + prune. One instance per model."""

    def __init__(self, cfg: BILMConfig | None = None) -> None:
        self.cfg = cfg or BILMConfig()
        # Compute target sums dynamically using ratios (0.625, 0.600, 0.575)
        ratios = (0.625, 0.600, 0.575)
        self.target_sums = np.asarray(
            [max(self.cfg.max_synapses * r, 4.0) for r in ratios], dtype=np.float32
        )
        self.every_n = int(self.cfg.homeostasis_every_n)
        self.prune_threshold = float(self.cfg.perm_prune)
        self.step_count = 0
        self.applications = 0
        self.total_pruned = 0

        # Safety check: target_sum must keep scaled permanences above connection threshold
        min_target = self.cfg.max_synapses * self.cfg.perm_threshold
        for idx, target in enumerate(self.target_sums):
            if float(target) < min_target:
                raise ValueError(
                    f"Homeostasis target sum[{idx}]={target} is below "
                    f"the safety floor {min_target:.2f}."
                )

    def maybe_apply(self, cortex) -> bool:
        """Call once per token. Applies homeostasis every `every_n` steps."""
        self.step_count += 1
        if self.step_count % self.every_n != 0:
            return False
        for layer_idx, layer in enumerate(cortex.layers):
            target = float(self.target_sums[min(layer_idx, len(self.target_sums) - 1)])
            self._rescale(layer, target, self.cfg.perm_threshold)
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
    def _rescale(layer, target_sum: float, perm_threshold: float) -> None:
        floor = perm_threshold * 1.1  # HOMEOSTASIS_FLOOR_RATIO = 1.1
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
