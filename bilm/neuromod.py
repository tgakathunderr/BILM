"""
BILM — neuromod.py
Acetylcholine neuromodulator: surprise + symbol habituation → learning rate.

Sources:
  BIM 3 core/neuromod.py:  Rolling-mean surprise, per-symbol log habituation
  BIM 4 brainstem.py:      Variance-based habituation (prevents ACh saturation
                            on predictable-but-repetitive patterns like spaces)

The BIM 4 addition is critical for BILM: without variance checking, training on
English text saturates ACh permanently on common function words ("the", "is", "of"),
which drives max learning rate on every token and destabilizes the cortex.
"""
from __future__ import annotations

from collections import deque

from bilm.config import (
    ACH_VARIANCE_THRESHOLD,
    ACH_VARIANCE_WINDOW,
    LR_SCALE_MAX,
    LR_SCALE_MIN,
    SURPRISE_ROLLING_WINDOW,
)


class Neuromod:
    """
    ACh modulator. One instance per BILM.

    ACh = f(rolling_surprise, per_symbol_habituation, variance_check)
    """

    def __init__(self) -> None:
        self._surprise_history: deque[float] = deque(maxlen=SURPRISE_ROLLING_WINDOW)
        self._variance_window: deque[float] = deque(maxlen=ACH_VARIANCE_WINDOW)
        self.ach: float = LR_SCALE_MIN  # current ACh level

    def update(self, surprise: float) -> float:
        """
        Update ACh given the latest surprise signal.
        Returns the new ACh value (= learning rate scale for the cortex).
        """
        self._surprise_history.append(float(surprise))
        self._variance_window.append(float(surprise))

        rolling_mean = (
            sum(self._surprise_history) / len(self._surprise_history)
            if self._surprise_history else 1.0
        )

        # BIM 4 variance-based habituation:
        # If surprise has been consistently high but low-variance, it is
        # predictable noise (e.g., a repeated character), NOT genuine novelty.
        # Halve ACh to prevent saturation-driven plasticity on noise.
        effective_surprise = rolling_mean
        if len(self._variance_window) == ACH_VARIANCE_WINDOW:
            mean_v = sum(self._variance_window) / ACH_VARIANCE_WINDOW
            variance = sum((x - mean_v) ** 2 for x in self._variance_window) / ACH_VARIANCE_WINDOW
            if variance < ACH_VARIANCE_THRESHOLD and rolling_mean > 0.5:
                effective_surprise *= 0.5

        # Map effective surprise [0, 1] → [LR_SCALE_MIN, LR_SCALE_MAX]
        self.ach = LR_SCALE_MIN + (LR_SCALE_MAX - LR_SCALE_MIN) * effective_surprise
        self.ach = max(LR_SCALE_MIN, min(LR_SCALE_MAX, self.ach))
        return self.ach

    def lr_scale(self, habituation: float = 1.0) -> float:
        """Combined learning rate = ACh * per-symbol habituation factor."""
        scale = self.ach * float(habituation)
        return max(LR_SCALE_MIN, min(LR_SCALE_MAX, scale))

    def get_state(self) -> dict:
        return {
            "ACh": round(self.ach, 4),
            "rolling_surprise": round(
                sum(self._surprise_history) / max(1, len(self._surprise_history)), 4
            ),
        }
