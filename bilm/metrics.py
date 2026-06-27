from __future__ import annotations

import math
import numpy as np


def target_loss_bits(probabilities: np.ndarray, target: int) -> float:
    """Negative log2 probability assigned to one observed byte."""
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (256,):
        raise ValueError(f"expected 256 probabilities, got {probs.shape}")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(probs.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-9):
        raise ValueError(f"probabilities must sum to 1, got {total}")
    p = max(float(probs[int(target) & 0xFF]), np.finfo(np.float64).tiny)
    return -math.log2(p)


def bits_per_byte(losses: list[float] | np.ndarray) -> float:
    values = np.asarray(losses, dtype=np.float64)
    if not values.size:
        import warnings
        warnings.warn("bits_per_byte received empty input. Returning 0.0.")
        return 0.0
    return float(values.mean())


def byte_perplexity(bpb: float) -> float:
    return float(2.0 ** float(bpb))
