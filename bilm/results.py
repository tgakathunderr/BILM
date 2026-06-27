from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Prediction:
    probabilities: np.ndarray
    argmax: int
    predictive_columns: np.ndarray
    confidence: float


@dataclass(frozen=True)
class ObservationResult:
    target: int
    prior_prediction: Prediction
    next_prediction: Prediction
    loss_bits: float
    surprise: float


@dataclass(frozen=True)
class EvaluationReport:
    tokens: int
    bits_per_byte: float
    perplexity: float
    accuracy: float
