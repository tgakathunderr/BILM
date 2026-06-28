from __future__ import annotations

from collections import defaultdict, deque
import numpy as np

from bilm.metrics import target_loss_bits, bits_per_byte, byte_perplexity
from bilm.results import EvaluationReport

# Re-export new PyTorch baselines
from bilm.baselines.tiny_transformer import TinyTransformer
from bilm.baselines.lstm_lm import LSTM_LM
from bilm.baselines.minimal_ssm import MinimalSSM


class UnigramByteLM:
    """Unigram byte baseline with additive smoothing."""

    def __init__(self, alpha: float = 0.5) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.counts: np.ndarray = np.zeros(256, dtype=np.int64)

    def predict_next(self) -> np.ndarray:
        evidence = self.counts.astype(np.float64) + self.alpha
        return evidence / evidence.sum()

    def observe(self, byte: int, learn: bool = True) -> float:
        target = int(byte) & 0xFF
        loss = target_loss_bits(self.predict_next(), target)
        if learn:
            self.counts[target] += 1
        return loss

    def train(self, data: bytes) -> None:
        for value in data:
            self.observe(value, learn=True)

    def evaluate(self, data: bytes, warmup: int = 0) -> EvaluationReport:
        losses: list[float] = []
        correct = 0
        for index, value in enumerate(data):
            probs = self.predict_next()
            loss = self.observe(value, learn=False)
            if index >= warmup:
                losses.append(loss)
                correct += int(int(np.argmax(probs)) == int(value))
        bpb = bits_per_byte(losses)
        return EvaluationReport(
            tokens=len(losses),
            bits_per_byte=bpb,
            perplexity=byte_perplexity(bpb),
            accuracy=(correct / len(losses)) if losses else 0.0,
        )


class NGramByteLM:
    """Bounded byte n-gram baseline with additive smoothing."""

    def __init__(self, order: int = 2, alpha: float = 0.5) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.order = int(order)
        self.alpha = float(alpha)
        self.context: deque[int] = deque(maxlen=max(0, self.order - 1))
        self.counts: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(256, dtype=np.int64)
        )

    def _key(self) -> tuple[int, ...]:
        return tuple(self.context)

    def predict_next(self) -> np.ndarray:
        counts = self.counts.get(self._key())
        if counts is None:
            return np.full(256, 1.0 / 256.0, dtype=np.float64)
        evidence = counts.astype(np.float64) + self.alpha
        return evidence / evidence.sum()

    def observe(self, byte: int, learn: bool = True) -> float:
        target = int(byte) & 0xFF
        loss = target_loss_bits(self.predict_next(), target)
        if learn:
            self.counts[self._key()][target] += 1
        self.context.append(target)
        return loss

    def train(self, data: bytes) -> None:
        for value in data:
            self.observe(value, learn=True)

    def evaluate(self, data: bytes, warmup: int = 0) -> EvaluationReport:
        saved = deque(self.context, maxlen=self.context.maxlen)
        self.context.clear()
        losses: list[float] = []
        correct = 0
        try:
            for index, value in enumerate(data):
                probs = self.predict_next()
                loss = self.observe(value, learn=False)
                if index >= warmup:
                    losses.append(loss)
                    correct += int(int(np.argmax(probs)) == int(value))
        finally:
            self.context = saved
        bpb = bits_per_byte(losses)
        return EvaluationReport(
            tokens=len(losses),
            bits_per_byte=bpb,
            perplexity=byte_perplexity(bpb),
            accuracy=(correct / len(losses)) if losses else 0.0,
        )


__all__ = [
    "UnigramByteLM",
    "NGramByteLM",
    "TinyTransformer",
    "LSTM_LM",
    "MinimalSSM",
]
