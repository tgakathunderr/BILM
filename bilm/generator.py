from __future__ import annotations

import math
import numpy as np

from bilm.codec import ByteCodec
from bilm.bilm_config import BILMConfig


class Generator:
    """Stateless SDR → text decoder. Shares the ByteCodec lookup table."""

    def __init__(self, cfg: BILMConfig | None = None, codec: ByteCodec | None = None) -> None:
        self.cfg = cfg or BILMConfig()
        self.codec = codec or ByteCodec(self.cfg)
        self.rng = np.random.default_rng(7)

    def next_byte_argmax(self, predictive_columns: np.ndarray) -> int:
        """Return the single highest-overlap byte. Deterministic."""
        return self.codec.decode_argmax(predictive_columns)

    def next_byte_sample(
        self,
        predictive_columns: np.ndarray,
        temperature: float | None = None,
        top_k: int | None = None,
        recent_bytes: list[int] | None = None,
    ) -> int:
        """
        Temperature-scaled sampling from top-K candidates.
        Applies repetition penalty on bytes seen in the last 3 outputs.
        """
        temp = self.cfg.gen_temperature if temperature is None else temperature
        tk = self.cfg.gen_top_k if top_k is None else top_k

        scores = self.codec.overlap_scores(predictive_columns).astype(np.float64)

        # Repetition penalty
        if recent_bytes:
            recent_set = set(recent_bytes[-3:])
            for b in recent_set:
                scores[b] *= self.cfg.gen_repetition_penalty

        if temp <= 0.0:
            return int(np.argmax(scores))

        # Temperature division before np.argpartition
        logits = np.log(np.maximum(scores, 1e-9)) / float(temp)
        logits -= logits.max()

        k = min(int(tk), len(logits))
        top_ids = np.argpartition(logits, -k)[-k:]
        top_logits = logits[top_ids]

        # Sort descending
        order = np.argsort(-top_logits)
        top_ids = top_ids[order]
        top_logits = top_logits[order]

        # Softmax over top-K
        probs = np.exp(top_logits)
        probs /= probs.sum()

        chosen_idx = int(self.rng.choice(len(top_ids), p=probs))
        return int(top_ids[chosen_idx])

    def generate_text(
        self,
        model,
        prompt: str,
        max_chars: int = 200,
        temperature: float | None = None,
    ) -> str:
        """
        Generate text by feeding a prompt through the model then sampling.
        """
        temp = self.cfg.gen_temperature if temperature is None else temperature

        # Prime the cortex with the prompt (no learning during generation)
        for b in prompt.encode("utf-8", errors="replace"):
            model.tick(b, learn=False)

        output_bytes: list[int] = []

        for _ in range(max_chars):
            prediction = model.predict_next()
            probabilities = prediction.probabilities.copy()
            if output_bytes:
                for byte_id in set(output_bytes[-3:]):
                    probabilities[byte_id] *= self.cfg.gen_repetition_penalty
                probabilities /= probabilities.sum()

            if temp <= 0.0:
                next_b = int(np.argmax(probabilities))
            else:
                logits = np.log(np.maximum(probabilities, 1e-12)) / temp
                logits -= logits.max()
                sample_probs = np.exp(logits)
                sample_probs /= sample_probs.sum()
                next_b = int(self.rng.choice(len(sample_probs), p=sample_probs))

            if next_b < 0:
                next_b = ord(" ")

            output_bytes.append(next_b)
            # Feed the chosen byte back as context (without learning)
            model.tick(next_b, learn=False)

        return bytes(output_bytes).decode("utf-8", errors="replace")
