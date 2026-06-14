from __future__ import annotations

import math
import numpy as np

from bilm.codec import ByteCodec
from bilm.config import (
    GEN_MAX_CHARS_DEFAULT,
    GEN_REPETITION_PENALTY,
    GEN_TEMPERATURE,
    GEN_TOP_K,
)


class Generator:
    """Stateless SDR → text decoder. Shares the ByteCodec lookup table."""

    def __init__(self, codec: ByteCodec) -> None:
        self.codec = codec

    def next_byte_argmax(self, predictive_columns: np.ndarray) -> int:
        """Return the single highest-overlap byte. Deterministic."""
        return self.codec.decode_argmax(predictive_columns)

    def next_byte_sample(
        self,
        predictive_columns: np.ndarray,
        temperature: float = GEN_TEMPERATURE,
        top_k: int = GEN_TOP_K,
        recent_bytes: list[int] | None = None,
    ) -> int:
        """
        Temperature-scaled sampling from top-K candidates.
        Applies repetition penalty on bytes seen in the last 3 outputs.
        """
        top_ids, top_scores = self.codec.decode_top_k(predictive_columns, top_k=top_k)
        if len(top_ids) == 0:
            return ord(" ")  # fallback: space

        scores = top_scores.astype(np.float64)

        # Repetition penalty (BIM 3 generation config)
        if recent_bytes:
            recent_set = set(recent_bytes[-3:])
            for i, byte_id in enumerate(top_ids):
                if int(byte_id) in recent_set:
                    scores[i] *= GEN_REPETITION_PENALTY

        # Temperature scaling
        if temperature <= 0.0:
            return int(top_ids[np.argmax(scores)])

        log_scores = np.log(np.maximum(scores, 1e-9)) / temperature
        log_scores -= log_scores.max()
        probs = np.exp(log_scores)
        probs /= probs.sum()

        chosen_idx = int(np.random.choice(len(top_ids), p=probs))
        return int(top_ids[chosen_idx])

    def generate_text(
        self,
        model,
        prompt: str,
        max_chars: int = GEN_MAX_CHARS_DEFAULT,
        temperature: float = GEN_TEMPERATURE,
    ) -> str:
        """
        Generate text by feeding a prompt through the model then sampling.

        Args:
            model:       BILM instance
            prompt:      Seed text (fed in without learning to prime the cortex)
            max_chars:   Number of characters to generate
            temperature: Sampling temperature (0 = argmax, >0 = stochastic)
        """
        # Prime the cortex with the prompt (no learning during generation)
        for b in prompt.encode("utf-8", errors="replace"):
            model.tick(b, learn=False)

        output_bytes: list[int] = []

        for _ in range(max_chars):
            pred_cols = model.cortex.get_predictive_columns()

            if temperature == 0.0:
                next_b = self.next_byte_argmax(pred_cols)
            else:
                next_b = self.next_byte_sample(
                    pred_cols,
                    temperature=temperature,
                    recent_bytes=output_bytes,
                )

            if next_b < 0:
                next_b = ord(" ")

            output_bytes.append(next_b)
            # Feed the chosen byte back as context (without learning)
            model.tick(next_b, learn=False)

        return bytes(output_bytes).decode("utf-8", errors="replace")
