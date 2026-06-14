from __future__ import annotations

import re
import numpy as np

from bilm.config import (
    CODEC_SDR_SEED,
    N_BYTES,
    SDR_SIZE,
    SDR_SPARSITY,
    SYMBOL_HABITUATION_LOG_OFFSET,
)

# Control tags (BIM 4 thalamus learning)
_TAG_RE = re.compile(r"\[(SLEEP|CHECKPOINT|RESET)\]", re.IGNORECASE)


def _build_byte_sdrs(seed: int = CODEC_SDR_SEED) -> np.ndarray:
    """Deterministic SDR per byte. Shape (N_BYTES, SDR_SPARSITY) int32."""
    rng = np.random.default_rng(seed)
    table = np.empty((N_BYTES, SDR_SPARSITY), dtype=np.int32)
    for b in range(N_BYTES):
        cols = rng.choice(SDR_SIZE, SDR_SPARSITY, replace=False).astype(np.int32)
        cols.sort()
        table[b] = cols
    return table


def _build_active_bits(byte_sdrs: np.ndarray) -> np.ndarray:
    """Dense (N_BYTES, SDR_SIZE) bool matrix for fast decode overlap."""
    bits = np.zeros((N_BYTES, SDR_SIZE), dtype=bool)
    for b in range(N_BYTES):
        bits[b, byte_sdrs[b]] = True
    return bits


class ByteCodec:
    """
    Fixed deterministic byte ↔ SDR map with per-symbol frequency tracking
    and BIM 4-style control tag parsing.
    """

    def __init__(self) -> None:
        self.byte_sdrs: np.ndarray = _build_byte_sdrs()
        self.active_bits: np.ndarray = _build_active_bits(self.byte_sdrs)
        self.frequencies: np.ndarray = np.zeros(N_BYTES, dtype=np.int64)
        self.total_seen: int = 0

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, byte: int) -> np.ndarray:
        """Encode a single byte to its SDR column indices."""
        b = int(byte) & 0xFF
        self.frequencies[b] += 1
        self.total_seen += 1
        return self.byte_sdrs[b]

    def encode_text(self, text: str) -> list[np.ndarray]:
        """Encode a string as a list of SDRs (UTF-8 bytes)."""
        return [self.encode(b) for b in text.encode("utf-8", errors="replace")]

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode_argmax(self, predictive_columns: np.ndarray) -> int:
        """Return the single most-likely byte given the Cortex's predicted columns."""
        if len(predictive_columns) == 0:
            return -1
        pred_dense = np.zeros(SDR_SIZE, dtype=bool)
        pred_dense[predictive_columns] = True
        overlaps = (self.active_bits & pred_dense).sum(axis=1)
        if int(overlaps.max()) == 0:
            return -1
        return int(np.argmax(overlaps))

    def decode_top_k(
        self,
        predictive_columns: np.ndarray,
        top_k: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (top_k byte ids, overlap scores) sorted descending."""
        if len(predictive_columns) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        pred_dense = np.zeros(SDR_SIZE, dtype=bool)
        pred_dense[predictive_columns] = True
        overlaps = (self.active_bits & pred_dense).sum(axis=1)
        if int(overlaps.max()) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        k = min(int(top_k), N_BYTES)
        top_ids = np.argpartition(overlaps, -k)[-k:].astype(np.int32)
        order = np.argsort(-overlaps[top_ids])
        top_ids = top_ids[order]
        return top_ids, overlaps[top_ids]

    # ------------------------------------------------------------------
    # Habituation (BIM 3)
    # ------------------------------------------------------------------

    def habituation_scale(self, byte: int) -> float:
        """Per-symbol learning-rate scale ∈ (0, 1]. Common bytes habituate."""
        b = int(byte) & 0xFF
        freq = float(self.frequencies[b])
        return 1.0 / float(np.log(SYMBOL_HABITUATION_LOG_OFFSET + 1.0 + freq))

    # ------------------------------------------------------------------
    # Tag parsing (BIM 4)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_tags(text: str) -> tuple[str, list[str]]:
        """
        Strip control tags from text. Returns (clean_text, list_of_tags).
        Tags: [SLEEP], [CHECKPOINT], [RESET]
        """
        tags_found = [m.group(1).upper() for m in _TAG_RE.finditer(text)]
        clean = _TAG_RE.sub("", text).strip()
        return clean, tags_found
