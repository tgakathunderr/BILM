from __future__ import annotations

import re
import numpy as np

from bilm.bilm_config import BILMConfig

# Control tags (BIM 4 thalamus learning)
_TAG_RE = re.compile(r"\[(SLEEP|CHECKPOINT|RESET)\]", re.IGNORECASE)


class ByteCodec:
    """
    Fixed deterministic byte ↔ SDR map with per-symbol frequency tracking
    and BIM 4-style control tag parsing.
    """

    def __init__(self, cfg: BILMConfig | None = None) -> None:
        self.cfg = cfg or BILMConfig()
        self.byte_sdrs: np.ndarray = self._build_byte_sdrs()
        self.active_bits: np.ndarray = self._build_active_bits()
        self.frequencies: np.ndarray = np.zeros(256, dtype=np.int64)
        self.total_seen: int = 0

    def _build_byte_sdrs(self, seed: int = 7) -> np.ndarray:
        """Deterministic SDR per byte. Shape (256, sdr_sparsity) int32."""
        rng = np.random.default_rng(seed)
        table = np.empty((256, self.cfg.sdr_sparsity), dtype=np.int32)
        for b in range(256):
            cols = rng.choice(self.cfg.sdr_size, self.cfg.sdr_sparsity, replace=False).astype(np.int32)
            cols.sort()
            table[b] = cols
        return table

    def _build_active_bits(self) -> np.ndarray:
        """Dense (256, sdr_size) bool matrix for fast decode overlap."""
        bits = np.zeros((256, self.cfg.sdr_size), dtype=bool)
        for b in range(256):
            bits[b, self.byte_sdrs[b]] = True
        return bits

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, byte: int, *, track: bool = True) -> np.ndarray:
        """Encode a single byte to its SDR column indices."""
        b = int(byte) & 0xFF
        if track:
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
        pred_dense = np.zeros(self.cfg.sdr_size, dtype=bool)
        pred_dense[predictive_columns] = True
        overlaps = (self.active_bits & pred_dense).sum(axis=1)
        if int(overlaps.max()) == 0:
            return -1
        return int(np.argmax(overlaps))

    def decode_to_distribution(
        self,
        predictive_columns: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Convert SDR overlap evidence into a normalized byte distribution."""
        if temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")
        if len(predictive_columns) == 0:
            return np.full(256, 1.0 / 256.0, dtype=np.float64)
        scores = self.overlap_scores(predictive_columns)
        logits = scores / float(temperature)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return probs

    def overlap_scores(self, predictive_columns: np.ndarray) -> np.ndarray:
        """Return byte-wise SDR overlap evidence without normalization."""
        if len(predictive_columns) == 0:
            return np.zeros(256, dtype=np.float64)
        pred_dense = np.zeros(self.cfg.sdr_size, dtype=bool)
        pred_dense[np.asarray(predictive_columns, dtype=np.int64)] = True
        return (self.active_bits & pred_dense).sum(axis=1).astype(np.float64)

    def decode_top_k(
        self,
        predictive_columns: np.ndarray,
        top_k: int = 8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (top_k byte ids, overlap scores) sorted descending."""
        if len(predictive_columns) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        pred_dense = np.zeros(self.cfg.sdr_size, dtype=bool)
        pred_dense[predictive_columns] = True
        overlaps = (self.active_bits & pred_dense).sum(axis=1)
        if int(overlaps.max()) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        k = min(int(top_k), 256)
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
        return 1.0 / float(np.log(1.0 + 1.0 + freq))  # SYMBOL_HABITUATION_LOG_OFFSET = 1.0

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
