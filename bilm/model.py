from __future__ import annotations

import os
import numpy as np

from bilm.codec import ByteCodec
from bilm.cortex import HierarchicalCortex
from bilm.neuromod import Neuromod
from bilm.homeostasis import Homeostasis
from bilm.hippocampus import Hippocampus
from bilm.generator import Generator
from bilm.config import CHECKPOINT_DIR


class BILM:
    """
    Biologically Inspired Language Model.

    Usage:
        model = BILM()
        model.train_on_file("data/wiki.txt")
        text = model.generate("The quick brown")
    """

    def __init__(self) -> None:
        self.codec = ByteCodec()
        self.cortex = HierarchicalCortex()
        self.neuromod = Neuromod()
        self.homeostasis = Homeostasis()
        self.hippocampus = Hippocampus()
        self.generator = Generator(self.codec)
        self.tokens_seen: int = 0

    # ------------------------------------------------------------------
    # Core tick — BIM 4 tick() architecture
    # ------------------------------------------------------------------

    def tick(self, byte_val: int, learn: bool = True) -> int:
        """
        Process one byte through the full biological pipeline.
        Returns the predicted next byte (argmax).

        Pipeline order:
          1. Codec:       byte → SDR
          2. Cortex:      SDR → prediction + surprise (Hebbian if learn=True)
          3. Neuromod:    surprise → ACh update
          4. Homeostasis: fire every N tokens
          5. Hippocampus: bind if surprise high; retrieve and apply apical bias
          6. Generator:   predicted columns → next byte
        """
        # 1. Encode
        sdr_cols = self.codec.encode(byte_val)

        # 2. Cortex step
        hab_scale = self.codec.habituation_scale(byte_val)
        lr = self.neuromod.lr_scale(hab_scale) if learn else 0.0
        surprise = self.cortex.step(sdr_cols, learn=learn, learn_rate_override=lr)

        # 3. Neuromod update
        if learn:
            self.neuromod.update(surprise)

        # 4. Homeostasis
        if learn:
            self.homeostasis.maybe_apply(self.cortex)

        # 5. Hippocampus
        if learn:
            current_cols = self.cortex.get_predictive_columns()
            bound = self.hippocampus.maybe_bind(sdr_cols, surprise)
            if not bound and surprise > 0.3:
                # Try retrieval to inject long-context prior
                retrieved = self.hippocampus.retrieve(sdr_cols)
                if retrieved.size > 0:
                    self.hippocampus.apply_to_cortex(retrieved, self.cortex)

        # 6. Generate prediction for next byte
        pred_cols = self.cortex.get_predictive_columns()
        next_byte = self.generator.next_byte_argmax(pred_cols)

        if learn:
            self.tokens_seen += 1

        return next_byte

    # ------------------------------------------------------------------
    # Sleep consolidation (BIM 4)
    # ------------------------------------------------------------------

    def sleep(self) -> None:
        """
        Offline consolidation. Not implemented as replay in BILM v1 because
        the Hippocampus stores weight-space attractors, not episode lists.
        Sleep in BILM = reset cortical context so the next sequence starts fresh.
        The Hippocampus weights persist across sleep.
        """
        self.cortex.reset_context()

    # ------------------------------------------------------------------
    # Text interface
    # ------------------------------------------------------------------

    def train_on_text(self, text: str) -> list[float]:
        """
        Train on a string, byte by byte. Returns per-token BPC list.
        BPC = -log2(P(correct)) ≈ -log2(overlap/SDR_SPARSITY)
        """
        bpc_log: list[float] = []
        data = text.encode("utf-8", errors="replace")
        for b in data:
            next_pred = self.tick(int(b), learn=True)
            # Approximate BPC from surprise in next neuromod state
            surprise = 1.0 - (self.neuromod.ach - 0.10) / 0.90
            surprise = max(1e-9, min(1.0, surprise))
            bpc = -float(np.log2(max(1e-9, 1.0 - surprise)))
            bpc_log.append(bpc)
        return bpc_log

    def generate(
        self,
        prompt: str,
        max_chars: int = 200,
        temperature: float = 0.8,
    ) -> str:
        """Generate text from a prompt."""
        return self.generator.generate_text(
            self,
            prompt=prompt,
            max_chars=max_chars,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> str:
        """Save all weight matrices to a .npz file."""
        if path is None:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            path = os.path.join(CHECKPOINT_DIR, "bilm.npz")

        arrays: dict[str, np.ndarray] = {}

        # Cortex layers
        for i, layer in enumerate(self.cortex.layers):
            arrays[f"cortex_L{i}_targets"] = layer.connected_targets
            arrays[f"cortex_L{i}_perms"] = layer.permanences
            arrays[f"cortex_L{i}_counts"] = layer.synapse_counts
            arrays[f"cortex_L{i}_usage"] = layer.cell_usage

        arrays["cortex_pools"] = self.cortex.temporal_pools

        # Hippocampus
        arrays["hippo_W"] = self.hippocampus.W

        # Codec frequencies
        arrays["codec_frequencies"] = self.codec.frequencies

        # Metadata
        arrays["tokens_seen"] = np.array([self.tokens_seen], dtype=np.int64)

        np.savez_compressed(path, **arrays)
        return path

    def load(self, path: str) -> None:
        """Load weights from a .npz checkpoint."""
        data = np.load(path)

        for i, layer in enumerate(self.cortex.layers):
            layer.connected_targets[:] = data[f"cortex_L{i}_targets"]
            layer.permanences[:] = data[f"cortex_L{i}_perms"]
            layer.synapse_counts[:] = data[f"cortex_L{i}_counts"]
            layer.cell_usage[:] = data[f"cortex_L{i}_usage"]

        self.cortex.temporal_pools[:] = data["cortex_pools"]
        self.hippocampus.W[:] = data["hippo_W"]
        self.codec.frequencies[:] = data["codec_frequencies"]
        self.tokens_seen = int(data["tokens_seen"][0])

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "tokens_seen": self.tokens_seen,
            "neuromod": self.neuromod.get_state(),
            "homeostasis": self.homeostasis.get_stats(),
            "hippocampus": self.hippocampus.get_stats(),
            "synapses_per_layer": self.cortex.total_synapses_per_layer(),
        }
