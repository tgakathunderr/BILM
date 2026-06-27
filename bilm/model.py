from __future__ import annotations

import os
import json
import numpy as np

from bilm.codec import ByteCodec
from bilm.cortex import HierarchicalCortex
from bilm.neuromod import Neuromod
from bilm.homeostasis import Homeostasis
from bilm.hippocampus import Hippocampus
from bilm.generator import Generator
from bilm.deep_readout import DeepAssociativeReadout
from bilm.dap import DenseAssociativeProjection
from bilm.fales import FeedbackAlignment
from bilm.srs import SparseRecurrentState
from bilm.config import CHECKPOINT_DIR
from bilm.metrics import target_loss_bits, bits_per_byte, byte_perplexity
from bilm.results import Prediction, ObservationResult, EvaluationReport
from bilm.bilm_config import BILMConfig


class BILM:
    """
    Biologically Inspired Language Model.

    Usage:
        model = BILM()
        model.train_on_file("data/wiki.txt")
        text = model.generate("The quick brown")
    """

    def __init__(self, config: BILMConfig | None = None) -> None:
        self.cfg         = config or BILMConfig()
        self.codec       = ByteCodec(self.cfg)
        self.cortex      = HierarchicalCortex(self.cfg)
        self.neuromod    = Neuromod(self.cfg)
        self.homeostasis = Homeostasis(self.cfg)
        self.hippocampus = Hippocampus(self.cfg)
        self.generator   = Generator(self.cfg, self.codec)
        self.readout     = DeepAssociativeReadout(self.cfg)
        self.dap         = DenseAssociativeProjection(self.cfg)
        self.feedback    = FeedbackAlignment(self.cfg)
        self.srs         = SparseRecurrentState(self.cfg)
        self._lookahead: list[tuple[np.ndarray, np.ndarray]] = []
        self.tokens_seen: int = 0

    # ------------------------------------------------------------------
    # Core tick — BIM 4 tick() architecture
    # ------------------------------------------------------------------

    def predict_next(self, temperature: float = 1.0) -> Prediction:
        """Return the next-byte distribution implied by the current context."""
        columns = self.cortex.get_predictive_columns().astype(np.int64, copy=True)
        srs_bias = self.srs.step(columns)

        ctx       = self.cortex.get_layer_sdr(self.cfg.n_layers - 1)
        retrieved = self.hippocampus.retrieve(ctx)

        if retrieved.size > 0:
            alpha   = self.cfg.hippo_readout_alpha
            p_fresh = self.readout.predict(columns, self.codec, temperature=temperature, logit_bias=srs_bias)
            p_hippo = self.readout.predict(retrieved.astype(np.int64), self.codec, temperature=temperature, logit_bias=srs_bias)
            probabilities = (1 - alpha) * p_fresh + alpha * p_hippo
        else:
            probabilities = self.readout.predict(columns, self.codec, temperature=temperature, logit_bias=srs_bias)

        argmax = int(np.argmax(probabilities))
        return Prediction(
            probabilities=probabilities,
            argmax=argmax,
            predictive_columns=columns,
            confidence=float(probabilities[argmax]),
        )

    def observe(self, byte_val: int, learn: bool = True) -> ObservationResult:
        """Score and consume one byte, optionally applying online learning."""
        target = int(byte_val) & 0xFF
        prior = self.predict_next()
        loss = target_loss_bits(prior.probabilities, target)

        if learn:
            self.readout.learn(
                prior.predictive_columns, prior.probabilities, target
            )

        sdr_cols = self.codec.encode(target, track=learn)

        # DAP projection passed as apical bias to Layer 0
        dap_dense = self.dap.project(sdr_cols)
        dap_bias = dap_dense @ self.dap.W.T
        self.cortex.layers[0].apply_apical_bias(dap_bias)

        hab_scale = self.codec.habituation_scale(target)
        lr = self.neuromod.lr_scale(hab_scale) if learn else 0.0
        surprise = self.cortex.step(sdr_cols, learn=learn, learn_rate_override=lr)

        # Step SRS with target sdr_cols to keep state in sync
        self.srs.step(sdr_cols)

        if learn:
            self.neuromod.update(surprise)
            self.homeostasis.maybe_apply(self.cortex)

            cortical_context = self.cortex.get_layer_sdr(self.cfg.n_layers - 1)
            self.hippocampus.maybe_bind(cortical_context, surprise)

            # FALES 3-step lookahead
            self._lookahead.append((prior.predictive_columns.copy(), prior.probabilities.copy()))
            if len(self._lookahead) > self.cfg.lookahead_steps + 1:
                self._lookahead.pop(0)
            if len(self._lookahead) >= self.cfg.lookahead_steps:
                old_cols, old_probs = self._lookahead[0]
                delayed_err = np.zeros(256, np.float32)
                delayed_err[target] = 1.0
                delayed_err -= old_probs
                dlr = lr * self.cfg.lookahead_lr_scale
                for i, layer in enumerate(self.cortex.layers):
                    self.feedback.apply(layer, i, delayed_err, dlr)

            self.tokens_seen += 1

        return ObservationResult(
            target=target,
            prior_prediction=prior,
            next_prediction=self.predict_next(),
            loss_bits=loss,
            surprise=float(surprise),
        )

    def tick(self, byte_val: int, learn: bool = True) -> int:
        """
        Process one byte through the full biological pipeline.
        Returns the predicted next byte (argmax).
        """
        return self.observe(byte_val, learn=learn).next_prediction.argmax

    def evaluate(
        self,
        data: bytes,
        *,
        warmup: int = 0,
        reset_context: bool = True,
    ) -> EvaluationReport:
        """Evaluate autoregressively without mutating persistent model state."""
        context = self._capture_context()
        losses: list[float] = []
        correct = 0
        try:
            if reset_context:
                self.cortex.reset_context()
            for index, value in enumerate(data):
                result = self.observe(value, learn=False)
                if index >= warmup:
                    losses.append(result.loss_bits)
                    correct += int(result.prior_prediction.argmax == int(value))
        finally:
            self._restore_context(context)
        bpb = bits_per_byte(losses)
        return EvaluationReport(
            tokens=len(losses),
            bits_per_byte=bpb,
            perplexity=byte_perplexity(bpb),
            accuracy=(correct / len(losses)) if losses else 0.0,
        )

    def _capture_context(self) -> dict:
        return {
            "pools": self.cortex.temporal_pools.copy(),
            "layers": [
                (
                    layer.active_cells.copy(),
                    layer.winner_cells.copy(),
                    layer.predictive_cells.copy(),
                    layer.prev_winner_cells.copy(),
                    layer.cell_usage.copy(),
                )
                for layer in self.cortex.layers
            ],
            "codec_frequencies": self.codec.frequencies.copy(),
            "codec_total_seen": self.codec.total_seen,
            "readout_state": self.readout.save_state() if hasattr(self.readout, "save_state") else {
                "weights": self.readout.weights.copy(),
                "bias": self.readout.bias.copy(),
            },
            "srs_state": self.srs.state.copy(),
            "lookahead": [(cols.copy(), probs.copy()) for cols, probs in self._lookahead],
            "neuromod_surprise": list(self.neuromod._surprise_history),
            "neuromod_variance": list(self.neuromod._variance_window),
            "neuromod_ach": self.neuromod.ach,
            "homeostasis_step_count": self.homeostasis.step_count,
            "homeostasis_applications": self.homeostasis.applications,
            "homeostasis_total_pruned": self.homeostasis.total_pruned,
            "hippo_binds": self.hippocampus.binds,
            "hippo_retrievals": self.hippocampus.retrievals,
            "hippo_last_active": self.hippocampus.last_active.copy(),
            "tokens_seen": self.tokens_seen,
        }

    def _restore_context(self, state: dict) -> None:
        self.cortex.temporal_pools[:] = state["pools"]
        for layer, values in zip(self.cortex.layers, state["layers"]):
            layer.active_cells[:], layer.winner_cells[:], layer.predictive_cells[:], layer.prev_winner_cells[:] = values[:4]
            layer.cell_usage[:] = values[4]
        self.codec.frequencies[:] = state["codec_frequencies"]
        self.codec.total_seen = state["codec_total_seen"]
        if hasattr(self.readout, "load_state"):
            self.readout.load_state(state["readout_state"])
        else:
            self.readout.weights[:] = state["readout_state"]["weights"]
            self.readout.bias[:] = state["readout_state"]["bias"]
        self.srs.state[:] = state["srs_state"]
        self._lookahead = [(cols.copy(), probs.copy()) for cols, probs in state["lookahead"]]
        self.neuromod._surprise_history.clear()
        self.neuromod._surprise_history.extend(state["neuromod_surprise"])
        self.neuromod._variance_window.clear()
        self.neuromod._variance_window.extend(state["neuromod_variance"])
        self.neuromod.ach = state["neuromod_ach"]
        self.homeostasis.step_count = state["homeostasis_step_count"]
        self.homeostasis.applications = state["homeostasis_applications"]
        self.homeostasis.total_pruned = state["homeostasis_total_pruned"]
        self.hippocampus.binds = state["hippo_binds"]
        self.hippocampus.retrievals = state["hippo_retrievals"]
        self.hippocampus.last_active = state["hippo_last_active"]
        self.tokens_seen = state["tokens_seen"]

    # ------------------------------------------------------------------
    # Sleep consolidation (BIM 4)
    # ------------------------------------------------------------------

    def sleep(self, n_replays: int = 32, learning_rate: float = 0.02) -> dict:
        """
        Offline neural consolidation through internally reactivated CA3 attractors.
        """
        replays = 0
        self.cortex.reset_context()
        for pattern in self.hippocampus.replay_patterns(n_replays):
            columns = self.hippocampus.cortical_columns_for(pattern)
            if columns.size:
                self.cortex.step(
                    columns,
                    learn=True,
                    learn_rate_override=float(learning_rate),
                )
                replays += 1
        self.cortex.reset_context()
        return {"replays": replays, "learning_rate": float(learning_rate)}

    # ------------------------------------------------------------------
    # Text interface
    # ------------------------------------------------------------------

    def train_on_text(self, text: str) -> list[float]:
        """
        Train on a string, byte by byte. Returns per-token BPC list.
        """
        bpc_log: list[float] = []
        data = text.encode("utf-8", errors="replace")
        for b in data:
            result = self.observe(int(b), learn=True)
            bpc_log.append(result.loss_bits)
        return bpc_log

    def generate(
        self,
        prompt: str,
        max_bytes: int = 200,
        temperature: float = 0.8,
    ) -> str:
        """Generate text from a prompt."""
        return self.generator.generate_text(
            self,
            prompt=prompt,
            max_chars=max_bytes,
            temperature=temperature,
        )

    def train_on_file(self, path: str, max_tokens: int | None = None) -> None:
        """Train on a file, byte by byte."""
        with open(path, "rb") as f:
            data = f.read()
        if max_tokens is not None:
            data = data[:max_tokens]
        for b in data:
            self.observe(int(b), learn=True)

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
        arrays["hippo_targets"] = self.hippocampus.targets
        arrays["hippo_weights"] = self.hippocampus.weights
        arrays["hippo_counts"] = self.hippocampus.counts
        if hasattr(self.readout, "save_state"):
            for k, v in self.readout.save_state().items():
                arrays[k] = v
        else:
            arrays["readout_weights"] = self.readout.weights
            arrays["readout_bias"] = self.readout.bias
            arrays["readout_updates"] = np.array([self.readout.updates], dtype=np.int64)
        if hasattr(self, "dap"):
            arrays["dap_W"] = self.dap.W
        if hasattr(self, "srs"):
            arrays["srs_W_k"] = self.srs.W_k
            arrays["srs_W_v"] = self.srs.W_v
            arrays["srs_W_q"] = self.srs.W_q
            arrays["srs_W_proj"] = self.srs.W_proj
            arrays["srs_decay"] = self.srs.decay
            arrays["srs_state"] = self.srs.state

        # Codec frequencies
        arrays["codec_frequencies"] = self.codec.frequencies
        arrays["codec_total_seen"] = np.array([self.codec.total_seen], dtype=np.int64)
        arrays["neuromod_surprise"] = np.asarray(self.neuromod._surprise_history, dtype=np.float64)
        arrays["neuromod_variance"] = np.asarray(self.neuromod._variance_window, dtype=np.float64)
        arrays["neuromod_ach"] = np.array([self.neuromod.ach], dtype=np.float64)
        arrays["homeostasis_state"] = np.array([
            self.homeostasis.step_count,
            self.homeostasis.applications,
            self.homeostasis.total_pruned,
        ], dtype=np.int64)
        arrays["hippo_last_active"] = self.hippocampus.last_active
        arrays["hippo_counters"] = np.array([
            self.hippocampus.binds, self.hippocampus.retrievals
        ], dtype=np.int64)
        for i, layer in enumerate(self.cortex.layers):
            arrays[f"cortex_L{i}_active"] = layer.active_cells
            arrays[f"cortex_L{i}_winner"] = layer.winner_cells
            arrays[f"cortex_L{i}_predictive"] = layer.predictive_cells
            arrays[f"cortex_L{i}_prev_winner"] = layer.prev_winner_cells
        rng_json = json.dumps(self.generator.rng.bit_generator.state).encode("utf-8")
        arrays["generator_rng"] = np.frombuffer(rng_json, dtype=np.uint8)
        arrays["checkpoint_version"] = np.array([2], dtype=np.int64)

        # Metadata
        arrays["tokens_seen"] = np.array([self.tokens_seen], dtype=np.int64)

        np.savez_compressed(path, **arrays)
        return path

    def load(self, path: str) -> None:
        """Load weights from a .npz checkpoint."""
        data = np.load(path, allow_pickle=False)

        for i, layer in enumerate(self.cortex.layers):
            layer.connected_targets[:] = data[f"cortex_L{i}_targets"]
            layer.permanences[:] = data[f"cortex_L{i}_perms"]
            layer.synapse_counts[:] = data[f"cortex_L{i}_counts"]
            layer.cell_usage[:] = data[f"cortex_L{i}_usage"]
            for suffix, target in (
                ("active", layer.active_cells),
                ("winner", layer.winner_cells),
                ("predictive", layer.predictive_cells),
                ("prev_winner", layer.prev_winner_cells),
            ):
                key = f"cortex_L{i}_{suffix}"
                if key in data:
                    target[:] = data[key]

        self.cortex.temporal_pools[:] = data["cortex_pools"]
        if "hippo_targets" in data:
            self.hippocampus.targets[:] = data["hippo_targets"]
            self.hippocampus.weights[:] = data["hippo_weights"]
            self.hippocampus.counts[:] = data["hippo_counts"]
        if hasattr(self.readout, "load_state") and "dar_W_embed" in data:
            state = {k: data[k] for k in data.files if k.startswith("dar_")}
            self.readout.load_state(state)
        elif "readout_weights" in data and hasattr(self.readout, "weights"):
            self.readout.weights[:] = data["readout_weights"]
            self.readout.bias[:] = data["readout_bias"]
            self.readout.updates = int(data["readout_updates"][0])
        if "dap_W" in data and hasattr(self, "dap"):
            self.dap.W[:] = data["dap_W"]
        if "srs_state" in data and hasattr(self, "srs"):
            self.srs.W_k[:] = data["srs_W_k"]
            self.srs.W_v[:] = data["srs_W_v"]
            self.srs.W_q[:] = data["srs_W_q"]
            self.srs.W_proj[:] = data["srs_W_proj"]
            self.srs.decay[:] = data["srs_decay"]
            self.srs.state[:] = data["srs_state"]
        self.codec.frequencies[:] = data["codec_frequencies"]
        self.tokens_seen = int(data["tokens_seen"][0])
        if "codec_total_seen" in data:
            self.codec.total_seen = int(data["codec_total_seen"][0])
        if "neuromod_surprise" in data:
            self.neuromod._surprise_history.clear()
            self.neuromod._variance_window.clear()
            self.neuromod._surprise_history.extend(data["neuromod_surprise"].tolist())
            self.neuromod._variance_window.extend(data["neuromod_variance"].tolist())
            self.neuromod.ach = float(data["neuromod_ach"][0])
        if "homeostasis_state" in data:
            hs = data["homeostasis_state"]
            self.homeostasis.step_count = int(hs[0])
            self.homeostasis.applications = int(hs[1])
            self.homeostasis.total_pruned = int(hs[2])
        if "hippo_last_active" in data:
            self.hippocampus.last_active = data["hippo_last_active"].copy()
            self.hippocampus.binds = int(data["hippo_counters"][0])
            self.hippocampus.retrievals = int(data["hippo_counters"][1])
        if "generator_rng" in data:
            state = json.loads(data["generator_rng"].tobytes().decode("utf-8"))
            self.generator.rng.bit_generator.state = state

    @classmethod
    def from_checkpoint(cls, path: str, config: BILMConfig | None = None) -> "BILM":
        """Create a new BILM instance and load state from a checkpoint."""
        model = cls(config)
        model.load(path)
        return model

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
