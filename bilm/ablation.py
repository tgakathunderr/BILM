"""Ablation experiments for BILM mechanisms."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, asdict

import numpy as np

from bilm import BILM
from bilm.config import (
    LEARNING_RATE_HEBB,
    N_CORTICAL_LAYERS,
    APICAL_SOURCE_LAYERS,
    HOMEOSTASIS_EVERY_N,
    SURPRISE_ROLLING_WINDOW,
)


@dataclass
class AblationConfig:
    name: str
    description: str
    apply_fn: object

    def apply(self, model: BILM) -> BILM:
        return self.apply_fn(model)


def _disable_apical_feedback(model: BILM) -> BILM:
    """Remove top-down apical feedback from higher layers."""
    original_step = model.cortex.step.__func__

    def patched_step(self, L1_active_columns, *, learn=True, learn_rate_override=-1.0):
        active_cols = np.asarray(L1_active_columns, dtype=np.int64)
        l1_surprise = 0.0
        for i in range(self.cfg.n_layers):
            surprise = self.layers[i].step(
                active_cols, learn=learn, learn_rate_override=learn_rate_override,
            )
            if i == 0:
                l1_surprise = surprise
            self.temporal_pools[i] *= self.decay_rates[i]
            if len(active_cols) > 0:
                self.temporal_pools[i, active_cols] += 1.0
            if i + 1 < self.cfg.n_layers:
                active_cols = self._top_k_of_pool(self.temporal_pools[i])
        self.temporal_pools[self.temporal_pools < 1e-30] = 0.0
        return l1_surprise

    import types
    model.cortex.step = types.MethodType(patched_step, model.cortex)
    return model


def _disable_temporal_pooling(model: BILM) -> BILM:
    """Zero temporal pooling — each layer gets raw input."""
    model.cortex.disable_temporal_pooling = True
    model.cortex.temporal_pools[:] = 0.0
    return model


def _disable_homeostasis(model: BILM) -> BILM:
    """Disable synaptic homeostasis."""
    model.homeostasis.every_n = 10**15
    return model


def _disable_hippocampus(model: BILM) -> BILM:
    """Disable CA3 episodic memory."""
    original_observe = model.observe.__func__

    def patched_observe(self, byte_val: int, learn: bool = True):
        from bilm.metrics import target_loss_bits
        target = int(byte_val) & 0xFF
        prior = self.predict_next()
        loss = target_loss_bits(prior.probabilities, target)
        if learn:
            self.readout.learn(prior.predictive_columns, prior.probabilities, target)
        sdr_cols = self.codec.encode(target, track=learn)
        hab_scale = self.codec.habituation_scale(target)
        lr = self.neuromod.lr_scale(hab_scale) if learn else 0.0
        surprise = self.cortex.step(sdr_cols, learn=learn, learn_rate_override=lr)
        if learn:
            self.neuromod.update(surprise)
            self.homeostasis.maybe_apply(self.cortex)
            self.tokens_seen += 1
        from bilm.results import ObservationResult
        return ObservationResult(
            target=target,
            prior_prediction=prior,
            next_prediction=self.predict_next(),
            loss_bits=loss,
            surprise=float(surprise),
        )

    import types
    model.observe = types.MethodType(patched_observe, model)
    return model


def _disable_neuromodulation(model: BILM) -> BILM:
    """Fix learning rate to constant, ignoring surprise."""
    model.neuromod.lr_scale = lambda habituation=1.0: model.cfg.lr_hebb
    return model


def _single_layer_only(model: BILM) -> BILM:
    """Only use L1 (no hierarchy)."""
    for i in range(1, model.cfg.n_layers):
        model.cortex.layers[i].reset_context()

    original_step = model.cortex.step.__func__

    def patched_step(self, L1_active_columns, *, learn=True, learn_rate_override=-1.0):
        active_cols = np.asarray(L1_active_columns, dtype=np.int64)
        surprise = self.layers[0].step(
            active_cols, learn=learn, learn_rate_override=learn_rate_override,
        )
        self.temporal_pools[0] *= self.decay_rates[0]
        if len(active_cols) > 0:
            self.temporal_pools[0, active_cols] += 1.0
        for i in range(1, self.cfg.n_layers):
            self.temporal_pools[i] *= self.decay_rates[i]
        self.temporal_pools[self.temporal_pools < 1e-30] = 0.0
        return surprise

    import types
    model.cortex.step = types.MethodType(patched_step, model.cortex)
    return model


ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig("full", "No ablation (control)", lambda m: m),
    AblationConfig("no_apical", "Disable apical feedback", _disable_apical_feedback),
    AblationConfig("no_homeostasis", "Disable homeostasis", _disable_homeostasis),
    AblationConfig("no_hippocampus", "Disable CA3 memory", _disable_hippocampus),
    AblationConfig("no_neuromod", "Fix learning rate (no ACh)", _disable_neuromodulation),
    AblationConfig("l1_only", "Single cortical layer", _single_layer_only),
]


def run_ablation(
    train_data: bytes,
    eval_data: bytes,
    configs: list[AblationConfig] | None = None,
    warmup: int = 256,
) -> list[dict]:
    """Run ablation experiment comparing different mechanism configurations."""
    if configs is None:
        configs = ABLATION_CONFIGS

    results = []
    for config in configs:
        model = BILM()
        if config.name != "full":
            model = config.apply(model)

        t0 = time.time()
        for b in train_data:
            model.observe(int(b), learn=True)
        train_time = time.time() - t0

        report = model.evaluate(
            eval_data,
            warmup=min(warmup, max(0, len(eval_data) - 1)),
            reset_context=True,
        )

        results.append({
            "mechanism": config.name,
            "description": config.description,
            "enabled": config.name == "full" or not config.name.startswith("no_"),
            "bpb": report.bits_per_byte,
            "accuracy": report.accuracy,
            "perplexity": report.perplexity,
            "wall_seconds": train_time,
            "tokens": len(train_data),
        })

    return results
