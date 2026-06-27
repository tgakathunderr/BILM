"""
BILMConfig: fully arbitrary model configuration.

Transformers don't hardcode sizes — you pass d_model=768, n_layers=12 and it
works. BILM works the same way. Named presets are reference examples, not limits.

    # Fully custom:
    model = BILM(BILMConfig(sdr_size=32768, n_layers=4,
                             layer_decays=(0.80, 0.90, 0.97, 0.99),
                             dar_hidden_1=1024, dar_hidden_2=512))

    # Preset as starting point, override freely:
    cfg = BILMConfig.from_preset("300m").replace(n_layers=8, dar_hidden_1=3072)
    model = BILM(cfg)

    # Check before allocating:
    print(cfg.param_count(), cfg.ram_estimate_gb())

    # Save/load alongside checkpoint:
    cfg.save("run_cfg.json")
    cfg = BILMConfig.load("run_cfg.json")
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class BILMConfig:
    # ── SDR geometry — scale sdr_size like d_model ─────────────────────────
    sdr_size: int = 16_384
    cells_per_column: int = 8
    max_synapses: int = 24
    sdr_sparsity: int = 64

    # ── Cortex depth — scale n_layers like n_layers ────────────────────────
    n_layers: int = 3
    layer_decays: tuple[float, ...] = (0.80, 0.95, 0.99)
    apical_source_layers: tuple[int, ...] = (1, 2)
    apical_bias_threshold: float = 0.05

    # ── Synaptic plasticity ────────────────────────────────────────────────
    lr_hebb: float = 0.20
    lr_ltd: float = -0.10
    perm_threshold: float = 0.50
    perm_initial: float = 0.55
    perm_prune: float = 0.40

    # ── Hippocampus ────────────────────────────────────────────────────────
    hippo_size: int = 8_192
    hippo_sparsity: int = 164
    hippo_lr: float = 0.50
    hippo_ltd: float = -0.02
    hippo_retrieve_iter: int = 8
    hippo_bind_threshold: float = 0.60
    hippo_max_synapses: int = 64

    # ── Deep Associative Readout (DAR) — replaces LocalByteReadout ─────────
    dar_hidden_1: int = 512        # scale like ffn_dim
    dar_hidden_2: int = 256
    dar_batch_size: int = 64
    dar_embed_lr: float = 0.005
    dar_hidden_lr: float = 0.003
    dar_output_lr: float = 0.001
    dar_output_lr_decay: float = 5e-6  # CL fix: decaying lr prevents Adam forgetting
    dar_adam_beta1: float = 0.9
    dar_adam_beta2: float = 0.999
    dar_adam_eps: float = 1e-8

    # ── Hippocampus readout mixture (Phase 6) ──────────────────────────────
    hippo_readout_alpha: float = 0.20  # 20% hippo, 80% fresh cortex

    # ── Neuromodulation ────────────────────────────────────────────────────
    surprise_window: int = 64
    ach_min: float = 0.10
    ach_max: float = 1.00
    ach_variance_window: int = 10
    ach_variance_threshold: float = 0.01

    # ── Homeostasis ────────────────────────────────────────────────────────
    homeostasis_every_n: int = 500

    # ── Lookahead (Phase 4) ─────────────────────────────────────────────────
    lookahead_steps: int = 3
    lookahead_lr_scale: float = 0.30

    # ── SRS (Phase 5) ──────────────────────────────────────────────────────
    srs_dim: int = 512
    srs_decay: float = 0.95

    # ── DAP (Phase 3) ──────────────────────────────────────────────────────
    dap_dim: int = 1024
    dap_lr: float = 0.002

    # ── Generator ──────────────────────────────────────────────────────────
    gen_temperature: float = 0.8
    gen_top_k: int = 32
    gen_repetition_penalty: float = 0.6

    # ── Training ───────────────────────────────────────────────────────────
    sleep_every_n: int = 50_000
    checkpoint_every_n: int = 100_000
    report_every_n: int = 1_000

    def __post_init__(self) -> None:
        if len(self.layer_decays) != self.n_layers:
            raise ValueError(
                f"layer_decays has {len(self.layer_decays)} entries "
                f"but n_layers={self.n_layers}. Must match."
            )
        if self.sdr_sparsity >= self.sdr_size:
            raise ValueError("sdr_sparsity must be < sdr_size")
        if self.dar_hidden_1 < self.dar_hidden_2:
            raise ValueError("dar_hidden_1 must be >= dar_hidden_2")

    def param_count(self) -> int:
        total_cells = self.sdr_size * self.cells_per_column
        cortex = self.n_layers * total_cells * self.max_synapses
        hippo  = self.hippo_size * self.hippo_sparsity
        dar    = (self.sdr_size * self.dar_hidden_1
                  + self.dar_hidden_1 * self.dar_hidden_2
                  + self.dar_hidden_2 * 256)
        return cortex + hippo + dar

    def ram_estimate_gb(self) -> float:
        return self.param_count() * 4 / (1024 ** 3)

    def replace(self, **kwargs) -> "BILMConfig":
        d = asdict(self)
        d.update(kwargs)
        for f in ("layer_decays", "apical_source_layers"):
            if f in d and not isinstance(d[f], tuple):
                d[f] = tuple(d[f])
        return BILMConfig(**d)

    def save(self, path: str) -> None:
        d = asdict(self)
        d["layer_decays"] = list(d["layer_decays"])
        d["apical_source_layers"] = list(d["apical_source_layers"])
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: str) -> "BILMConfig":
        d = json.loads(Path(path).read_text())
        d["layer_decays"] = tuple(d["layer_decays"])
        d["apical_source_layers"] = tuple(d["apical_source_layers"])
        return cls(**d)

    @classmethod
    def from_preset(cls, name: str) -> "BILMConfig":
        if name not in BILM_PRESETS:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(BILM_PRESETS.keys())}. "
                f"Or pass BILMConfig() directly with any values."
            )
        return BILM_PRESETS[name]

    def describe(self) -> str:
        return (
            f"BILMConfig(sdr={self.sdr_size:,}, layers={self.n_layers}, "
            f"cells={self.cells_per_column}, hippo={self.hippo_size:,}, "
            f"dar={self.dar_hidden_1}->{self.dar_hidden_2}) "
            f"| params~{self.param_count():,} | ram~{self.ram_estimate_gb():.2f}GB"
        )


# ── Reference presets — EXAMPLES, not limits ──────────────────────────────
# Named by approximate parameter count. Pass BILMConfig() directly for anything else.
BILM_PRESETS: dict[str, BILMConfig] = {
    "micro": BILMConfig(
        sdr_size=8_192, cells_per_column=4, max_synapses=16, sdr_sparsity=32,
        n_layers=2, layer_decays=(0.80, 0.99),
        hippo_size=2_048, dar_hidden_1=256, dar_hidden_2=128,
        srs_dim=128, dap_dim=256,
        # ~0.3M params | ~50MB | ~2000 TPS | testing
    ),
    "300m": BILMConfig(
        sdr_size=65_536, cells_per_column=8, max_synapses=32, sdr_sparsity=128,
        n_layers=6, layer_decays=(0.70, 0.80, 0.90, 0.95, 0.98, 0.99),
        hippo_size=32_768, dar_hidden_1=2048, dar_hidden_2=1024,
        srs_dim=1024, dap_dim=2048,
        # ~300M params | ~2GB | ~80 TPS
    ),
    "1b": BILMConfig(
        sdr_size=262_144, cells_per_column=8, max_synapses=48, sdr_sparsity=256,
        n_layers=12,
        layer_decays=(0.60, 0.70, 0.80, 0.87, 0.92, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999),
        hippo_size=131_072, dar_hidden_1=4096, dar_hidden_2=2048,
        srs_dim=2048, dap_dim=4096,
        # ~1B params | ~8GB | ~15 TPS | needs Numba
    ),
    "7b": BILMConfig(
        sdr_size=1_048_576, cells_per_column=8, max_synapses=64, sdr_sparsity=512,
        n_layers=20, layer_decays=tuple(0.60 + 0.39 * i / 19 for i in range(20)),
        hippo_size=524_288, dar_hidden_1=8192, dar_hidden_2=4096,
        srs_dim=4096, dap_dim=8192,
        # ~7B params | ~56GB | ~2 TPS | cluster
    ),
}
