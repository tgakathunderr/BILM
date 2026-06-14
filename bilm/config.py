"""
BILM — Biologically Inspired Language Model
config.py: All hyperparameters in one place.

Sources:
  BIM 1: SDR geometry (16384 × 64)
  BIM 3: Cortex, homeostasis, hippocampus, synapse thresholds
  BIM 4: Variance-based ACh habituation window
"""
from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# SDR / Cortex geometry  [BIM 1]
# ---------------------------------------------------------------------------
SDR_SIZE: int = 16_384
SDR_SPARSITY: int = 64
CELLS_PER_COLUMN: int = 8
MAX_SYNAPSES_PER_CELL: int = 24

# ---------------------------------------------------------------------------
# Synaptic plasticity  [BIM 3]
# ---------------------------------------------------------------------------
SYNAPSE_CONNECTION_THRESHOLD: float = 0.50
SYNAPSE_INITIAL_PERMANENCE: float = 0.55
LEARNING_RATE_HEBB: float = 0.20
LEARNING_RATE_LTD: float = -0.10
SYNAPSE_PRUNE_THRESHOLD: float = 0.40

# ---------------------------------------------------------------------------
# Hierarchical cortex  [BIM 3, bug-fixed]
# ---------------------------------------------------------------------------
N_CORTICAL_LAYERS: int = 3
LAYER_DECAY_RATES: tuple[float, ...] = (0.80, 0.95, 0.99)
APICAL_SOURCE_LAYERS: tuple[int, ...] = (1, 2)
APICAL_BIAS_THRESHOLD: float = 0.05   # BIM 3 bug fix — was 0.50

# ---------------------------------------------------------------------------
# Synaptic homeostasis  [BIM 3]
# ---------------------------------------------------------------------------
_HOMEOSTASIS_RATIOS: tuple[float, ...] = (0.625, 0.600, 0.575)
HOMEOSTASIS_TARGET_SUM_PER_LAYER: tuple[float, ...] = tuple(
    max(MAX_SYNAPSES_PER_CELL * r, 4.0) for r in _HOMEOSTASIS_RATIOS
)
HOMEOSTASIS_EVERY_N: int = 500
HOMEOSTASIS_FLOOR_RATIO: float = 1.1

# ---------------------------------------------------------------------------
# Hippocampus  [BIM 3]
# ---------------------------------------------------------------------------
HIPPO_SIZE: int = 8_192
HIPPO_SPARSITY: int = 164
HIPPO_LEARNING_RATE: float = 0.50
HIPPO_LTD: float = -0.02
HIPPO_RETRIEVE_ITER: int = 8
HIPPO_BIND_SURPRISE_THRESHOLD: float = 0.60
HIPPO_WEIGHT_MAX: float = 1.0

# ---------------------------------------------------------------------------
# Neuromodulator (ACh)  [BIM 3 base + BIM 4 variance habituation]
# ---------------------------------------------------------------------------
SURPRISE_ROLLING_WINDOW: int = 64
LR_SCALE_MIN: float = 0.10
LR_SCALE_MAX: float = 1.00
SYMBOL_HABITUATION_LOG_OFFSET: float = 1.0
ACH_VARIANCE_WINDOW: int = 10          # BIM 4: variance check window
ACH_VARIANCE_THRESHOLD: float = 0.01  # BIM 4: if variance < this AND surprise high → halve ACh

# ---------------------------------------------------------------------------
# Codec  [BIM 3]
# ---------------------------------------------------------------------------
N_BYTES: int = 256
CODEC_SDR_SEED: int = 7

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
GEN_TEMPERATURE: float = 0.8
GEN_TOP_K: int = 32
GEN_REPETITION_PENALTY: float = 0.6
GEN_MAX_CHARS_DEFAULT: int = 200

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
SLEEP_EVERY_N_TOKENS: int = 50_000
REPORT_EVERY_N_TOKENS: int = 1_000
CHECKPOINT_EVERY_N_TOKENS: int = 100_000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR: str = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR: str = os.path.join(BASE_DIR, "logs")
DATA_DIR: str = os.path.join(BASE_DIR, "data")
