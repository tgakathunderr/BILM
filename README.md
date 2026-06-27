# BILM 2 — Lifelong Language Substrate

BILM is an experimental, byte-native language model built around sparse cortical
sequence learning, online local plasticity, neuromodulation, homeostasis, and
hippocampal associative memory.

The current release is **2.0.0 alpha**. It provides valid autoregressive probability
distributions and bits-per-byte evaluation. Its intended research advantage is
continual online learning with lower catastrophic forgetting on consumer CPUs.

## Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | Complete | Defect confirmation: predictive_cells in-place write, context capture with cell_usage, evaluation idempotency |
| Phase 1 | Complete | Semantic correctness: v2 API, side-effect-free evaluation, checkpoint v2 roundtrip, determinism, typed result objects |
| Phase 2 | Complete | Byte LM: local probabilistic readout, unigram/bigram/trigram baselines, grammar generators, learning curves |
| Phase 3 | Complete | Hierarchy: ablation framework (apical, homeostasis, CA3, neuromod, L1-only), temporal pooling |
| Phase 4 | Complete | Memory: CA3 binding/retrieval, bounded sparse recurrent, consolidation replay |
| Phase 5 | Complete | Benchmark: 5-domain continual learning (English, code, scientific, legal, multilingual), 3 domain orders |
| Phase 6 | Complete | Scaling: memory profile, throughput profile, checkpoint profile, scaling curves |
| Phase 7 | Complete | Challenger: BILM vs baselines comparison with criteria checking |

### Architecture

```
       [Raw Byte Stream]
              │
      ┌───────▼───────┐
      │ Sensory Codec │ (Hash-based SDR projection, N=16384, w=64)
      └───────┬───────┘
              │ (0.39% Sparse SDR)
      ┌───────▼────────┐
      │  Predictive    ◄──────────────────┐ Apical Feedback
      │    Cortex      ├──────────────┐   │ (Contextual Bias)
      └───────┬────────┘              │   │
              │                       ▼───┴───┐
              │ High surprise       ┌─────────┴─┐
              └─────────────────────►   CA3     │
                                        Hippocampus│ (Recurrent Attractor)
                                        └───────────┘
```

### Bug Fixes (Phase 0)

- `cortex.py`: `_generate_predictions()` now writes `predictive_cells` in-place
  instead of replacing the array reference, eliminating memory churn.
- `model.py`: `_capture_context()` now captures **all** mutable state including
  `cell_usage`, codec frequencies, neuromod windows, homeostasis counters, and
  hippocampus state. `_restore_context()` restores everything.
- `model.py`: `generate()` API now uses `max_bytes` parameter per spec.
- `model.py`: Added `BILM.from_checkpoint()` classmethod.
- `baselines.py`: Added `UnigramByteLM` baseline.
- `benchmark.py`: Fixed RSS measurement to use actual process RSS instead of
  hardcoded 268.4 MB estimate.

### Evaluation Contract

- `predict_next()` returns the distribution before seeing the current byte.
- `observe(byte)` scores the prior prediction against the byte, then learns.
- `evaluate()` captures all state, processes bytes with `learn=False`, and
  restores all state. It is provably side-effect-free.
- Autoregressive alignment: prediction from context through position `i` is
  scored against byte `i`.

## Installation

```bash
pip install -e .
pip install -e ".[dev]"
```

Python 3.9+ is required. Dependencies: `numpy>=1.20.0`, `numba>=0.57.0`.

## API

```python
from bilm import BILM

model = BILM()
prediction = model.predict_next()
result = model.observe(ord("A"), learn=True)
print(result.loss_bits, result.next_prediction.argmax)

report = model.evaluate(b"held-out bytes", warmup=4)
print(report.bits_per_byte, report.perplexity, report.accuracy)

text = model.generate("The quick brown", max_bytes=200, temperature=0.8)

model.save("checkpoint.npz")
model = BILM.from_checkpoint("checkpoint.npz")
```

### Result Types

- `Prediction`: `probabilities[256]`, `argmax`, `predictive_columns`, `confidence`
- `ObservationResult`: `target`, `prior_prediction`, `next_prediction`, `loss_bits`, `surprise`
- `EvaluationReport`: `tokens`, `bits_per_byte`, `perplexity`, `accuracy`

## Training and Evaluation

```bash
python -m bilm.train --text data/corpus.txt --max-tokens 100000
python -m bilm.benchmark --test accuracy --train-tokens 100000
python -m pytest -q
```

## Continual Learning

```bash
python -m bilm.continual_benchmark \
  --model bilm \
  --domain english:data/english-train.bin:data/english-eval.bin \
  --domain code:data/code-train.bin:data/code-eval.bin

python -m bilm.cl_benchmark --domain-size 5000 --eval-size 1000
```

## Experiments

```bash
python -m bilm.run_experiments --experiment all --train-size 5000
python -m bilm.challenger --domain-size 3000 --eval-size 500
python -m bilm.scaling
```

### Synthetic Grammars

7 deterministic grammar generators for controlled evaluation:
- `ab_pattern`, `abc_pattern` — repeating patterns
- `simple_grammar` — balanced a^n b^n
- `nested_structure` — bracket nesting
- `delayed_copy_10`, `delayed_copy_20` — long-range context
- `language` — mini-English with word structure

### Ablation Configurations

- `full` — no ablation (control)
- `no_apical` — disable apical feedback
- `no_homeostasis` — disable synaptic homeostasis
- `no_hippocampus` — disable CA3 memory
- `no_neuromod` — fix learning rate (no ACh)
- `l1_only` — single cortical layer

## Test Suite

64 tests covering:
- Autoregressive alignment and no target leakage
- Evaluation side-effect-free contract (including `cell_usage`)
- Probability finiteness, normalization, calibration
- Checkpoint v2 roundtrip (bit-identical)
- Determinism and seed reproducibility
- CA3 binding, retrieval, bounded memory
- Consolidation replay
- No NaN, unbounded memory, or runaway synapses
- RSS, TPS, checkpoint size, latency budgets

## Research Rules

- Acquisition and retention are always reported together.
- Evaluation never updates learned or adaptive state.
- Biological mechanisms require controlled ablations before capability claims.
- Every published number identifies code version, configuration, dataset hash,
  domain order, seed, hardware, and raw output.
- "Better than Transformers" is initially a hypothesis about continual learning
  at near-matched language quality, not a general claim.

## License

MIT
