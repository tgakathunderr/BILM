# BILM (Biologically Inspired Language Model)

BILM is a continuously-learning, CPU-native alternative to Transformers. It utilizes a biologically inspired architecture based on Sparse Distributed Representations (SDRs), a predictive cortical hierarchy, and an episodic CA3 Hippocampus.

Unlike Transformers, BILM is designed to learn continuously from a streaming input of bytes without ever suffering from catastrophic forgetting. It achieves competitive character-level language modeling performance while training and generating text at >100 Tokens-Per-Second on a standard consumer CPU.

## Key Features

1. **Zero Catastrophic Forgetting:** Fine-tuning BILM on new domains (e.g., Python code) does not degrade its performance on previously learned domains (e.g., English text). In benchmarks, it often *improves* its base character prediction due to generalized low-level cortical sharing while protecting sequence semantics using Hippocampal sparse oneshot-binding.
2. **CPU-Native Efficiency:** BILM uses a completely sparse binary architecture, implemented via JIT-compiled Numba kernels. It does not require a GPU. It runs extremely fast on CPU architectures.
3. **Continuous Hebbian Learning:** Instead of backpropagation over a frozen dataset, BILM continuously updates its synaptic weights in real-time as it streams over text. It uses Acetylcholine (ACh) variance-based habituation to scale its learning rate dynamically depending on how "surprising" the input is.
4. **Episodic Long-Range Context:** The CA3 Hippocampus stores high-surprise cortical states. When a partial cue is recognized later, it auto-associatively retrieves the full pattern and injects it top-down, providing long-range context without a traditional KV-cache.

## Installation

You can install BILM via `pip` or by cloning the repository.

```bash
git clone https://github.com/your-org/BILM.git
cd BILM
pip install -e .
```

### Requirements
- Python 3.9+
- `numpy`
- `numba`

## Usage

### Training

You can train BILM on any raw text file. It learns sequentially and continuously.

```bash
python -m bilm.train --text data/corpus.txt --tokens 1000000
```
*Note: This will output periodic Bits-Per-Character (BPC) metrics and save a checkpoint `.npz` file.*

### Generation

Once trained (or even completely untrained, it learns as it generates if you want), you can prompt the model:

```bash
python -m bilm.generate --prompt "The quick brown fox" --max-chars 200 --temperature 1.0
```

### Benchmarks

BILM includes a suite of benchmarks demonstrating its capabilities.

```bash
# Run the Perplexity / BPC test on WikiText-2
python -m bilm.benchmark --test perplexity

# Run the Catastrophic Forgetting test
python -m bilm.benchmark --test forgetting

# Run the Efficiency test
python -m bilm.benchmark --test efficiency
```

## Architecture

BILM's architecture draws inspiration from mammalian neurobiology:
- **Sensory Codec (`bilm/codec.py`)**: Hashes incoming bytes into fixed SDRs (16,384 dimension, 64 active bits) to ensure 0.39% sparsity.
- **Hierarchical Cortex (`bilm/cortex.py`)**: A 3-layer predictive sequence memory with bottom-up pooling and top-down apical biasing.
- **Neuromodulator (`bilm/neuromod.py`)**: Tracks prediction error (surprise) to release Acetylcholine (ACh) and govern synaptic plasticity.
- **Hippocampus (`bilm/hippocampus.py`)**: A CA3 attractor network that binds sparse memory representations during high-ACh events for one-shot retrieval.
- **Synaptic Homeostasis (`bilm/homeostasis.py`)**: Prunes dead synapses and rescales permanence to prevent network saturation.

## License
MIT License
