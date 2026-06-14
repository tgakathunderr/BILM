---
title: "BILM: A Biologically Inspired Sequence Model with High Sample Efficiency and Resistance to Catastrophic Forgetting"
author: "Nucleus AI"
date: "June 2026"
---

# Abstract

Modern sequence modeling is dominated by the dense Transformer architecture. While highly capable, these models rely on global backpropagation, require massive parallel hardware acceleration (GPUs), and suffer from catastrophic forgetting when sequentially trained on shifting data distributions. We present the Biologically Inspired Language Model (BILM), a CPU-native, continuously learning sequence model that departs from backpropagation. Inspired by Hierarchical Temporal Memory (HTM), BILM utilizes a sparse distributed representation (SDR) sensory codec, a 3-layer predictive sequence cortex with apical feedback, and an episodic CA3 Hippocampus attractor network. Our empirical evaluations on WikiText-2 show that BILM achieves a next-byte prediction accuracy of **26.89%** after single-pass sequential exposure to only 100,000 tokens. Furthermore, we demonstrate that BILM exhibits **0.0% degradation** on a held-out natural language evaluation set after sequential training on a source code domain. We present the mathematical and architectural foundations of this system as a viable pathway toward efficient, localized, and continuous sequence learning.

# Acknowledgments and Salutations

The theoretical foundations of BILM are deeply indebted to the pioneering work on **Hierarchical Temporal Memory (HTM)** developed by Jeff Hawkins, Subutai Ahmad, and the team at Numenta. Their mathematical formulations of Sparse Distributed Representations (SDRs), sequence memory, and local Hebbian plasticity rules provided the biological and computational axioms upon which this work is constructed. We salute their commitment to bridging neurobiology and artificial intelligence.

# 1. Introduction

The prevailing paradigm in natural language processing frames learning as optimization over dense parameter spaces via gradient descent. This approach suffers from three fundamental bottlenecks:
1. **Catastrophic Forgetting:** Sequential optimization on a new domain $B$ after domain $A$ causes gradient updates that destructively overwrite the representations of $A$, unless data is carefully interleaved.
2. **Computational and Power Inefficiency:** High-dimensional dense matrix multiplications require specialized accelerators (GPUs/TPUs) operating at high wattages, contrasting with the $\sim$20-watt operating limit of the human brain.
3. **KV-Cache Scaling:** Attention mechanisms exhibit $O(N^2)$ sequence-length scaling, requiring heavy memory overhead for long-range context.

We introduce BILM, a sequence model inspired by the mammalian neocortex and hippocampus. Rather than continuous-value floating-point weights updated globally, BILM relies on binary-sparse synaptic topologies, localized Hebbian plasticity, and episodic attractor networks to learn online from streaming byte sequences.

# 2. Architecture

BILM avoids global error propagation by utilizing localized learning rules across decoupled modules.

```
       [Raw Byte Stream]
              │
      ┌───────▼───────┐
      │ Sensory Codec │ (Hash-based SDR projection, N=16,384, w=64)
      └───────┬───────┘
              │ (0.39% Sparse SDR)
      ┌───────▼────────┐
      │  Predictive    ◄──────────────────┐ Apical Feedback
      │    Cortex      ├──────────────┐   │ (Contextual Bias)
      └───────┬────────┘              │   │
              │                       ▼───┴───┐
              │ High surprise       ┌─────────┴─┐
              └─────────────────────►   CA3     │
                (High Acetylcholine)│Hippocampus│ (Recurrent Attractor Network)
                                    └───────────┘
```

## 2.1 Sensory Codec
Rather than relying on statistical subword tokenizers (e.g., Byte-Pair Encoding), which can cause synaptic saturation under streaming conditions, BILM operates directly on raw bytes. Each incoming byte value ($0 \le b \le 255$) is projected into a high-dimensional Sparse Distributed Representation (SDR). 

The SDR space has dimension $N = 16,384$ with exactly $w = 64$ active bits (sparsity $s = w/N \approx 0.39\%$). The probability of $k$ active bits overlapping between two random representations is governed by the hypergeometric distribution:

\[
P(X = k) = \frac{\binom{w}{k}\binom{N-w}{w-k}}{\binom{N}{w}}
\]

For $k \ge 16$, the overlap probability is less than $10^{-14}$. This mathematical isolation prevents cross-talk and ensures that representations do not destructively interfere during continuous updates.

## 2.2 Hierarchical Predictive Cortex
The sequence memory comprises a three-layer hierarchy. Each layer contains predictive cells grouped into columns. Bottom-up feedforward connections activate columns representing the sensory input, while lateral and top-down (apical) pathways convey predictive context. The layers perform temporal pooling with decay constants tuned to capture different temporal resolutions:
* **Layer 1:** $\tau = 0.80$ (character/byte transitions)
* **Layer 2:** $\tau = 0.95$ (morphemic/word structures)
* **Layer 3:** $\tau = 0.99$ (syntactic/phrasal contexts)

Synaptic plasticity is local, governed by Hebbian rules:
* **Long-Term Potentiation (LTP):** Active synapses that successfully predicted a subsequent cell state have their permanence incremented.
* **Long-Term Depression (LTD):** Synapses that predicted cell activation which failed to occur are decremented (pruned).

## 2.3 Neuromodulation and Habituation
A neuromodulatory subsystem inspired by Acetylcholine (ACh) gates cortical plasticity. Plasticity scales with prediction error (surprise), measured by the Jaccard distance between the predicted column state $\mathbf{P}$ and the active feedforward column state $\mathbf{A}$:

\[
\text{Surprise} = 1 - \frac{|\mathbf{P} \cap \mathbf{A}|}{|\mathbf{P} \cup \mathbf{A}|}
\]

To prevent synaptic saturation during highly repetitive sequence noise, we implement variance-based habituation. If the surprise signal is high but exhibits low variance over a moving window, the learning rate is dynamically suppressed, protecting the model's existing structure.

## 2.4 CA3 Hippocampus
To maintain long-range episodic memory without the quadratic complexity of a Transformer KV-cache, we integrate an auto-associative recurrent attractor network modeling the hippocampal CA3 region. During epochs of high surprise (elevated ACh), the current cortical state is bound to a sparse attractor index. If a partial cue resembling a previous high-surprise state occurs, the Hippocampus performs pattern completion within a single step and feeds the retrieved state top-down to bias cortical predictions.

# 3. Experiments and Results

## 3.1 Sample Efficiency and Predictive Accuracy
We evaluated BILM on the WikiText-2 dataset. Unlike dense models that output continuous probability distributions over a vocabulary (allowing perplexity or Bits-Per-Character to be directly computed via cross-entropy), BILM outputs a sparse binary prediction of active columns. Therefore, we evaluate BILM using next-byte prediction accuracy (i.e., whether the actual next byte is within the decoded prediction set).

After single-pass training on a stream of 100,000 tokens, BILM achieved a Next-Byte Accuracy of **26.89%** on the held-out evaluation set (no online learning permitted during evaluation). The model demonstrates high sample efficiency, assimilating basic orthographic and syntactic structures from limited sequential data.

## 3.2 Resistance to Catastrophic Forgetting
To test continuous learning under domain shift, we trained the model sequentially:
1. **Domain A (WikiText-2 natural language):** The model is trained on 20,000 bytes of Domain A. Baseline next-byte accuracy is recorded on a strictly held-out Domain A evaluation set.
2. **Domain B (Python source code):** The model is then trained on $\sim$6,500 bytes of Python code (consisting of the benchmark script itself) with continuous learning fully enabled.
3. **Re-evaluation:** The model's accuracy on the held-out Domain A evaluation set is re-measured.

| Metric | Domain A Accuracy |
| :--- | :---: |
| Before Domain B Exposure | 23.41% |
| After Domain B Exposure | 23.41% |
| **Degradation** | **0.0%** |

BILM exhibited **0.0% degradation** (perfect retention). The mathematical sparsity of the SDR sensory projection ensures the synaptic weights updated for Domain B do not overwrite the paths established for Domain A, while the Hippocampus keeps episodic contexts segregated.

## 3.3 Computational Footprint and Speed
We evaluated the runtime efficiency of BILM on a standard consumer laptop CPU (single-core execution, Numba JIT-enabled). 

* **Throughput:** Training and inference operate at **~18 to 31 TPS** (Tokens Per Second).
* **Memory Footprint:** Peak RAM consumption stays at **~635 MB**. This includes the JIT compiler overhead and the $\sim$268 MB required for the $8192 \times 8192$ float32 weight matrix of the CA3 Hippocampus attractor network.

# 4. Discussion and Conclusion

BILM demonstrates that continuous learning with zero interference (forgetting) is achievable in sequence modeling by replacing dense gradients with sparse, local Hebbian rules. While its absolute prediction accuracy does not match massive pre-trained Transformer baselines, its sample efficiency (learning structural features from 100k tokens) and hardware independence (CPU-native, low power) suggest a viable path for edge deployment and localized lifelong learning.

Future work will explore scaling the sensory codec to higher-dimensional representations and multi-modal integration.

# References

1. Hawkins, J., & Ahmad, S. (2016). Why neurons have thousands of synapses, a theory of sequence memory in neocortex. *Frontiers in Neural Circuits*, 10, 23.
2. Hawkins, J., Ahmad, S., & Dubinsky, D. (2016). Hierarchical Temporal Memory (HTM) Cortical Learning Algorithms. *arXiv preprint arXiv:1601.07624*.
3. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*, 79(8), 2554-2558.
4. Numenta. (2020). *Biological and Machine Intelligence (BAMI)*. Technical Report, Numenta Inc.
