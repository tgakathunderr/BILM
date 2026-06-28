# BILM vs Transformer Comparison Report

## Claim 1: BILM forgets measurably less
| Model | Forgetting (%) | 95% CI |
|-------|---------------|--------|
| BILM  | 0.0 ± 0.0 | - |
| Transformer | 15563.4 ± 2287.3 | - |
| LSTM | 19200.6 ± 1428.4 | - |
| MinimalSSM | 5.3 ± 0.1 | - |

**Verdict: INCONCLUSIVE** (p = 0.0659)

## Claim 2: BILM adapts faster to new domains
| Model | Adaptation tokens |
|-------|-------------------|
| BILM  | 575.0 ± 0.0 |
| Transformer | 5.0 ± 0.0 |
| LSTM | 4.0 ± 1.4 |
| MinimalSSM | 9.5 ± 2.1 |

**Verdict: SUPPORTED**

## Claim 3: BILM BPB within 30% of matched Transformer
| Model | BPB at 1000 tokens |
|-------|-------------------|
| TinyTransformer | 4.457 |
| BILM | 8.377 |
| Gap | 87.9% |

**Verdict: NOT SUPPORTED** (within 30% threshold)

## Peak RAM footprint (during training)
| Model | Peak RAM Delta (MB) |
|-------|-------------------|
| BILM  | 74.50 |
| Transformer | 42.18 |
| LSTM | 153.13 |
| MinimalSSM | 101.26 |
