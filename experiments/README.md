# BILM experiments

Experiment configurations, dataset manifests, and compact result summaries belong
here. Large corpora, checkpoints, traces, and raw generated artifacts remain ignored.

The continual benchmark accepts repeatable domain specifications:

```bash
python -m bilm.continual_benchmark \
  --model bilm \
  --domain english:data/english-train.bin:data/english-eval.bin \
  --domain code:data/code-train.bin:data/code-eval.bin
```

Every published result must record the exact command, Git commit, Python and package
versions, hardware, dataset hashes, domain order, and seed.
