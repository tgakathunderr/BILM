import sys
sys.path.append('.')
from bilm.hippocampus import Hippocampus
from bilm.bilm_config import BILMConfig
import numpy as np

def test_phase6_ca3():
    cfg = BILMConfig()
    h = Hippocampus(cfg)
    rng = np.random.default_rng(0)
    pats = [rng.choice(cfg.hippo_size, cfg.hippo_sparsity, replace=False) for _ in range(50)]
    for p in pats:
        h._hebbian_bind(p)

    ok = 0
    for p in pats:
        noisy = np.concatenate([
            p[:cfg.hippo_sparsity // 2],
            rng.choice(cfg.hippo_size, cfg.hippo_sparsity // 2, replace=False)
        ])
        ret = h.retrieve_ca3(noisy)
        if len(np.intersect1d(p, ret)) > cfg.hippo_sparsity * 0.5:
            ok += 1

    acc = ok / len(pats)
    print(f"CA3 accuracy: {acc:.1%}")
    assert acc > 0.80, f"{acc:.1%} < 80% gate"
