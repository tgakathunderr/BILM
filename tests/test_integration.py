"""Full pipeline: train -> save -> load -> train more -> evaluate -> generate."""
import tempfile
import os
from bilm import BILM
from bilm.bilm_config import BILMConfig
from bilm.grammars import generate_language_with_syntax


def test_full_pipeline():
    cfg = BILMConfig(
        sdr_size=4096, n_layers=2, layer_decays=(0.80, 0.99),
        dar_hidden_1=128, dar_hidden_2=64,
        hippo_size=512, srs_dim=64, dap_dim=128
    )
    model = BILM(cfg)
    data  = generate_language_with_syntax(5000)

    for b in data[:2500]:
        model.observe(b, learn=True)
    bpb_mid = model.evaluate(data[2000:2500]).bits_per_byte

    with tempfile.TemporaryDirectory() as d:
        ckpt     = os.path.join(d, "test.npz")
        cfg_path = os.path.join(d, "test.json")
        model.save(ckpt)
        cfg.save(cfg_path)

        m2 = BILM(BILMConfig.load(cfg_path))
        m2.load(ckpt)

        for b in data[2500:]:
            m2.observe(b, learn=True)
        bpb_end = m2.evaluate(data[4000:]).bits_per_byte

        text = m2.generate("the ", max_bytes=50, temperature=0.8)
        text.encode("utf-8")   # raises if output is invalid bytes

    assert bpb_end < bpb_mid, f"BPB did not improve: {bpb_mid:.4f} -> {bpb_end:.4f}"
    assert bpb_end < 8.0,     f"BPB {bpb_end:.4f} > 8.0 — architecture broken"
    print(f"PASSED: {bpb_mid:.4f} -> {bpb_end:.4f}")


def test_arbitrary_config():
    """Completely custom config works end-to-end — no hardcoded sizes."""
    cfg = BILMConfig(
        sdr_size=2048, n_layers=1, layer_decays=(0.95,),
        dar_hidden_1=64, dar_hidden_2=32, hippo_size=256,
        srs_dim=32, dap_dim=64
    )
    m = BILM(cfg)
    for b in b"hello world " * 200:
        m.observe(b, learn=True)
    r = m.evaluate(b"hello world " * 50)
    assert r.bits_per_byte < 8.0
    print(f"PASSED: {cfg.describe()}")


def test_evaluate_is_non_mutating():
    """evaluate() must not change model state."""
    m = BILM()
    data = b"the quick brown fox " * 100
    for b in data:
        m.observe(b, learn=True)
    bpb1 = m.evaluate(data[:500]).bits_per_byte
    bpb2 = m.evaluate(data[:500]).bits_per_byte
    assert abs(bpb1 - bpb2) < 1e-6, f"evaluate() is mutating: {bpb1} != {bpb2}"
    print("PASSED: evaluate() non-mutating")
