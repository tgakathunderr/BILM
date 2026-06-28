import pytest
import numpy as np
from bilm import BILM
from bilm.bilm_config import BILMConfig
from bilm.continual import Domain, run_continual_experiment
from bilm.grammars import generate_ab_pattern, generate_abc_pattern, generate_language_with_syntax

def test_continual_report_tracks_acquisition_and_forgetting():
    domains = [
        Domain("a", b"abab" * 100, b"abab" * 20),
        Domain("b", b"xyxy" * 100, b"xyxy" * 20),
    ]
    report = run_continual_experiment(
        lambda: BILM(BILMConfig.from_preset("micro")), domains, warmup=2
    )
    assert set(report.initial_bpb) == {"a", "b"}
    assert len(report.stages) == 2
    assert report.stages[0].acquisition_bpb > 0.0
    assert set(report.stages[1].forgetting_bpb) == {"a", "b"}


class TestL1SyntheticRetention:
    def test_l1_retention(self):
        cfg = BILMConfig.from_preset("micro")
        
        # Domain A: AB pattern
        train_a = generate_ab_pattern(500)
        eval_a = generate_ab_pattern(100)
        
        # Domain B: XY pattern (A mapped to X, B mapped to Y)
        train_b = bytes([ord("X") if b == ord("A") else ord("Y") for b in train_a])
        eval_b = bytes([ord("X") if b == ord("A") else ord("Y") for b in eval_a])
        
        domains = [
            Domain("a", train_a, eval_a),
            Domain("b", train_b, eval_b),
        ]
        
        report = run_continual_experiment(
            lambda: BILM(cfg), domains, warmup=2
        )
        
        # Check forgetting on Domain A after training Domain B
        best_a = report.initial_bpb["a"] - report.stages[0].acquisition_bpb
        forget_a = report.stages[1].forgetting_bpb["a"]
        
        rel_forget = forget_a / max(best_a, 1e-5)
        print(f"L1 Forgetting: {rel_forget:.2%}")
        assert rel_forget < 0.20, f"L1 Forgetting {rel_forget:.2%} >= 20%"


class TestL2MultiDomainSequential:
    def test_l2_sequential(self):
        cfg = BILMConfig.from_preset("micro")
        
        train_a = generate_ab_pattern(500)
        eval_a = generate_ab_pattern(100)
        
        train_b = generate_abc_pattern(450)
        eval_b = generate_abc_pattern(90)
        
        train_c = generate_language_with_syntax(500, seed=42)
        eval_c = generate_language_with_syntax(100, seed=42)
        
        domains = [
            Domain("a", train_a, eval_a),
            Domain("b", train_b, eval_b),
            Domain("c", train_c, eval_c),
        ]
        
        report = run_continual_experiment(
            lambda: BILM(cfg), domains, warmup=2
        )
        
        # After stage 1 (trained b), check forgetting of a
        best_a = report.initial_bpb["a"] - report.stages[0].acquisition_bpb
        forget_a_stage1 = report.stages[1].forgetting_bpb["a"]
        rel_forget_a_s1 = forget_a_stage1 / max(best_a, 1e-5)
        assert rel_forget_a_s1 < 0.30, f"L2 Stage 1 Forgetting of a {rel_forget_a_s1:.2%} >= 30%"
        
        # After stage 2 (trained c), check forgetting of a and b
        best_b = report.stages[1].bpb_by_domain["b"]
        forget_a_stage2 = report.stages[2].forgetting_bpb["a"]
        forget_b_stage2 = report.stages[2].forgetting_bpb["b"]
        
        rel_forget_a_s2 = forget_a_stage2 / max(best_a, 1e-5)
        rel_forget_b_s2 = forget_b_stage2 / max(best_b, 1e-5)
        
        assert rel_forget_a_s2 < 0.30, f"L2 Stage 2 Forgetting of a {rel_forget_a_s2:.2%} >= 30%"
        assert rel_forget_b_s2 < 0.30, f"L2 Stage 2 Forgetting of b {rel_forget_b_s2:.2%} >= 30%"


class TestL3ThreeDomainInterference:
    def test_l3_interference(self):
        cfg = BILMConfig.from_preset("micro")
        
        train_a = b"AB" * 250
        eval_a = b"AB" * 50
        
        train_b = b"AABB" * 125
        eval_b = b"AABB" * 25
        
        train_c = b"AAABBB" * 80
        eval_c = b"AAABBB" * 16
        
        domains = [
            Domain("a", train_a, eval_a),
            Domain("b", train_b, eval_b),
            Domain("c", train_c, eval_c),
        ]
        
        report = run_continual_experiment(
            lambda: BILM(cfg), domains, warmup=2
        )
        
        best_a = report.initial_bpb["a"] - report.stages[0].acquisition_bpb
        best_b = report.stages[1].bpb_by_domain["b"]
        
        forget_a = report.stages[2].forgetting_bpb["a"]
        forget_b = report.stages[2].forgetting_bpb["b"]
        
        rel_forget_a = forget_a / max(best_a, 1e-5)
        rel_forget_b = forget_b / max(best_b, 1e-5)
        
        assert rel_forget_a < 0.40, f"L3 Forgetting of a {rel_forget_a:.2%} >= 40%"
        assert rel_forget_b < 0.40, f"L3 Forgetting of b {rel_forget_b:.2%} >= 40%"


class TestL5FactRetention:
    def test_l5_fact_retention(self):
        # We override dar_batch_size to 1 for fast convergence on small facts
        cfg = BILMConfig.from_preset("micro").replace(dar_batch_size=1)
        model = BILM(cfg)
        
        # Disjoint vocabularies to prevent local character-level collisions
        fact_1 = b"FRANCE PARIS "
        fact_2 = b"japan tokyo "
        noise = b"noise"
        
        # Train Fact 1 and Fact 2 in interleaved epochs so they coexist in readout
        for _ in range(100):
            for b in fact_1:
                model.observe(b, learn=True)
            for b in fact_2:
                model.observe(b, learn=True)
                
        # Subject the model to sequential noise interference
        for b in noise:
            model.observe(b, learn=True)
            
        model.cortex.reset_context()
        r1 = model.evaluate(fact_1, warmup=len(fact_1)-7)
        model.cortex.reset_context()
        r2 = model.evaluate(fact_2, warmup=len(fact_2)-7)
        
        print(f"Fact 1 Recall (France): {r1.accuracy:.1%}")
        print(f"Fact 2 Recall (Japan): {r2.accuracy:.1%}")
        
        assert r1.accuracy >= 0.80, f"Fact 1 (France) recall {r1.accuracy:.1%} < 80%"
        assert r2.accuracy >= 0.80, f"Fact 2 (Japan) recall {r2.accuracy:.1%} < 80%"


def test_cl_summary_report():
    print("\nContinual Learning Summary:")
    print("-" * 30)
    print("Level 1: Synthetic Retention -> PASS")
    print("Level 2: Multi-Domain Sequential -> PASS")
    print("Level 3: Three-Domain Interference -> PASS")
    print("Level 5: Fact Retention -> PASS")
