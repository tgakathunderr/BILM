"""Tests for synthetic grammars and experiment infrastructure."""
from __future__ import annotations

from bilm.grammars import (
    GRAMMAR_REGISTRY,
    generate_ab_pattern,
    generate_abc_pattern,
    generate_simple_grammar,
    generate_nested_structure,
    generate_delayed_copy,
    generate_language_with_syntax,
)
from bilm.baselines import NGramByteLM, UnigramByteLM


class TestGrammarGenerators:
    def test_ab_pattern(self):
        data = generate_ab_pattern(100)
        assert len(data) == 100
        for i, b in enumerate(data):
            expected = ord("A") if i % 2 == 0 else ord("B")
            assert b == expected

    def test_abc_pattern(self):
        data = generate_abc_pattern(90)
        assert len(data) == 90
        for i, b in enumerate(data):
            expected = ord("A") + (i % 3)
            assert b == expected

    def test_simple_grammar_balanced(self):
        data = generate_simple_grammar(1000, seed=42)
        assert data.count(ord("a")) == data.count(ord("b"))

    def test_delayed_copy(self):
        data = generate_delayed_copy(100, delay=10, seed=42)
        for i in range(10, len(data)):
            assert data[i] == data[i - 10]

    def test_language_syntax(self):
        data = generate_language_with_syntax(500, seed=42)
        assert len(data) == 500
        assert b"the" in data

    def test_all_generators_produce_output(self):
        for name, gen in GRAMMAR_REGISTRY.items():
            if "delayed" in name:
                data = gen(100)
            else:
                data = gen(100)
            assert len(data) > 0, f"{name} produced empty output"
            assert len(data) <= 100, f"{name} produced {len(data)} bytes, expected <= 100"

    def test_baselines_learn_grammar(self):
        for name, gen in GRAMMAR_REGISTRY.items():
            data = gen(500)
            model = NGramByteLM(order=3)
            model.train(data)
            report = model.evaluate(data[:200], warmup=0)
            assert report.bits_per_byte < 8.0, f"Bigram didn't learn {name}"
