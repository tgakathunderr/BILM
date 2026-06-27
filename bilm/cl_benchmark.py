"""Phase 5: Continual learning benchmark with 5 byte-stream domains."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from bilm import BILM
from bilm.baselines import UnigramByteLM, NGramByteLM
from bilm.continual import Domain, ContinualModel, run_continual_experiment
from bilm.grammars import (
    generate_ab_pattern,
    generate_language_with_syntax,
    generate_nested_structure,
    generate_delayed_copy,
    generate_simple_grammar,
)


def generate_domain_english(length: int) -> bytes:
    """Domain 1: General English text (procedural, word-level structure)."""
    return generate_language_with_syntax(length, seed=42)


def generate_domain_code(length: int) -> bytes:
    """Domain 2: Source code (indentation-heavy, keyword-based)."""
    keywords = [b"def ", b"return ", b"if ", b"else:", b"for ", b"in ", b"class ", b"import "]
    operators = [b" == ", b" != ", b" = ", b" + ", b" - ", b" * ", b"()"]
    indent_levels = [b"    ", b"        ", b""]
    rng = np.random.default_rng(123)
    result = bytearray()
    while len(result) < length:
        indent = rng.choice(indent_levels)
        keyword = rng.choice(keywords)
        name = bytes(rng.integers(ord("a"), ord("z"), size=rng.integers(3, 8), dtype=np.uint8))
        if rng.random() < 0.4:
            op = rng.choice(operators)
            value = bytes(rng.integers(ord("0"), ord("9"), size=rng.integers(1, 4), dtype=np.uint8))
            line = indent + keyword + name + op + value + b"\n"
        else:
            line = indent + keyword + name + b"\n"
        result.extend(line)
    return bytes(result[:length])


def generate_domain_scientific(length: int) -> bytes:
    """Domain 3: Scientific prose (Latin terms, numbers, formulas)."""
    phrases = [
        b"the results indicate that ",
        b"as shown in Figure ",
        b"the experiment was conducted ",
        b"based on our analysis of ",
        b"the data suggests a strong ",
        b"correlation between the variables ",
        b"the hypothesis was supported ",
        b"in agreement with previous work ",
        b"the methodology employed ",
        b"statistical significance was observed ",
    ]
    rng = np.random.default_rng(456)
    result = bytearray()
    while len(result) < length:
        phrase = rng.choice(phrases)
        number = bytes(str(rng.integers(1, 1000)).encode())
        result.extend(phrase)
        result.extend(number)
        result.append(ord(" "))
    return bytes(result[:length])


def generate_domain_legal(length: int) -> bytes:
    """Domain 4: Legal prose (formal, repetitive clauses)."""
    clauses = [
        b"notwithstanding any provision to the contrary ",
        b"the party of the first part shall ",
        b"in accordance with the terms and conditions ",
        b"subject to the limitations set forth herein ",
        b"the indemnifying party shall defend ",
        b"any and all claims arising out of ",
        b"the obligations under this agreement ",
        b"force majeure events shall excuse ",
        b"the governing law of this contract ",
        b"disputes shall be resolved by arbitration ",
    ]
    rng = np.random.default_rng(789)
    result = bytearray()
    while len(result) < length:
        clause = rng.choice(clauses)
        section = b"Section " + bytes(str(rng.integers(1, 50)).encode()) + b". "
        result.extend(section)
        result.extend(clause)
    return bytes(result[:length])


def generate_domain_multilingual(length: int) -> bytes:
    """Domain 5: Multilingual UTF-8 (mixed scripts)."""
    scripts = {
        "latin": b"The quick brown fox jumps over the lazy dog. ",
        "accent": "caf\u00e9 r\u00e9sum\u00e9 naive r\u00e9sum\u00e9 ".encode("utf-8"),
        "cyrillic": "\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440 ".encode("utf-8"),
        "cjk": "\u4f60\u597d\u4e16\u754c\u4f60\u597d\u4e16\u754c ".encode("utf-8"),
        "arabic": "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645 ".encode("utf-8"),
    }
    rng = np.random.default_rng(101)
    script_names = list(scripts.keys())
    result = bytearray()
    while len(result) < length:
        name = rng.choice(script_names)
        result.extend(scripts[name])
    return bytes(result[:length])


DOMAIN_GENERATORS = {
    "english": generate_domain_english,
    "code": generate_domain_code,
    "scientific": generate_domain_scientific,
    "legal": generate_domain_legal,
    "multilingual": generate_domain_multilingual,
}

DOMAIN_ORDERS = [
    ["english", "code", "scientific", "legal", "multilingual"],
    ["code", "english", "multilingual", "scientific", "legal"],
    ["multilingual", "legal", "code", "english", "scientific"],
]


def run_continual_benchmark(
    domain_size: int = 5000,
    eval_size: int = 1000,
    orders: list[list[str]] | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run the full continual learning benchmark."""
    if orders is None:
        orders = DOMAIN_ORDERS

    all_results = {}

    for model_name in ["bilm", "bigram", "trigram"]:
        all_results[model_name] = {"orders": []}

        for order_idx, order in enumerate(orders):
            print(f"  {model_name} order {order_idx}: {' -> '.join(order)}")

            domains = []
            for name in order:
                gen = DOMAIN_GENERATORS[name]
                train = gen(domain_size)
                eval_data = gen(eval_size)
                domains.append(Domain(name=name, train=train, evaluation=eval_data))

            if model_name == "bilm":
                factory = BILM
            elif model_name == "bigram":
                factory = lambda: NGramByteLM(order=2)
            elif model_name == "trigram":
                factory = lambda: NGramByteLM(order=3)
            else:
                factory = UnigramByteLM

            t0 = time.time()
            report = run_continual_experiment(factory, domains, warmup=min(256, eval_size // 2))
            wall_time = time.time() - t0

            all_results[model_name]["orders"].append({
                "order": order,
                "initial_bpb": report.initial_bpb,
                "stages": [asdict(s) for s in report.stages],
                "wall_seconds": wall_time,
            })

    if output_dir:
        path = Path(output_dir) / "continual_benchmark.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    return all_results


def summarize_benchmark(results: dict) -> None:
    """Print a summary of continual learning benchmark results."""
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING BENCHMARK SUMMARY")
    print("=" * 70)

    for model_name, model_data in results.items():
        print(f"\n--- {model_name.upper()} ---")
        for order_data in model_data["orders"]:
            order = order_data["order"]
            stages = order_data["stages"]
            wall = order_data["wall_seconds"]

            avg_bpb = np.mean([
                np.mean(list(stage["bpb_by_domain"].values()))
                for stage in stages
            ])
            total_forgetting = np.mean([
                np.mean(list(stage["forgetting_bpb"].values()))
                for stage in stages
            ])

            print(f"  Order: {' -> '.join(order)}")
            print(f"    Avg BPB: {avg_bpb:.4f}  Total forgetting: {total_forgetting:.4f}  Time: {wall:.1f}s")
            for stage in stages:
                domain = stage["trained_domain"]
                acq = stage["acquisition_bpb"]
                print(f"      After {domain}: acq={acq:+.4f}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BILM 2 continual learning benchmark")
    parser.add_argument("--domain-size", type=int, default=5000)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    print("=== Phase 5: Continual Learning Benchmark ===")
    results = run_continual_benchmark(
        domain_size=args.domain_size,
        eval_size=args.eval_size,
        output_dir=args.output_dir,
    )
    summarize_benchmark(results)


if __name__ == "__main__":
    main()
