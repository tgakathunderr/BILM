"""Main experiment runner for BILM 2 — Phase 2 and Phase 3 experiments."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from bilm.grammars import GRAMMAR_REGISTRY


def run_grammar_experiments(output_dir: str, train_size: int, seeds: list[int]) -> None:
    """Run all grammar experiments."""
    from bilm.experiments_runner import run_all_grammar_experiments

    print("=== Phase 2: Grammar Experiments ===")
    output = Path(output_dir) / "grammar_results.json"
    results = run_all_grammar_experiments(
        output_path=str(output),
        train_size=train_size,
        seeds=seeds,
    )

    print("\n--- Summary ---")
    for grammar, result in results.items():
        baselines = result["baselines"]
        bilm = result["bilm"]
        print(f"\n  {grammar}:")
        for name, curve in baselines.items():
            pts = curve["points"]
            if pts:
                print(f"    {name:>8s}: BPB={pts[0]['bpb']:.4f}  acc={pts[0]['accuracy']:.4f}")
        if bilm:
            final = bilm[0]["points"][-1] if bilm[0]["points"] else None
            if final:
                print(f"    {'bilm':>8s}: BPB={final['bpb']:.4f}  acc={final['accuracy']:.4f}")


def run_ablation_experiment(output_dir: str, train_size: int) -> None:
    """Run ablation experiment."""
    from bilm.ablation import run_ablation
    from bilm.grammars import generate_language_with_syntax

    print("\n=== Phase 3: Ablation Experiment ===")
    data = generate_language_with_syntax(train_size + 2000)
    train = data[:train_size]
    eval_data = data[train_size:train_size + 2000]

    results = run_ablation(train, eval_data, warmup=256)

    output = Path(output_dir) / "ablation_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n--- Ablation Results ---")
    print(f"  {'Mechanism':<20s} {'BPB':>8s} {'Accuracy':>10s} {'Time(s)':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10}")
    for r in results:
        print(
            f"  {r['mechanism']:<20s} {r['bpb']:>8.4f} "
            f"{r['accuracy']:>10.4f} {r['wall_seconds']:>10.1f}"
        )


def run_baselines_benchmark(output_dir: str, train_size: int) -> None:
    """Run baselines on WikiText-2 if available, else synthetic."""
    from bilm.baselines import UnigramByteLM, NGramByteLM
    from bilm.metrics import bits_per_byte

    print("\n=== Baseline Benchmark ===")

    data_path = Path(output_dir).parent / "data" / "wikitext2_train.txt"
    eval_path = Path(output_dir).parent / "data" / "wikitext2_test.txt"

    if data_path.exists() and eval_path.exists():
        train_data = data_path.read_bytes()[:train_size]
        eval_data = eval_path.read_bytes()[:50_000]
        print(f"  Using WikiText-2: train={len(train_data):,} eval={len(eval_data):,}")
    else:
        from bilm.grammars import generate_language_with_syntax
        combined = generate_language_with_syntax(train_size + 10_000)
        train_data = combined[:train_size]
        eval_data = combined[train_size:]
        print(f"  Using synthetic language: train={len(train_data):,} eval={len(eval_data):,}")

    baselines = {
        "unigram": UnigramByteLM(),
        "bigram": NGramByteLM(order=2),
        "trigram": NGramByteLM(order=3),
    }

    results = {}
    for name, model in baselines.items():
        t0 = time.time()
        model.train(train_data)
        elapsed = time.time() - t0
        report = model.evaluate(eval_data, warmup=0)
        results[name] = {
            "bpb": report.bits_per_byte,
            "accuracy": report.accuracy,
            "perplexity": report.perplexity,
            "wall_seconds": elapsed,
        }
        print(f"  {name:>8s}: BPB={report.bits_per_byte:.4f}  acc={report.accuracy:.4f}  time={elapsed:.2f}s")

    output = Path(output_dir) / "baselines.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="BILM 2 experiment runner")
    parser.add_argument("--experiment", choices=["grammar", "ablation", "baselines", "all"], default="all")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    start = time.time()

    if args.experiment in ("baselines", "all"):
        run_baselines_benchmark(args.output_dir, args.train_size)

    if args.experiment in ("grammar", "all"):
        run_grammar_experiments(args.output_dir, args.train_size, args.seeds)

    if args.experiment in ("ablation", "all"):
        run_ablation_experiment(args.output_dir, args.train_size)

    elapsed = time.time() - start
    print(f"\n=== All experiments complete in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
