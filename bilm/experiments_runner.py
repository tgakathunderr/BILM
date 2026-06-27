"""Learning curve experiments and ablation framework for BILM."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from bilm import BILM
from bilm.baselines import UnigramByteLM, NGramByteLM
from bilm.grammars import GRAMMAR_REGISTRY
from bilm.metrics import bits_per_byte, byte_perplexity


@dataclass(frozen=True)
class LearningPoint:
    tokens: int
    bpb: float
    perplexity: float
    accuracy: float
    wall_seconds: float


@dataclass(frozen=True)
class LearningCurve:
    model_name: str
    dataset: str
    seed: int
    points: list[LearningPoint]

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset": self.dataset,
            "seed": self.seed,
            "points": [asdict(p) for p in self.points],
        }


@dataclass(frozen=True)
class AblationResult:
    mechanism: str
    enabled: bool
    bpb: float
    accuracy: float
    wall_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def measure_learning_curve(
    model_fn,
    train_data: bytes,
    eval_data: bytes,
    checkpoints: list[int],
    *,
    warmup: int = 256,
    label: str = "model",
    dataset: str = "unknown",
    seed: int = 0,
) -> LearningCurve:
    """Train a model and measure BPB at specified token checkpoints."""
    model = model_fn()
    points: list[LearningPoint] = []
    t0 = time.time()
    tokens_done = 0

    for target in checkpoints:
        while tokens_done < target:
            b = train_data[tokens_done % len(train_data)]
            model.observe(int(b), learn=True)
            tokens_done += 1

        elapsed = time.time() - t0
        report = model.evaluate(
            eval_data,
            warmup=min(warmup, max(0, len(eval_data) - 1)),
            reset_context=True,
        )
        points.append(LearningPoint(
            tokens=target,
            bpb=report.bits_per_byte,
            perplexity=report.perplexity,
            accuracy=report.accuracy,
            wall_seconds=elapsed,
        ))

    return LearningCurve(
        model_name=label,
        dataset=dataset,
        seed=seed,
        points=points,
    )


def measure_baselines(
    train_data: bytes,
    eval_data: bytes,
    *,
    warmup: int = 0,
) -> dict[str, LearningCurve]:
    """Measure all baseline models on the same data."""
    baselines = {
        "unigram": lambda: UnigramByteLM(),
        "bigram": lambda: NGramByteLM(order=2),
        "trigram": lambda: NGramByteLM(order=3),
    }
    results = {}
    for name, factory in baselines.items():
        model = factory()
        t0 = time.time()
        model.train(train_data)
        elapsed = time.time() - t0
        report = model.evaluate(eval_data, warmup=warmup)
        results[name] = LearningCurve(
            model_name=name,
            dataset="synthetic",
            seed=0,
            points=[LearningPoint(
                tokens=len(train_data),
                bpb=report.bits_per_byte,
                perplexity=report.perplexity,
                accuracy=report.accuracy,
                wall_seconds=elapsed,
            )],
        )
    return results


def run_grammar_experiment(
    grammar_name: str,
    train_size: int = 10_000,
    eval_size: int = 2_000,
    bilm_seeds: list[int] | None = None,
) -> dict:
    """Run BILM + baselines on a synthetic grammar."""
    if grammar_name not in GRAMMAR_REGISTRY:
        raise ValueError(f"Unknown grammar: {grammar_name}")

    generator = GRAMMAR_REGISTRY[grammar_name]
    train_data = generator(train_size + eval_size)
    train = train_data[:train_size]
    eval_data = train_data[train_size:train_size + eval_size]

    baselines = measure_baselines(train, eval_data)

    if bilm_seeds is None:
        bilm_seeds = [42]

    bilm_curves = []
    for seed in bilm_seeds:
        def factory(seed=seed):
            model = BILM()
            return model
        curve = measure_learning_curve(
            factory, train, eval_data,
            checkpoints=[1000, 5000, min(train_size, 10_000)],
            label="bilm",
            dataset=grammar_name,
            seed=seed,
        )
        bilm_curves.append(curve)

    return {
        "grammar": grammar_name,
        "train_size": train_size,
        "eval_size": eval_size,
        "baselines": {k: v.to_dict() for k, v in baselines.items()},
        "bilm": [c.to_dict() for c in bilm_curves],
    }


def run_all_grammar_experiments(
    output_path: str | None = None,
    train_size: int = 5_000,
    seeds: list[int] | None = None,
) -> dict:
    """Run experiments on all registered grammars."""
    if seeds is None:
        seeds = [42]

    all_results = {}
    for grammar_name in GRAMMAR_REGISTRY:
        print(f"  Running {grammar_name} ...")
        result = run_grammar_experiment(
            grammar_name,
            train_size=train_size,
            bilm_seeds=seeds,
        )
        all_results[grammar_name] = result

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    return all_results
