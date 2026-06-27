"""Phase 7: Challenger evaluation — compare BILM against baselines on continual learning."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from bilm import BILM
from bilm.baselines import UnigramByteLM, NGramByteLM
from bilm.continual import Domain, ContinualModel, run_continual_experiment
from bilm.cl_benchmark import (
    DOMAIN_GENERATORS,
    generate_domain_english,
    generate_domain_code,
    generate_domain_scientific,
    generate_domain_legal,
    generate_domain_multilingual,
)


@dataclass
class ChallengerResult:
    model_name: str
    domain_order: list[str]
    initial_bpb: dict[str, float]
    final_bpb: dict[str, float]
    acquisition: dict[str, float]
    forgetting: dict[str, float]
    avg_forgetting: float
    avg_bpb: float
    peak_ram_mb: float | None = None
    wall_seconds: float = 0.0


def run_single_domain_sequence(
    model_factory,
    domains: list[Domain],
    model_name: str,
    order: list[str],
) -> ChallengerResult:
    """Run a model through a domain sequence and collect metrics."""
    t0 = time.time()
    report = run_continual_experiment(model_factory, domains, warmup=256)
    wall = time.time() - t0

    final_bpb = report.stages[-1].bpb_by_domain if report.stages else {}
    acquisition = {s.trained_domain: s.acquisition_bpb for s in report.stages}
    forgetting = {}
    for s in report.stages:
        for domain, fb in s.forgetting_bpb.items():
            if domain not in forgetting or fb > forgetting[domain]:
                forgetting[domain] = fb

    avg_forgetting = float(np.mean(list(forgetting.values()))) if forgetting else 0.0
    all_bpb = list(final_bpb.values()) if final_bpb else [0.0]
    avg_bpb = float(np.mean(all_bpb))

    return ChallengerResult(
        model_name=model_name,
        domain_order=order,
        initial_bpb=report.initial_bpb,
        final_bpb=final_bpb,
        acquisition=acquisition,
        forgetting=forgetting,
        avg_forgetting=avg_forgetting,
        avg_bpb=avg_bpb,
        wall_seconds=wall,
    )


def run_challenger_evaluation(
    domain_size: int = 3000,
    eval_size: int = 500,
) -> dict:
    """Run the full challenger evaluation."""
    orders = [
        ["english", "code", "scientific", "legal", "multilingual"],
        ["code", "english", "multilingual", "scientific", "legal"],
    ]

    models = {
        "unigram": lambda: UnigramByteLM(),
        "bigram": lambda: NGramByteLM(order=2),
        "trigram": lambda: NGramByteLM(order=3),
        "bilm": BILM,
    }

    all_results = {}

    for model_name, factory in models.items():
        all_results[model_name] = []
        for order_idx, order in enumerate(orders):
            print(f"  {model_name} order {order_idx}: {' -> '.join(order)}")

            domains = []
            for name in order:
                gen = DOMAIN_GENERATORS[name]
                train = gen(domain_size)
                eval_data = gen(eval_size)
                domains.append(Domain(name=name, train=train, evaluation=eval_data))

            result = run_single_domain_sequence(factory, domains, model_name, order)
            all_results[model_name].append(asdict(result))

    return all_results


def check_challenger_criteria(results: dict) -> dict:
    """Check if BILM meets the challenger criteria."""
    criteria = {}

    bilm_results = results.get("bilm", [])
    bigram_results = results.get("bigram", [])

    if not bilm_results or not bigram_results:
        return {"error": "insufficient results"}

    for idx in range(min(len(bilm_results), len(bigram_results))):
        bilm = bilm_results[idx]
        bigram = bigram_results[idx]

        bilm_avg = bilm["avg_bpb"]
        bigram_avg = bigram["avg_bpb"]
        bilm_forget = bilm["avg_forgetting"]
        bigram_forget = bigram["avg_forgetting"]

        criteria[f"order_{idx}"] = {
            "bilm_avg_bpb": bilm_avg,
            "bigram_avg_bpb": bigram_avg,
            "bilm_avg_forgetting": bilm_forget,
            "bigram_avg_forgetting": bigram_forget,
            "bpb_gap_closure": (
                (bigram_avg - bilm_avg) / max(bigram_avg, 1e-9)
                if bigram_avg > 0 else 0.0
            ),
            "forgetting_improvement": (
                (bigram_forget - bilm_forget) / max(bigram_forget, 1e-9)
                if bigram_forget > 0 else 0.0
            ),
        }

    return criteria


def print_challenger_report(results: dict, criteria: dict) -> None:
    """Print a formatted challenger evaluation report."""
    print("\n" + "=" * 70)
    print("CHALLENGER EVALUATION REPORT")
    print("=" * 70)

    for model_name, model_results in results.items():
        print(f"\n--- {model_name.upper()} ---")
        for r in model_results:
            order = r["domain_order"]
            print(f"  Order: {' -> '.join(order)}")
            print(f"    Avg BPB: {r['avg_bpb']:.4f}  Avg Forgetting: {r['avg_forgetting']:.4f}  Time: {r['wall_seconds']:.1f}s")
            for domain, bpb in r["final_bpb"].items():
                acq = r["acquisition"].get(domain, 0.0)
                forget = r["forgetting"].get(domain, 0.0)
                print(f"      {domain:>14s}: BPB={bpb:.4f}  acq={acq:+.4f}  forget={forget:.4f}")

    print(f"\n--- CRITERIA ---")
    for key, vals in criteria.items():
        if key == "error":
            continue
        print(f"  {key}:")
        print(f"    BPB gap closure: {vals['bpb_gap_closure']:.1%}")
        print(f"    Forgetting improvement: {vals['forgetting_improvement']:.1%}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BILM 2 challenger evaluation")
    parser.add_argument("--domain-size", type=int, default=3000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    print("=== Phase 7: Challenger Evaluation ===")
    results = run_challenger_evaluation(
        domain_size=args.domain_size,
        eval_size=args.eval_size,
    )

    criteria = check_challenger_criteria(results)

    print_challenger_report(results, criteria)

    output = {
        "results": results,
        "criteria": criteria,
    }
    path = Path(args.output_dir) / "challenger_evaluation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
