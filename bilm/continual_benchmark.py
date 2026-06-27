from __future__ import annotations

import argparse
from pathlib import Path

from bilm import BILM
from bilm.baselines import UnigramByteLM, NGramByteLM
from bilm.continual import Domain, run_continual_experiment


def _domain(value: str) -> Domain:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("domain must be NAME:TRAIN_PATH:EVAL_PATH")
    name, train_path, eval_path = parts
    return Domain(
        name=name,
        train=Path(train_path).read_bytes(),
        evaluation=Path(eval_path).read_bytes(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BILM continual-learning benchmark")
    parser.add_argument("--domain", action="append", type=_domain, required=True)
    parser.add_argument("--model", choices=["bilm", "unigram", "bigram", "trigram"], default="bilm")
    parser.add_argument("--warmup", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    factories = {
        "bilm": BILM,
        "unigram": UnigramByteLM,
        "bigram": lambda: NGramByteLM(order=2),
        "trigram": lambda: NGramByteLM(order=3),
    }
    report = run_continual_experiment(
        factories[args.model], args.domain, warmup=args.warmup
    )
    rendered = report.to_json()
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
