from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Protocol
import json

from bilm.results import EvaluationReport


class ContinualModel(Protocol):
    def observe(self, byte: int, learn: bool = True): ...
    def evaluate(self, data: bytes, warmup: int = 0) -> EvaluationReport: ...


@dataclass(frozen=True)
class Domain:
    name: str
    train: bytes
    evaluation: bytes


@dataclass(frozen=True)
class DomainStage:
    trained_domain: str
    bpb_by_domain: dict[str, float]
    acquisition_bpb: float
    forgetting_bpb: dict[str, float]


@dataclass(frozen=True)
class ContinualReport:
    initial_bpb: dict[str, float]
    stages: list[DomainStage]

    def to_dict(self) -> dict:
        return {
            "initial_bpb": self.initial_bpb,
            "stages": [asdict(stage) for stage in self.stages],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def run_continual_experiment(
    model_factory: Callable[[], ContinualModel],
    domains: list[Domain],
    *,
    warmup: int = 256,
) -> ContinualReport:
    if not domains:
        raise ValueError("at least one domain is required")
    names = [domain.name for domain in domains]
    if len(names) != len(set(names)):
        raise ValueError("domain names must be unique")

    model = model_factory()

    def evaluate_all() -> dict[str, float]:
        return {
            domain.name: model.evaluate(
                domain.evaluation,
                warmup=min(warmup, max(0, len(domain.evaluation) - 1)),
            ).bits_per_byte
            for domain in domains
        }

    initial = evaluate_all()
    best = dict(initial)
    stages: list[DomainStage] = []

    for domain in domains:
        before = evaluate_all()[domain.name]
        for byte in domain.train:
            model.observe(byte, learn=True)
        current = evaluate_all()
        acquisition = before - current[domain.name]
        forgetting = {
            name: max(0.0, value - best[name])
            for name, value in current.items()
        }
        for name, value in current.items():
            best[name] = min(best[name], value)
        stages.append(
            DomainStage(
                trained_domain=domain.name,
                bpb_by_domain=current,
                acquisition_bpb=float(acquisition),
                forgetting_bpb=forgetting,
            )
        )

    return ContinualReport(initial_bpb=initial, stages=stages)
