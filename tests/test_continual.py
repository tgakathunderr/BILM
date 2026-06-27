from bilm.baselines import NGramByteLM
from bilm.continual import Domain, run_continual_experiment


def test_continual_report_tracks_acquisition_and_forgetting():
    domains = [
        Domain("a", b"abab" * 100, b"abab" * 20),
        Domain("b", b"xyxy" * 100, b"xyxy" * 20),
    ]
    report = run_continual_experiment(
        lambda: NGramByteLM(order=2), domains, warmup=2
    )
    assert set(report.initial_bpb) == {"a", "b"}
    assert len(report.stages) == 2
    assert report.stages[0].acquisition_bpb > 0.0
    assert set(report.stages[1].forgetting_bpb) == {"a", "b"}
