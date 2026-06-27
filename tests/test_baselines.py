import math

from bilm.baselines import NGramByteLM


def test_bigram_learns_repeated_sequence():
    model = NGramByteLM(order=2)
    data = b"abababababababab" * 20
    before = model.evaluate(data, warmup=2).bits_per_byte
    model.train(data)
    after = model.evaluate(data, warmup=2).bits_per_byte
    assert after < before
    assert math.isfinite(after)


def test_evaluation_does_not_change_counts():
    model = NGramByteLM(order=3)
    model.train(b"hello world")
    before = {key: value.copy() for key, value in model.counts.items()}
    model.evaluate(b"hello", warmup=1)
    assert before.keys() == model.counts.keys()
    for key in before:
        assert (before[key] == model.counts[key]).all()
