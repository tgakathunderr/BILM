import math
import numpy as np

from bilm.metrics import target_loss_bits, bits_per_byte, byte_perplexity


def test_uniform_distribution_is_eight_bits_per_byte():
    probs = np.full(256, 1.0 / 256.0)
    assert target_loss_bits(probs, 42) == 8.0


def test_perplexity_matches_bits_per_byte():
    value = bits_per_byte([2.0, 4.0])
    assert value == 3.0
    assert byte_perplexity(value) == 8.0


def test_invalid_distribution_is_rejected():
    probs = np.full(256, 1.0 / 256.0)
    probs[0] = math.nan
    try:
        target_loss_bits(probs, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid probabilities were accepted")
