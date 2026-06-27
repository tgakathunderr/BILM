import numpy as np

from bilm.codec import ByteCodec


def test_distribution_is_normalized_and_argmax_compatible():
    codec = ByteCodec()
    columns = codec.byte_sdrs[65]
    probs = codec.decode_to_distribution(columns)
    assert probs.shape == (256,)
    assert np.isclose(probs.sum(), 1.0)
    assert int(np.argmax(probs)) == codec.decode_argmax(columns)


def test_empty_prediction_is_uniform():
    codec = ByteCodec()
    probs = codec.decode_to_distribution(np.array([], dtype=np.int64))
    assert np.allclose(probs, 1.0 / 256.0)


def test_untracked_encoding_does_not_mutate_frequency_state():
    codec = ByteCodec()
    codec.encode(65, track=False)
    assert codec.total_seen == 0
    assert codec.frequencies[65] == 0
