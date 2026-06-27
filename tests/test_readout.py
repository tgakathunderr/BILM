import numpy as np

from bilm.codec import ByteCodec
from bilm.readout import LocalByteReadout


def test_local_readout_increases_target_probability():
    codec = ByteCodec()
    readout = LocalByteReadout()
    columns = codec.byte_sdrs[10]
    before = readout.predict(columns, codec)
    for _ in range(20):
        probs = readout.predict(columns, codec)
        readout.learn(columns, probs, target=65)
    after = readout.predict(columns, codec)
    assert np.isclose(after.sum(), 1.0)
    assert after[65] > before[65]
