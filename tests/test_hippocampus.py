import numpy as np

from bilm.hippocampus import Hippocampus


def test_hebbian_binding_writes_to_weight_matrix():
    hippo = Hippocampus()
    active = np.array([1, 3, 7], dtype=np.int32)
    hippo._hebbian_bind(active)
    assert hippo.get_stats()["active_weights"] == 6
    assert np.all(hippo.counts[active] == 2)


def test_sparse_memory_completes_a_partial_cue():
    hippo = Hippocampus()
    pattern = np.array([1, 3, 7, 11], dtype=np.int32)
    hippo._hebbian_bind(pattern)
    # Exercise recurrent dynamics directly with an equivalent cortical hash cue.
    energy = np.zeros(8192, dtype=np.float32)
    source = pattern[0]
    count = int(hippo.counts[source])
    np.add.at(energy, hippo.targets[source, :count], hippo.weights[source, :count])
    assert set(pattern[1:]).issubset(set(np.flatnonzero(energy)))


def test_internal_replay_reactivates_bound_patterns():
    hippo = Hippocampus()
    hippo._hebbian_bind(np.array([1, 3, 7, 11], dtype=np.int32))
    patterns = hippo.replay_patterns(2)
    assert patterns
    assert all(pattern.size > 1 for pattern in patterns)
