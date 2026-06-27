import numba
import numpy as np
from bilm.bilm_config import BILMConfig


@numba.njit
def srs_step_jit(
    active_cols: np.ndarray,
    W_k: np.ndarray,
    W_v: np.ndarray,
    W_q: np.ndarray,
    W_proj: np.ndarray,
    decay: np.ndarray,
    state: np.ndarray,
    sdr_sparsity: int,
) -> np.ndarray:
    if active_cols.size == 0:
        return np.zeros(256, dtype=np.float32)
    n = min(active_cols.size, sdr_sparsity)
    x = np.ones(n, dtype=np.float32)
    k = x @ W_k[:n]
    v = x @ W_v[:n]
    q = x @ W_q[:n]
    d_size = decay.size
    for i in range(d_size):
        d_val = decay[i]
        for j in range(d_size):
            state[i, j] = d_val * state[i, j] + (1.0 - d_val) * (k[i] * v[j])
            
    return (state @ q) @ W_proj


class SparseRecurrentState:
    def __init__(self, cfg: BILMConfig) -> None:
        self.cfg = cfg
        D, K = cfg.srs_dim, cfg.sdr_sparsity
        rng  = np.random.default_rng(77)
        self.W_k    = rng.normal(0, D**-0.5, (K, D)).astype(np.float32)
        self.W_v    = rng.normal(0, D**-0.5, (K, D)).astype(np.float32)
        self.W_q    = rng.normal(0, D**-0.5, (K, D)).astype(np.float32)
        self.W_proj = rng.normal(0, D**-0.5, (D, 256)).astype(np.float32)
        self.decay  = np.full(D, cfg.srs_decay, np.float32)
        self.state  = np.zeros((D, D), np.float32)

    def step(self, active_cols: np.ndarray) -> np.ndarray:
        return srs_step_jit(
            active_cols,
            self.W_k,
            self.W_v,
            self.W_q,
            self.W_proj,
            self.decay,
            self.state,
            self.cfg.sdr_sparsity,
        )

    def reset(self) -> None:
        self.state.fill(0.0)
