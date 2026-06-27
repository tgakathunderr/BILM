"""
Deep Associative Readout (DAR) — drop-in replacement for LocalByteReadout.

Architecture:
  L1: active SDR columns -> sum of embedding rows -> h1  (dar_hidden_1,)
  L2: h1 -> Linear -> ReLU -> h2                        (dar_hidden_2,)
  L3: h2 -> Linear -> logits                            (256,)

Learning:
  L1, L2: Online local delta rule, one token at a time
  L3:     Mini-batch Adam with decaying lr (CL-safe)

API is identical to LocalByteReadout: predict() and learn().
"""
import numba
import numpy as np
from bilm.bilm_config import BILMConfig

_relu      = lambda x: np.maximum(0.0, x)
_relu_grad = lambda x: (x > 0.0).astype(np.float32)


@numba.njit
def dar_fwd_jit(
    cols: np.ndarray,
    W_embed: np.ndarray,
    W_hidden: np.ndarray,
    b_hidden: np.ndarray,
    W_out: np.ndarray,
    b_out: np.ndarray,
    dar_hidden_1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cols.size > 0:
        h1 = np.zeros(dar_hidden_1, dtype=np.float32)
        for c in cols:
            h1 += W_embed[c]
    else:
        h1 = np.zeros(dar_hidden_1, dtype=np.float32)
    h2_pre = h1 @ W_hidden + b_hidden
    h2 = np.empty_like(h2_pre)
    for idx in range(h2_pre.size):
        h2[idx] = h2_pre[idx] if h2_pre[idx] > 0.0 else 0.0
    logits = h2 @ W_out + b_out
    return h1, h2, h2_pre, logits


@numba.njit
def dar_learn_jit(
    cols: np.ndarray,
    h1: np.ndarray,
    h2_pre: np.ndarray,
    error: np.ndarray,
    W_out: np.ndarray,
    W_hidden: np.ndarray,
    b_hidden: np.ndarray,
    W_embed: np.ndarray,
    dar_hidden_lr: float,
    dar_embed_lr: float,
) -> None:
    relu_grad = np.zeros_like(h2_pre)
    for idx in range(h2_pre.size):
        if h2_pre[idx] > 0.0:
            relu_grad[idx] = 1.0
            
    err2 = (error @ W_out.T) * relu_grad
    
    if cols.size > 0:
        h1_any = False
        for val in h1:
            if val != 0.0:
                h1_any = True
                break
        if h1_any:
            W_hidden += dar_hidden_lr * np.outer(h1, err2)
            
    b_hidden += dar_hidden_lr * err2
    
    if cols.size > 0:
        grad1 = err2 @ W_hidden.T
        for c in cols:
            W_embed[c] += dar_embed_lr * grad1


class DeepAssociativeReadout:

    def __init__(self, cfg: BILMConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(11)
        H1, H2, S = cfg.dar_hidden_1, cfg.dar_hidden_2, cfg.sdr_size
        self.W_embed  = rng.normal(0, S**-0.5,  (S,  H1)).astype(np.float32)
        self.W_hidden = rng.normal(0, H1**-0.5, (H1, H2)).astype(np.float32)
        self.b_hidden = np.zeros(H2,  dtype=np.float32)
        self.W_out    = rng.normal(0, H2**-0.5, (H2, 256)).astype(np.float32)
        self.b_out    = np.zeros(256, dtype=np.float32)
        # Adam state
        self.m_W = np.zeros_like(self.W_out)
        self.v_W = np.zeros_like(self.W_out)
        self.m_b = np.zeros_like(self.b_out)
        self.v_b = np.zeros_like(self.b_out)
        self.adam_step = 0
        # Mini-batch buffer
        self._bh2:  list[np.ndarray] = []
        self._btgt: list[int]        = []
        self._bprb: list[np.ndarray] = []
        self.updates = 0

    @property
    def weights(self) -> np.ndarray:
        return self.W_out

    @property
    def bias(self) -> np.ndarray:
        return self.b_out

    def _fwd(self, cols: np.ndarray):
        return dar_fwd_jit(
            cols,
            self.W_embed,
            self.W_hidden,
            self.b_hidden,
            self.W_out,
            self.b_out,
            self.cfg.dar_hidden_1,
        )

    def predict(self, predictive_columns, codec=None, temperature: float = 1.0, logit_bias: np.ndarray | None = None) -> np.ndarray:
        cols = np.asarray(predictive_columns, dtype=np.int64)
        _, _, _, logits = self._fwd(cols)
        if logit_bias is not None:
            logits = logits + logit_bias
        logits = logits.astype(np.float64) / max(temperature, 1e-8)
        logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        return p

    def learn(self, predictive_columns, probabilities, target: int) -> None:
        cols = np.asarray(predictive_columns, dtype=np.int64)
        h1, h2, h2_pre, _ = self._fwd(cols)

        error = -np.asarray(probabilities, np.float32)
        error[int(target) & 0xFF] += 1.0

        # Accumulate for Adam batch (L3)
        self._bh2.append(h2.copy())
        self._btgt.append(int(target) & 0xFF)
        self._bprb.append(np.asarray(probabilities, np.float32))
        if len(self._bh2) >= self.cfg.dar_batch_size:
            self._adam_update()

        # Local delta updates JIT
        dar_learn_jit(
            cols,
            h1,
            h2_pre,
            error,
            self.W_out,
            self.W_hidden,
            self.b_hidden,
            self.W_embed,
            self.cfg.dar_hidden_lr,
            self.cfg.dar_embed_lr,
        )

        self.updates += 1

    def _adam_update(self) -> None:
        if not self._bh2:
            return
        self.adam_step += 1
        t  = self.adam_step
        # Decaying lr — CL fix
        lr = self.cfg.dar_output_lr / (1.0 + self.cfg.dar_output_lr_decay * t)

        H = np.stack(self._bh2)
        P = np.stack(self._bprb).copy()
        for i, tgt in enumerate(self._btgt):
            P[i, tgt] -= 1.0
        P /= len(self._btgt)
        gW, gb = H.T @ P, P.sum(0)

        b1, b2, eps = self.cfg.dar_adam_beta1, self.cfg.dar_adam_beta2, self.cfg.dar_adam_eps
        self.m_W = b1*self.m_W + (1-b1)*gW;  self.v_W = b2*self.v_W + (1-b2)*gW**2
        self.m_b = b1*self.m_b + (1-b1)*gb;  self.v_b = b2*self.v_b + (1-b2)*gb**2
        mhW = self.m_W/(1-b1**t);  vhW = self.v_W/(1-b2**t)
        mhb = self.m_b/(1-b1**t);  vhb = self.v_b/(1-b2**t)
        self.W_out -= lr * mhW / (np.sqrt(vhW) + eps)
        self.b_out -= lr * mhb / (np.sqrt(vhb) + eps)
        self._bh2.clear(); self._btgt.clear(); self._bprb.clear()

    def save_state(self) -> dict:
        return {
            "dar_W_embed": self.W_embed, "dar_W_hidden": self.W_hidden,
            "dar_b_hidden": self.b_hidden, "dar_W_out": self.W_out,
            "dar_b_out": self.b_out, "dar_m_W": self.m_W, "dar_v_W": self.v_W,
            "dar_m_b": self.m_b, "dar_v_b": self.v_b,
            "dar_adam_step": np.array([self.adam_step], np.int64),
            "dar_updates":   np.array([self.updates],   np.int64),
        }

    def load_state(self, a: dict) -> None:
        self.W_embed[:] = a["dar_W_embed"]; self.W_hidden[:] = a["dar_W_hidden"]
        self.b_hidden[:] = a["dar_b_hidden"]; self.W_out[:] = a["dar_W_out"]
        self.b_out[:] = a["dar_b_out"]; self.m_W[:] = a["dar_m_W"]
        self.v_W[:] = a["dar_v_W"]; self.m_b[:] = a["dar_m_b"]; self.v_b[:] = a["dar_v_b"]
        self.adam_step = int(a["dar_adam_step"][0])
        self.updates   = int(a["dar_updates"][0])
