import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bilm.metrics import target_loss_bits, bits_per_byte, byte_perplexity
from bilm.results import Prediction, ObservationResult, EvaluationReport

class SSMModel(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, d_state=16):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.log_A = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).view(1, 1, d_state) * 0.1))
        self.x_proj = nn.Linear(d_model, d_state * 2)
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_state = d_state
        self.d_model = d_model

    def forward(self, x):
        x_emb = self.token_emb(x)
        b, t, d = x_emb.size()
        A = torch.exp(-torch.exp(self.log_A))
        proj = self.x_proj(x_emb)
        B, C = torch.chunk(proj, 2, dim=-1)
        h = torch.zeros(b, self.d_state, dtype=x_emb.dtype, device=x_emb.device)
        outs = []
        for step in range(t):
            x_step = x_emb[:, step, :]
            B_step = B[:, step, :]
            C_step = C[:, step, :]
            h = A[0] * h + B_step * x_step.mean(-1, keepdim=True)
            out_step = x_step + (h.sum(-1, keepdim=True) * C_step).mean(-1, keepdim=True)
            outs.append(out_step)
        out = torch.stack(outs, dim=1)
        out = self.ln(out)
        logits = self.lm_head(out)
        return logits

class MinimalSSM:
    def __init__(self, lr=1e-3):
        self.model = SSMModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.context = []
        self.max_len = 512

    def predict_next(self, temperature: float = 1.0) -> Prediction:
        self.model.eval()
        if not self.context:
            probs = np.full(256, 1.0 / 256.0, dtype=np.float64)
        else:
            inputs = torch.tensor([self.context[-self.max_len:]], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(inputs)[0, -1, :]
            logits = logits.double().numpy()
            logits = logits / max(temperature, 1e-8)
            logits -= logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()

        argmax = int(np.argmax(probs))
        return Prediction(
            probabilities=probs,
            argmax=argmax,
            predictive_columns=np.array([], dtype=np.int64),
            confidence=float(probs[argmax])
        )

    def observe(self, byte: int, learn: bool = True) -> ObservationResult:
        target = int(byte) & 0xFF
        prior = self.predict_next()
        loss_val = target_loss_bits(prior.probabilities, target)

        if learn and self.context:
            self.model.train()
            inputs = torch.tensor([self.context[-self.max_len:]], dtype=torch.long)
            logits = self.model(inputs)[0, -1, :]
            target_tensor = torch.tensor([target], dtype=torch.long)
            loss = F.cross_entropy(logits.unsqueeze(0), target_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.context.append(target)
        if len(self.context) > self.max_len:
            self.context.pop(0)

        next_pred = self.predict_next()
        return ObservationResult(
            target=target,
            prior_prediction=prior,
            next_prediction=next_pred,
            loss_bits=loss_val,
            surprise=0.0
        )

    def evaluate(self, data: bytes, warmup: int = 0) -> EvaluationReport:
        saved_context = list(self.context)
        self.context.clear()
        losses = []
        correct = 0
        for index, value in enumerate(data):
            prior = self.predict_next()
            loss = self.observe(value, learn=False).loss_bits
            if index >= warmup:
                losses.append(loss)
                correct += int(prior.argmax == int(value))
        self.context = saved_context
        bpb = bits_per_byte(losses)
        return EvaluationReport(
            tokens=len(losses),
            bits_per_byte=bpb,
            perplexity=byte_perplexity(bpb),
            accuracy=(correct / len(losses)) if losses else 0.0,
        )

    def generate(self, prompt: str, max_bytes: int = 128, temperature: float = 0.8) -> str:
        saved_context = list(self.context)
        self.context.clear()
        for b in prompt.encode("utf-8", errors="replace"):
            self.observe(b, learn=False)
            
        output_bytes = []
        for _ in range(max_bytes):
            prior = self.predict_next(temperature=temperature)
            next_b = prior.argmax
            output_bytes.append(next_b)
            self.observe(next_b, learn=False)
            
        self.context = saved_context
        return bytes(output_bytes).decode("utf-8", errors="replace")
