import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bilm.metrics import target_loss_bits, bits_per_byte, byte_perplexity
from bilm.results import Prediction, ObservationResult, EvaluationReport

class LSTMModel(nn.Module):
    def __init__(self, vocab_size=256, d_model=128, n_layers=2, d_ff=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_ff, num_layers=n_layers,
            batch_first=True, dropout=0.0
        )
        self.ln = nn.LayerNorm(d_ff)
        self.lm_head = nn.Linear(d_ff, vocab_size, bias=False)

    def forward(self, x, h=None):
        x = self.token_emb(x)
        out, h_next = self.lstm(x, h)
        out = self.ln(out)
        logits = self.lm_head(out)
        return logits, h_next

class LSTM_LM:
    def __init__(self, lr=1e-3):
        self.model = LSTMModel()
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
                logits, _ = self.model(inputs)
                logits = logits[0, -1, :]
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
            logits, _ = self.model(inputs)
            logits = logits[0, -1, :]
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
