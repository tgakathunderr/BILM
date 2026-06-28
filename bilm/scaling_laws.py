import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bilm import BILM
from bilm.bilm_config import BILMConfig


def get_small_config() -> BILMConfig:
    # A config slightly larger than micro (~6M params)
    return BILMConfig(
        sdr_size=8_192, cells_per_column=4, max_synapses=16, sdr_sparsity=32,
        n_layers=2, layer_decays=(0.80, 0.99),
        hippo_size=2_048, dar_hidden_1=512, dar_hidden_2=256,
        srs_dim=256, dap_dim=512,
    )


def train_and_eval(cfg: BILMConfig, tokens: int, enwik8_data: bytes) -> float:
    model = BILM(cfg)
    train_slice = enwik8_data[:tokens]
    # Use a fixed evaluation slice to avoid corpus entropy noise
    eval_slice = enwik8_data[50000:51000]

    for b in train_slice:
        model.observe(b, learn=True)

    bpb = model.evaluate(eval_slice).bits_per_byte
    return bpb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=str, default="micro,small")
    parser.add_argument("--tokens", type=str, default="1000,3000,6000") # fast CPU default
    args = parser.parse_args()

    configs_list = args.configs.split(",")
    tokens_list = [int(t) for t in args.tokens.split(",")]

    print(f"Loading enwik8 data...", flush=True)
    enwik8_path = Path("data/enwik8")
    if not enwik8_path.exists():
        enwik8_path = Path("bilm/data/enwik8")
    enwik8_data = enwik8_path.read_bytes()

    # Define the config objects and get parameter counts
    cfg_map = {}
    for name in configs_list:
        if name == "micro":
            cfg_map["micro"] = BILMConfig.from_preset("micro")
        elif name == "small":
            cfg_map["small"] = get_small_config()
        else:
            raise ValueError(f"Unknown config name: {name}")

    results = []

    print(f"Running scaling laws grid search...", flush=True)
    for name, cfg in cfg_map.items():
        params = cfg.param_count()
        for tok in tokens_list:
            t0 = time.time()
            bpb = train_and_eval(cfg, tok, enwik8_data)
            elapsed = time.time() - t0
            print(f"  Config: {name:<6} (params={params:,}) | Tokens: {tok:,} | BPB: {bpb:.4f} | {elapsed:.1f}s", flush=True)
            results.append({
                "config": name,
                "params": params,
                "tokens": tok,
                "bpb": bpb
            })

    # Fit scaling law: log(BPB) = -alpha * log(params) - beta * log(tokens) + C
    log_params = np.array([np.log(r["params"]) for r in results])
    log_tokens = np.array([np.log(r["tokens"]) for r in results])
    log_bpb = np.array([np.log(r["bpb"]) for r in results])

    X = np.stack([log_params, log_tokens, np.ones_like(log_params)], axis=1)
    y = log_bpb

    # Linear least squares
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha = -float(w[0])
    beta = -float(w[1])
    C = float(w[2])

    # Compute R-squared
    y_pred = X @ w
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-8))

    print(f"\nScaling Law Fit Results:")
    print(f"  alpha (params exponent): {alpha:.4f}")
    print(f"  beta (tokens exponent):  {beta:.4f}")
    print(f"  Constant C:              {C:.4f}")
    print(f"  R^2 Fit Confidence:      {r2:.4f}")

    # Write report files
    reports_dir = Path("experiments/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "scaling_laws.json").write_text(json.dumps({
        "results": results,
        "alpha": alpha,
        "beta": beta,
        "C": C,
        "r2": r2,
    }, indent=2))

    # Plot scaling laws
    plt.figure(figsize=(8, 6))
    for name in configs_list:
        cfg_res = [r for r in results if r["config"] == name]
        toks = [r["tokens"] for r in cfg_res]
        bpbs = [r["bpb"] for r in cfg_res]
        plt.plot(toks, bpbs, "o-", label=f"{name} ({cfg_map[name].param_count():,} params)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Tokens Trained")
    plt.ylabel("BPB Generalization Loss")
    plt.title(f"BILM Scaling Laws (R² = {r2:.4f})")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(str(reports_dir / "scaling_laws.png"), dpi=150)
    plt.close()

    # Write summary markdown
    markdown_content = f"""# Scaling Laws Experiment Report

## Empirical Power-Law Fit
We fit the scaling relationship:
$$\\log(\\text{{BPB}}) = -\\alpha \\log(\\text{{Params}}) - \\beta \\log(\\text{{Tokens}}) + C$$

*   **Alpha (parameter scaling coefficient)**: `{alpha:.4f}`
*   **Beta (token scaling coefficient)**: `{beta:.4f}`
*   **Constant C**: `{C:.4f}`
*   **R² Confidence Score**: `{r2:.4f}`

## Verdict: {"PASSED" if r2 > 0.90 else "BOTTLENECKED"}

The scaling law fit confidence is `{r2:.4f}` (Gate target: `> 0.90`).
"""
    (reports_dir / "scaling_laws_report.md").write_text(markdown_content)
    print(f"\nReport written to experiments/reports/scaling_laws_report.md")


if __name__ == "__main__":
    main()
