import argparse
import json
import time
import math
from pathlib import Path
import numpy as np
from scipy import stats

from bilm import BILM
from bilm.bilm_config import BILMConfig
from bilm.baselines.tiny_transformer import TinyTransformer
from bilm.grammars import generate_ab_pattern


def run_cl_experiment(seed: int, tokens: int):
    # Set random seeds
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    # 1. Generate disjoint domain data
    # Domain A: AB pattern
    train_a = generate_ab_pattern(tokens)
    eval_a = generate_ab_pattern(100)

    # Domain B: XY pattern
    train_b = bytes([ord("X") if b == ord("A") else ord("Y") for b in train_a])
    eval_b = bytes([ord("X") if b == ord("A") else ord("Y") for b in eval_a])

    # 2. Initialize models (using micro scale for fast CPU execution)
    cfg = BILMConfig.from_preset("micro")
    model_bilm = BILM(cfg)
    model_trans = TinyTransformer()

    results = {}

    # === BILM Training ===
    # Train A
    for b in train_a:
        model_bilm.observe(b, learn=True)
    bpb_a_best_bilm = model_bilm.evaluate(eval_a).bits_per_byte

    # Train B and measure adaptation tokens
    adapt_tokens_bilm = tokens
    for idx, b in enumerate(train_b):
        r = model_bilm.observe(b, learn=True)
        if r.loss_bits < 5.0 and adapt_tokens_bilm == tokens:
            adapt_tokens_bilm = idx + 1
            
    bpb_a_after_bilm = model_bilm.evaluate(eval_a).bits_per_byte
    forgetting_bilm = max(0.0, (bpb_a_after_bilm - bpb_a_best_bilm) / max(bpb_a_best_bilm, 1e-5)) * 100.0

    # === Transformer Training ===
    # Train A
    for b in train_a:
        model_trans.observe(b, learn=True)
    bpb_a_best_trans = model_trans.evaluate(eval_a).bits_per_byte

    # Train B and measure adaptation tokens
    adapt_tokens_trans = tokens
    for idx, b in enumerate(train_b):
        r = model_trans.observe(b, learn=True)
        if r.loss_bits < 5.0 and adapt_tokens_trans == tokens:
            adapt_tokens_trans = idx + 1
            
    bpb_a_after_trans = model_trans.evaluate(eval_a).bits_per_byte
    forgetting_trans = max(0.0, (bpb_a_after_trans - bpb_a_best_trans) / max(bpb_a_best_trans, 1e-5)) * 100.0

    return {
        "forgetting_bilm": forgetting_bilm,
        "forgetting_trans": forgetting_trans,
        "adapt_bilm": adapt_tokens_bilm,
        "adapt_trans": adapt_tokens_trans,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=10000) # Default to 10k for fast execution on CPU
    args = parser.parse_args()

    print(f"Running CL Proof: seeds={args.seeds}, tokens={args.tokens}...", flush=True)

    metrics = []
    for s in range(args.seeds):
        t0 = time.time()
        res = run_cl_experiment(s, args.tokens)
        elapsed = time.time() - t0
        print(f"  Seed {s} finished in {elapsed:.1f}s | BILM Forget: {res['forgetting_bilm']:.1f}%, Trans Forget: {res['forgetting_trans']:.1f}%", flush=True)
        metrics.append(res)

    # Compute statistics
    f_bilm = [m["forgetting_bilm"] for m in metrics]
    f_trans = [m["forgetting_trans"] for m in metrics]
    a_bilm = [m["adapt_bilm"] for m in metrics]
    a_trans = [m["adapt_trans"] for m in metrics]

    mean_f_bilm, std_f_bilm = np.mean(f_bilm), np.std(f_bilm, ddof=1)
    mean_f_trans, std_f_trans = np.mean(f_trans), np.std(f_trans, ddof=1)
    mean_a_bilm, std_a_bilm = np.mean(a_bilm), np.std(a_bilm, ddof=1)
    mean_a_trans, std_a_trans = np.mean(a_trans), np.std(a_trans, ddof=1)

    # Paired t-test
    t_stat_f, p_val_f = stats.ttest_rel(f_bilm, f_trans)
    t_stat_a, p_val_a = stats.ttest_rel(a_bilm, a_trans)

    # 95% confidence interval of difference
    diff_f = np.array(f_trans) - np.array(f_bilm)
    mean_diff_f = np.mean(diff_f)
    sem_diff_f = stats.sem(diff_f)
    ci_f = stats.t.interval(0.95, len(diff_f)-1, loc=mean_diff_f, scale=sem_diff_f)

    significant_f = "YES" if p_val_f < 0.05 else "NO"
    significant_a = "YES" if p_val_a < 0.05 else "NO"

    verdict = "SUPPORTED" if (mean_f_bilm < mean_f_trans and p_val_f < 0.05) else "INCONCLUSIVE"

    # Write report
    report_dir = Path("experiments/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    results_path = results_dir / f"cl_proof_{timestamp}.json"
    results_path.write_text(json.dumps({
        "forgetting_bilm": f_bilm,
        "forgetting_trans": f_trans,
        "adapt_bilm": a_bilm,
        "adapt_trans": a_trans,
    }, indent=2))

    report_content = f"""## CL Proof Results

| Metric | BILM | Transformer | p-value | Significant |
|--------|------|-------------|---------|-------------|
| Forgetting (%) | {mean_f_bilm:.1f} ± {std_f_bilm:.1f} | {mean_f_trans:.1f} ± {std_f_trans:.1f} | {p_val_f:.4f} | {significant_f} |
| Adaptation tokens | {mean_a_bilm:.1f} ± {std_a_bilm:.1f} | {mean_a_trans:.1f} ± {std_a_trans:.1f} | {p_val_a:.4f} | {significant_a} |

## Verdict: {verdict}

BILM forgets {mean_f_trans/max(mean_f_bilm, 1e-5):.1f}x less than Transformer (p={p_val_f:.4f}, 95% CI of diff: [{ci_f[0]:.1f}, {ci_f[1]:.1f}]).
"""
    (report_dir / "cl_proof_report.md").write_text(report_content)
    print("\n" + report_content)


if __name__ == "__main__":
    main()
