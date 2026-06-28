import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
import numpy as np

from bilm import BILM
from bilm.bilm_config import BILMConfig
from bilm.federated_merge import merge_checkpoints


def train_worker(worker_id: int, cfg_path: str, data_slice: bytes, out_path: str) -> None:
    print(f"  Worker {worker_id} started training on {len(data_slice):,} bytes...", flush=True)
    cfg = BILMConfig.load(cfg_path)
    model = BILM(cfg)

    # Train sequentially
    for b in data_slice:
        model.observe(b, learn=True)

    # Save shard checkpoint
    model.save(out_path)
    print(f"  Worker {worker_id} finished, checkpoint saved to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--tokens_per_worker", type=int, default=3000) # fast CPU default
    args = parser.parse_args()

    print(f"Loading enwik8 data...", flush=True)
    enwik8_path = Path("data/enwik8")
    if not enwik8_path.exists():
        enwik8_path = Path("bilm/data/enwik8")
    enwik8_data = enwik8_path.read_bytes()

    total_tokens = args.workers * args.tokens_per_worker
    print(f"Starting parallel training: {args.workers} workers, {args.tokens_per_worker} tokens/worker (Total={total_tokens:,})...", flush=True)

    # Create temporary config to pass to workers
    cfg = BILMConfig.from_preset("micro")
    temp_dir = Path("experiments/checkpoints")
    temp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = str(temp_dir / "temp_worker_cfg.json")
    cfg.save(cfg_path)

    processes = []
    shard_paths = []

    # Spawn workers
    for i in range(args.workers):
        start_idx = i * args.tokens_per_worker
        end_idx = start_idx + args.tokens_per_worker
        data_slice = enwik8_data[start_idx:end_idx]

        out_path = str(temp_dir / f"shard_worker_{i}.npz")
        shard_paths.append(out_path)

        p = mp.Process(
            target=train_worker,
            args=(i, cfg_path, data_slice, out_path)
        )
        processes.append(p)
        p.start()

    # Wait for all workers
    for p in processes:
        p.join()

    # 1. Merge checkpoints
    merged_path = str(temp_dir / "parallel_merged.npz")
    merge_checkpoints(shard_paths, merged_path)

    # 2. Train Single-Stream Model for comparison
    print(f"\nTraining single-stream model on {total_tokens:,} tokens...", flush=True)
    single_model = BILM(cfg)
    single_slice = enwik8_data[:total_tokens]
    t0 = time.time()
    for b in single_slice:
        single_model.observe(b, learn=True)
    elapsed_single = time.time() - t0
    single_path = str(temp_dir / "single_stream.npz")
    single_model.save(single_path)

    # 3. Evaluate and calculate gap
    eval_slice = enwik8_data[total_tokens:total_tokens + 1000]
    
    # Load merged model
    merged_model = BILM(cfg)
    merged_model.load(merged_path)

    bpb_single = single_model.evaluate(eval_slice).bits_per_byte
    bpb_merged = merged_model.evaluate(eval_slice).bits_per_byte
    gap = (bpb_merged - bpb_single) / bpb_single * 100.0

    print(f"\nEvaluation Results:")
    print(f"  Single-Stream BPB: {bpb_single:.4f} (trained in {elapsed_single:.1f}s)")
    print(f"  Merged Model BPB:  {bpb_merged:.4f}")
    print(f"  Relative BPB Gap:  {gap:.1f}%")

    # Clean up temp config
    if os.path.exists(cfg_path):
        os.remove(cfg_path)

    # Gate: gap < 10%
    assert gap < 10.0, f"Merged model failed BPB gap gate: {gap:.1f}% >= 10.0%"
    print("\nPhase 11 BPB Gap Gate: PASSED")

    # Gate: Merged model can still learn continually after merge
    print("\nVerifying merged model CL adaptation...", flush=True)
    cl_data = b"AB" * 100
    for b in cl_data:
        merged_model.observe(b, learn=True)
    r = merged_model.evaluate(cl_data[:50])
    print(f"  CL BPB after merge training: {r.bits_per_byte:.4f}")
    assert r.bits_per_byte < 7.8, "Merged model failed to learn CL pattern"
    print("Phase 11 CL Adaptation Gate: PASSED")
    print("All Phase 11 gates PASSED")


if __name__ == "__main__":
    main()
