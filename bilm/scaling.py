"""Phase 6: Consumer-CPU scaling and profiling."""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from bilm import BILM
from bilm.config import SDR_SIZE, SDR_SPARSITY, CELLS_PER_COLUMN, MAX_SYNAPSES_PER_CELL, HIPPO_SIZE


@dataclass(frozen=True)
class MemoryProfile:
    sdr_size: int
    cells_per_column: int
    max_synapses_per_cell: int
    hippo_size: int
    cortex_arrays_mb: float
    hippo_arrays_mb: float
    readout_mb: float
    codec_mb: float
    total_estimated_mb: float


@dataclass(frozen=True)
class ThroughputProfile:
    observe_tps: float
    evaluate_tps: float
    tokens_per_second: float


@dataclass(frozen=True)
class CheckpointProfile:
    checkpoint_size_mb: float
    save_time_s: float
    load_time_s: float


def estimate_memory_footprint() -> MemoryProfile:
    """Estimate RAM usage of all BILM arrays in MB."""
    total_cells = SDR_SIZE * CELLS_PER_COLUMN

    cortex_per_layer = (
        total_cells * MAX_SYNAPSES_PER_CELL * 4 +   # connected_targets (int32)
        total_cells * MAX_SYNAPSES_PER_CELL * 4 +   # permanences (float32)
        total_cells * 4 +                            # synapse_counts (int32)
        total_cells * 1 +                            # active_cells (bool)
        total_cells * 1 +                            # winner_cells (bool)
        total_cells * 1 +                            # predictive_cells (bool)
        total_cells * 1 +                            # prev_winner_cells (bool)
        total_cells * 4                              # cell_usage (int32)
    )
    cortex_total = cortex_per_layer * 3  # 3 layers
    pools = SDR_SIZE * 4 * 3              # temporal pools

    hippo = (
        HIPPO_SIZE * 64 * 4 +             # targets (int32)
        HIPPO_SIZE * 64 * 4 +             # weights (float32)
        HIPPO_SIZE * 2                     # counts (int16)
    )

    readout = SDR_SIZE * 256 * 4 + 256 * 4  # weights + bias

    codec = 256 * SDR_SPARSITY * 4 + 256 * 8 * 1 + 256 * 8  # byte_sdrs + active_bits + frequencies

    total = cortex_total + pools + hippo + readout + codec

    return MemoryProfile(
        sdr_size=SDR_SIZE,
        cells_per_column=CELLS_PER_COLUMN,
        max_synapses_per_cell=MAX_SYNAPSES_PER_CELL,
        hippo_size=HIPPO_SIZE,
        cortex_arrays_mb=cortex_total / (1024 * 1024),
        hippo_arrays_mb=hippo / (1024 * 1024),
        readout_mb=readout / (1024 * 1024),
        codec_mb=codec / (1024 * 1024),
        total_estimated_mb=total / (1024 * 1024),
    )


def profile_throughput(train_bytes: int = 200, eval_bytes: int = 100) -> ThroughputProfile:
    """Measure tokens per second for observe and evaluate."""
    model = BILM()
    data = b"throughput test " * (train_bytes // 16 + 1)

    t0 = time.perf_counter()
    for b in data[:train_bytes]:
        model.observe(int(b), learn=True)
    train_time = time.perf_counter() - t0
    observe_tps = train_bytes / max(train_time, 1e-9)

    eval_data = data[:eval_bytes]
    t0 = time.perf_counter()
    model.evaluate(eval_data, warmup=0, reset_context=True)
    eval_time = time.perf_counter() - t0
    eval_tps = eval_bytes / max(eval_time, 1e-9)

    return ThroughputProfile(
        observe_tps=observe_tps,
        evaluate_tps=eval_tps,
        tokens_per_second=observe_tps,
    )


def profile_checkpoint(train_bytes: int = 100) -> CheckpointProfile:
    """Measure checkpoint save/load size and time."""
    model = BILM()
    data = b"checkpoint profiling " * (train_bytes // 20 + 1)
    for b in data[:train_bytes]:
        model.observe(int(b), learn=True)

    import tempfile
    path = os.path.join(tempfile.gettempdir(), "bilm_profile.npz")

    t0 = time.perf_counter()
    model.save(path)
    save_time = time.perf_counter() - t0

    size_mb = os.path.getsize(path) / (1024 * 1024)

    t0 = time.perf_counter()
    model2 = BILM.from_checkpoint(path)
    load_time = time.perf_counter() - t0

    os.unlink(path)

    return CheckpointProfile(
        checkpoint_size_mb=size_mb,
        save_time_s=save_time,
        load_time_s=load_time,
    )


def generate_scaling_curves() -> dict:
    """Generate scaling curves for key parameters."""
    baseline_config = {
        "sdr_size": SDR_SIZE,
        "cells_per_column": CELLS_PER_COLUMN,
        "max_synapses": MAX_SYNAPSES_PER_CELL,
        "hippo_size": HIPPO_SIZE,
    }

    mem_profile = estimate_memory_footprint()
    throughput = profile_throughput()
    ckpt = profile_checkpoint()

    return {
        "baseline_config": baseline_config,
        "memory_profile": asdict(mem_profile),
        "throughput": asdict(throughput),
        "checkpoint": asdict(ckpt),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="BILM 2 profiling")
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    print("=== Phase 6: Consumer-CPU Scaling ===")
    print("\nMemory profile:")
    mem = estimate_memory_footprint()
    print(f"  Cortex arrays: {mem.cortex_arrays_mb:.1f} MB")
    print(f"  Hippocampus:   {mem.hippo_arrays_mb:.1f} MB")
    print(f"  Readout:       {mem.readout_mb:.1f} MB")
    print(f"  Codec:         {mem.codec_mb:.1f} MB")
    print(f"  Total est.:    {mem.total_estimated_mb:.1f} MB")

    print("\nThroughput profile:")
    tp = profile_throughput()
    print(f"  Observe TPS:   {tp.observe_tps:.1f}")
    print(f"  Evaluate TPS:  {tp.evaluate_tps:.1f}")

    print("\nCheckpoint profile:")
    ckpt = profile_checkpoint()
    print(f"  Size:          {ckpt.checkpoint_size_mb:.1f} MB")
    print(f"  Save time:     {ckpt.save_time_s:.3f}s")
    print(f"  Load time:     {ckpt.load_time_s:.3f}s")

    results = generate_scaling_curves()
    output = Path(args.output_dir) / "scaling_profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
