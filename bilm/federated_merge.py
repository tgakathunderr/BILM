import numpy as np
import json
from pathlib import Path


def merge_checkpoints(paths: list[str], output_path: str, perm_prune: float = 0.40) -> None:
    """
    FederatedMerge weight merge protocol.
    Combines N independent models into a single base model.
    """
    if not paths:
        raise ValueError("At least one checkpoint path must be provided")

    print(f"Merging {len(paths)} checkpoints...", flush=True)
    data_list = [np.load(p, allow_pickle=False) for p in paths]

    # Get tokens seen lists and compute weights
    tokens_seen_list = []
    for d in data_list:
        if "tokens_seen" in d:
            tokens_seen_list.append(int(d["tokens_seen"][0]))
        else:
            tokens_seen_list.append(0)

    total_tokens = sum(tokens_seen_list)
    if total_tokens == 0:
        weights = [1.0 / len(paths)] * len(paths)
    else:
        weights = [t / total_tokens for t in tokens_seen_list]

    merged: dict[str, np.ndarray] = {}

    # 1. Codec frequencies & total seen
    merged["codec_frequencies"] = sum(d["codec_frequencies"] for d in data_list)
    merged["codec_total_seen"] = np.array([sum(int(d["codec_total_seen"][0]) for d in data_list)], dtype=np.int64)

    # 2. Context state: RESET to zero
    merged["cortex_pools"] = np.zeros_like(data_list[0]["cortex_pools"])
    if "srs_state" in data_list[0]:
        merged["srs_state"] = np.zeros_like(data_list[0]["srs_state"])

    # Find the number of layers
    n_layers = 0
    while f"cortex_L{n_layers}_targets" in data_list[0]:
        n_layers += 1

    # 3. Cortex Layer Merge (Permanences & Targets)
    for L in range(n_layers):
        print(f"  Merging Cortex Layer {L}...", flush=True)
        # Weighted average of permanences
        perms_stack = np.stack([d[f"cortex_L{L}_perms"] for d in data_list], axis=0)
        weighted_perms = sum(w * perms_stack[idx] for idx, w in enumerate(weights))

        # Targets: take from highest-weight instance
        targets_stack = np.stack([d[f"cortex_L{L}_targets"] for d in data_list], axis=0)
        max_model_idx = np.argmax(perms_stack, axis=0)
        merged_targets = np.take_along_axis(targets_stack, np.expand_dims(max_model_idx, axis=0), axis=0)[0]

        # Enforce sparsity (prune low permanences)
        prune_mask = weighted_perms < perm_prune
        weighted_perms[prune_mask] = 0.0
        merged_targets[prune_mask] = -1

        # Recompute counts & usage
        merged_counts = np.sum(merged_targets != -1, axis=1).astype(np.int32)
        merged_usage = (sum(d[f"cortex_L{L}_usage"] for d in data_list) // len(paths)).astype(np.int32)

        merged[f"cortex_L{L}_targets"] = merged_targets
        merged[f"cortex_L{L}_perms"] = weighted_perms
        merged[f"cortex_L{L}_counts"] = merged_counts
        merged[f"cortex_L{L}_usage"] = merged_usage

        # Reset active winner context lists
        merged[f"cortex_L{L}_active"] = np.zeros_like(data_list[0][f"cortex_L{L}_active"])
        merged[f"cortex_L{L}_winner"] = np.zeros_like(data_list[0][f"cortex_L{L}_winner"])
        merged[f"cortex_L{L}_predictive"] = np.zeros_like(data_list[0][f"cortex_L{L}_predictive"])
        merged[f"cortex_L{L}_prev_winner"] = np.zeros_like(data_list[0][f"cortex_L{L}_prev_winner"])

    # 4. Hippocampus: Concatenate and trim
    print("  Merging Hippocampus memory bank...", flush=True)
    hippo_size, max_synapses = data_list[0]["hippo_targets"].shape
    merged_hippo_targets = np.full((hippo_size, max_synapses), -1, dtype=np.int32)
    merged_hippo_weights = np.zeros((hippo_size, max_synapses), dtype=np.float32)

    for cell in range(hippo_size):
        # Gather all active synapses across shards for this cell
        cell_targets = []
        cell_weights = []
        for d in data_list:
            for s in range(max_synapses):
                t = int(d["hippo_targets"][cell, s])
                w_val = float(d["hippo_weights"][cell, s])
                if t != -1:
                    cell_targets.append(t)
                    cell_weights.append(w_val)

        if len(cell_targets) == 0:
            continue

        # De-duplicate: keep max weight for same targets
        unique_map = {}
        for t, w in zip(cell_targets, cell_weights):
            if t not in unique_map or w > unique_map[t]:
                unique_map[t] = w

        # Sort by weight descending
        sorted_pairs = sorted(unique_map.items(), key=lambda x: x[1], reverse=True)
        # Trim to max_synapses
        sorted_pairs = sorted_pairs[:max_synapses]

        for s_idx, (t, w) in enumerate(sorted_pairs):
            merged_hippo_targets[cell, s_idx] = t
            merged_hippo_weights[cell, s_idx] = w

    merged_hippo_counts = np.sum(merged_hippo_targets != -1, axis=1).astype(np.int16)

    merged["hippo_targets"] = merged_hippo_targets
    merged["hippo_weights"] = merged_hippo_weights
    merged["hippo_counts"] = merged_hippo_counts
    merged["hippo_last_active"] = np.zeros_like(data_list[0]["hippo_last_active"])
    merged["hippo_counters"] = np.array([
        sum(int(d["hippo_counters"][0]) for d in data_list),
        sum(int(d["hippo_counters"][1]) for d in data_list),
    ], dtype=np.int64)

    # 5. DAR (Readout) Weighted Average
    print("  Merging DeepReadout embeddings & hidden layers...", flush=True)
    dar_keys = ["dar_W_embed", "dar_W_hidden", "dar_b_hidden"]
    for k in dar_keys:
        merged[k] = sum(w * d[k] for w, d in zip(weights, data_list))

    # W_out and b_out
    merged["dar_W_out"] = sum(w * d["dar_W_out"] for w, d in zip(weights, data_list))
    merged["dar_b_out"] = sum(w * d["dar_b_out"] for w, d in zip(weights, data_list))

    # Reset Adam state
    merged["dar_m_W"] = np.zeros_like(merged["dar_W_out"], dtype=np.float64)
    merged["dar_v_W"] = np.zeros_like(merged["dar_W_out"], dtype=np.float64)
    merged["dar_m_b"] = np.zeros_like(merged["dar_b_out"], dtype=np.float64)
    merged["dar_v_b"] = np.zeros_like(merged["dar_b_out"], dtype=np.float64)
    merged["dar_adam_step"] = np.array([0], dtype=np.int64)
    merged["dar_updates"] = np.array([sum(int(d["dar_updates"][0]) for d in data_list) // len(paths)], dtype=np.int64)

    # 6. DAP & SRS dense layers
    if "dap_W" in data_list[0]:
        merged["dap_W"] = sum(w * d["dap_W"] for w, d in zip(weights, data_list))

    srs_keys = ["srs_W_k", "srs_W_v", "srs_W_q", "srs_W_proj", "srs_decay"]
    for k in srs_keys:
        if k in data_list[0]:
            merged[k] = sum(w * d[k] for w, d in zip(weights, data_list))

    # 7. Neuromodulation, Homeostasis, RNG, metadata
    merged["neuromod_surprise"] = np.array([], dtype=np.float64)
    merged["neuromod_variance"] = np.array([], dtype=np.float64)
    merged["neuromod_ach"] = np.array([1.0], dtype=np.float64)

    merged["homeostasis_state"] = np.array([
        sum(int(d["homeostasis_state"][0]) for d in data_list) // len(paths),
        sum(int(d["homeostasis_state"][1]) for d in data_list) // len(paths),
        sum(int(d["homeostasis_state"][2]) for d in data_list) // len(paths),
    ], dtype=np.int64)

    merged["generator_rng"] = data_list[0]["generator_rng"]
    merged["checkpoint_version"] = np.array([2], dtype=np.int64)
    merged["tokens_seen"] = np.array([total_tokens], dtype=np.int64)

    np.savez_compressed(output_path, **merged)
    print(f"FederatedMerge completed: saved to {output_path}", flush=True)
