"""Extension 2: Density-matched random mask reference analysis for IMP subnets."""

import csv
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from extension_analysis.mask_stability import (
    DEFAULT_MODEL,
    EXPECTED_COMMON_CONFIG,
    EXPECTED_DATASETS,
    EXPECTED_LAYERS,
    EXPECTED_LEVELS,
    EXPECTED_SEEDS,
    EXPECTED_SHAPES,
    compute_file_sha256,
    get_git_info,
    verify_execution_artifacts,
)

DEFAULT_BASE_SEED = 6792026
DEFAULT_REPLICATES = 2000

# Strict index maps to avoid any Python hash() dependency
DATASET_INDEX_MAP = {"mnist": 0, "fashion_mnist": 1}
LAYER_INDEX_MAP = {"fc1.weight": 0, "fc2.weight": 1, "fc3.weight": 2}

# Convergence / stability diagnostic tolerances
DIAGNOSTIC_TOLERANCE_MEAN = 0.001
DIAGNOSTIC_TOLERANCE_QUANTILE = 0.005
MIN_REPLICATES_FOR_DIAGNOSTIC = 2000


def get_layer_rng(
    base_seed: int,
    dataset_name: str,
    level: int,
    replicate_idx: int,
    layer_name: str,
) -> np.random.Generator:
    """Derive an independent, order-invariant, prefix-stable RNG for a specific layer realization."""
    if dataset_name not in DATASET_INDEX_MAP:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'; must be one of {list(DATASET_INDEX_MAP.keys())}"
        )
    if layer_name not in LAYER_INDEX_MAP:
        raise ValueError(
            f"Unknown layer '{layer_name}'; must be one of {list(LAYER_INDEX_MAP.keys())}"
        )

    dataset_id = DATASET_INDEX_MAP[dataset_name]
    layer_id = LAYER_INDEX_MAP[layer_name]

    # Explicit 5-tuple entropy
    seed_tuple = (
        int(base_seed),
        int(dataset_id),
        int(level),
        int(replicate_idx),
        int(layer_id),
    )
    ss = np.random.SeedSequence(seed_tuple)
    return np.random.default_rng(ss)


def sample_joint_hypergeometric_overlaps(
    n_total: int,
    n_retained: int,
    num_seeds: int,
    rng: np.random.Generator,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Sample exact joint intersections and unions for 2 or 3 independent random subsets of size K from N.

    Returns dict mapping (idx_a, idx_b) -> (intersection, union).
    Indices: 0, 1, 2.
    """
    if num_seeds not in (2, 3):
        raise ValueError(f"num_seeds must be 2 or 3, got {num_seeds}")

    k = n_retained
    n = n_total

    if k < 0 or k > n:
        raise ValueError(f"Retained count k={k} must be within [0, {n}]")

    if k == n:
        pairs = [(0, 1)] if num_seeds == 2 else [(0, 1), (0, 2), (1, 2)]
        return {p: (n, n) for p in pairs}

    if k == 0:
        pairs = [(0, 1)] if num_seeds == 2 else [(0, 1), (0, 2), (1, 2)]
        return {p: (0, 0) for p in pairs}

    # 1. First pair (0, 1) intersection is standard hypergeometric
    i01 = int(rng.hypergeometric(k, n - k, k))
    u01 = 2 * k - i01

    if num_seeds == 2:
        return {(0, 1): (i01, u01)}

    # 2. Partition N into 4 regions defined by seed 0 and seed 1:
    # R11 = 0 & 1, size = i01
    # R10 = 0 \ 1, size = k - i01
    # R01 = 1 \ 0, size = k - i01
    # R00 = (0 | 1)^c, size = n - u01
    n11 = i01
    n10 = k - i01
    n01 = k - i01
    n00 = n - u01

    # Seed 2 is an independent random draw of k items from N.
    # Allocation of seed 2 across (R11, R10, R01, R00) follows a multivariate hypergeometric.
    c11 = int(rng.hypergeometric(n11, n - n11, k))
    rem_n = n - n11
    rem_k = k - c11

    c10 = int(rng.hypergeometric(n10, rem_n - n10, rem_k))
    rem_n -= n10
    rem_k -= c10

    c01 = int(rng.hypergeometric(n01, rem_n - n01, rem_k))

    i02 = c11 + c10
    u02 = 2 * k - i02

    i12 = c11 + c01
    u12 = 2 * k - i12

    return {
        (0, 1): (i01, u01),
        (0, 2): (i02, u02),
        (1, 2): (i12, u12),
    }


def sample_explicit_masks_overlap(
    n_total: int,
    n_retained: int,
    num_seeds: int,
    rng: np.random.Generator,
) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Explicit boolean mask generation (exact baseline simulation for unit test verification)."""
    masks = []
    for _ in range(num_seeds):
        m = np.zeros(n_total, dtype=bool)
        chosen = rng.choice(n_total, n_retained, replace=False)
        m[chosen] = True
        masks.append(m)

    pairs = [(0, 1)] if num_seeds == 2 else [(0, 1), (0, 2), (1, 2)]
    results = {}
    for i, j in pairs:
        inter = int((masks[i] & masks[j]).sum())
        uni = int((masks[i] | masks[j]).sum())
        results[(i, j)] = (inter, uni)
    return results


def validate_and_extract_ext1_data(
    df_pairs: pd.DataFrame,
    df_summary: pd.DataFrame,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Strictly validate Extension 1 pairs and summary tables.

    Checks:
    - Required columns and valid categories.
    - Direction canonicality (seed_a < seed_b), no self-comparison, no duplicates.
    - All pairs have records for 'overall' and all three layers.
    - Layerwise survivor counts match across all pairs and seeds.
    - weights_remaining matches layerwise counts.
    - Summary mean, std, and n_pairs recompute exactly from pairs table.
    """
    req_pair_cols = [
        "dataset",
        "model",
        "level",
        "weights_remaining",
        "seed_a",
        "seed_b",
        "scope",
        "surviving_a",
        "surviving_b",
        "intersection",
        "union",
        "jaccard",
    ]
    for c in req_pair_cols:
        if c not in df_pairs.columns:
            raise KeyError(f"Missing required column '{c}' in Extension 1 pairs table")

    req_sum_cols = [
        "dataset",
        "model",
        "level",
        "weights_remaining",
        "scope",
        "mean",
        "std",
        "min",
        "max",
        "n_pairs",
    ]
    for c in req_sum_cols:
        if c not in df_summary.columns:
            raise KeyError(f"Missing required column '{c}' in Extension 1 summary table")

    # Layer parameter sizes
    layer_sizes = {
        lyr: int(math.prod(EXPECTED_SHAPES[lyr])) for lyr in EXPECTED_LAYERS
    }
    total_network_params = sum(layer_sizes.values())

    # Check pair validity
    for row in df_pairs.itertuples():
        if row.model != DEFAULT_MODEL:
            raise ValueError(f"Invalid model '{row.model}' in pairs table")
        if row.dataset not in EXPECTED_DATASETS:
            raise ValueError(f"Invalid dataset '{row.dataset}' in pairs table")
        if row.scope not in ["overall"] + EXPECTED_LAYERS:
            raise ValueError(f"Invalid scope '{row.scope}' in pairs table")
        if row.seed_a >= row.seed_b:
            raise ValueError(
                f"Pair direction violated or self-comparison: seed_a={row.seed_a}, seed_b={row.seed_b}"
            )

    # Check for duplicates
    dup_mask = df_pairs.duplicated(subset=["dataset", "level", "seed_a", "seed_b", "scope"])
    if dup_mask.any():
        dups = df_pairs[dup_mask]
        raise ValueError(f"Duplicate records found in pairs table: {dups.to_dict(orient='records')}")

    # Re-verify summary against pairs
    sum_lookup = {}
    for row in df_summary.itertuples():
        sum_lookup[(row.dataset, row.level, row.scope)] = {
            "mean": float(row.mean),
            "std": float(row.std) if pd.notnull(row.std) else float("nan"),
            "n_pairs": int(row.n_pairs),
            "weights_remaining": float(row.weights_remaining),
        }

    grouped_conditions = {}
    grouped_pairs = df_pairs.groupby(["dataset", "level"])

    for (ds, lvl), grp in grouped_pairs:
        # Check overall rows
        overall_rows = grp[grp["scope"] == "overall"]
        pairs_list = sorted(list(set(zip(overall_rows["seed_a"], overall_rows["seed_b"]))))
        unique_seeds = sorted(list(set(overall_rows["seed_a"]).union(set(overall_rows["seed_b"]))))
        n_pairs = len(pairs_list)

        # Pair configuration validation
        if len(unique_seeds) == 3:
            expected_pairs = [(unique_seeds[0], unique_seeds[1]), (unique_seeds[0], unique_seeds[2]), (unique_seeds[1], unique_seeds[2])]
            if pairs_list != expected_pairs:
                raise ValueError(
                    f"Incomplete 3-seed pair set for {ds} lvl {lvl}: got {pairs_list}, expected {expected_pairs}"
                )
        elif len(unique_seeds) == 2:
            expected_pairs = [(unique_seeds[0], unique_seeds[1])]
            if pairs_list != expected_pairs:
                raise ValueError(
                    f"Incomplete 2-seed pair set for {ds} lvl {lvl}: got {pairs_list}, expected {expected_pairs}"
                )
        else:
            raise ValueError(
                f"Unsupported seed count {len(unique_seeds)} for {ds} lvl {lvl}: {unique_seeds}"
            )

        # Check that every pair has exactly overall and all 3 layers
        for p in pairs_list:
            p_sub = grp[(grp["seed_a"] == p[0]) & (grp["seed_b"] == p[1])]
            scopes_found = set(p_sub["scope"])
            if scopes_found != {"overall", "fc1.weight", "fc2.weight", "fc3.weight"}:
                raise ValueError(
                    f"Pair {p} in {ds} lvl {lvl} missing scopes: found {scopes_found}"
                )

        # Validate layerwise survival counts
        layer_counts = {}
        for lyr in EXPECTED_LAYERS:
            lyr_rows = grp[grp["scope"] == lyr]
            surv_a_vals = lyr_rows["surviving_a"].unique()
            surv_b_vals = lyr_rows["surviving_b"].unique()
            if len(surv_a_vals) != 1 or len(surv_b_vals) != 1 or surv_a_vals[0] != surv_b_vals[0]:
                raise ValueError(
                    f"Mismatched surviving counts in {ds} lvl {lvl} {lyr}: surv_a={surv_a_vals}, surv_b={surv_b_vals}"
                )
            k_lyr = int(surv_a_vals[0])
            n_lyr = layer_sizes[lyr]
            if k_lyr < 0 or k_lyr > n_lyr:
                raise ValueError(
                    f"Surviving count {k_lyr} out of range [0, {n_lyr}] for {lyr} in {ds} lvl {lvl}"
                )
            layer_counts[lyr] = (n_lyr, k_lyr)

        # Check overall weights_remaining
        tot_surv = sum(k for _, k in layer_counts.values())
        expected_frac = tot_surv / total_network_params
        actual_frac = float(overall_rows["weights_remaining"].iloc[0])
        if abs(expected_frac - actual_frac) > 1e-6:
            raise ValueError(
                f"weights_remaining mismatch in {ds} lvl {lvl}: calc={expected_frac} vs recorded={actual_frac}"
            )

        # Re-check summary values for every scope
        for scp in ["overall"] + EXPECTED_LAYERS:
            scp_sub = grp[grp["scope"] == scp]
            recomp_n = len(scp_sub)
            recomp_mean = float(scp_sub["jaccard"].mean())
            recomp_sd = float(scp_sub["jaccard"].std(ddof=1)) if recomp_n > 1 else float("nan")

            if (ds, lvl, scp) not in sum_lookup:
                raise KeyError(f"Condition {(ds, lvl, scp)} missing in summary table")

            sum_entry = sum_lookup[(ds, lvl, scp)]
            if sum_entry["n_pairs"] != recomp_n:
                raise ValueError(
                    f"Summary n_pairs mismatch for {(ds, lvl, scp)}: {sum_entry['n_pairs']} != {recomp_n}"
                )
            if abs(sum_entry["mean"] - recomp_mean) > 1e-6:
                raise ValueError(
                    f"Summary mean mismatch for {(ds, lvl, scp)}: {sum_entry['mean']} != {recomp_mean}"
                )
            if recomp_n > 1:
                if abs(sum_entry["std"] - recomp_sd) > 1e-6:
                    raise ValueError(
                        f"Summary std mismatch for {(ds, lvl, scp)}: {sum_entry['std']} != {recomp_sd}"
                    )
            else:
                if not math.isnan(sum_entry["std"]):
                    raise ValueError(
                        f"Summary std must be NaN for n_pairs=1, got {sum_entry['std']}"
                    )

        grouped_conditions[(ds, lvl)] = {
            "dataset": ds,
            "level": lvl,
            "weights_remaining": actual_frac,
            "unique_seeds": unique_seeds,
            "pairs": pairs_list,
            "n_pairs": n_pairs,
            "layer_counts": layer_counts,
        }

    return grouped_conditions


def simulate_single_replicate(
    dataset_name: str,
    level: int,
    replicate_idx: int,
    layer_counts: Dict[str, Tuple[int, int]],
    unique_seeds: List[int],
    base_seed: int = DEFAULT_BASE_SEED,
) -> Dict[str, float]:
    """Simulate one Monte Carlo replicate with independent RNG per layer.

    Guarantees that layer traversal order does NOT affect any layer's realization.
    """
    num_seeds = len(unique_seeds)
    if num_seeds not in (2, 3):
        raise ValueError(f"Expected 2 or 3 seeds, got {num_seeds}")

    pair_indices = [(0, 1)] if num_seeds == 2 else [(0, 1), (0, 2), (1, 2)]

    pair_layer_stats = {p: {"inter": {}, "union": {}} for p in pair_indices}
    layer_jaccards_by_pair = {lyr: [] for lyr in EXPECTED_LAYERS}

    for lyr_name in EXPECTED_LAYERS:
        n_total, n_retained = layer_counts[lyr_name]
        # Independent layer-specific RNG
        rng = get_layer_rng(
            base_seed=base_seed,
            dataset_name=dataset_name,
            level=level,
            replicate_idx=replicate_idx,
            layer_name=lyr_name,
        )
        overlaps = sample_joint_hypergeometric_overlaps(
            n_total=n_total,
            n_retained=n_retained,
            num_seeds=num_seeds,
            rng=rng,
        )
        for p in pair_indices:
            inter, uni = overlaps[p]
            pair_layer_stats[p]["inter"][lyr_name] = inter
            pair_layer_stats[p]["union"][lyr_name] = uni
            j_layer = (inter / uni) if uni > 0 else float("nan")
            layer_jaccards_by_pair[lyr_name].append(j_layer)

    replicate_scopes = {}
    for lyr_name in EXPECTED_LAYERS:
        vals = layer_jaccards_by_pair[lyr_name]
        valid_vals = [v for v in vals if not math.isnan(v)]
        replicate_scopes[lyr_name] = (
            float(np.mean(valid_vals)) if valid_vals else float("nan")
        )

    # Compute overall: pool intersections and unions across all layers per pair, then average across pairs
    overall_jaccards = []
    for p in pair_indices:
        tot_inter = sum(pair_layer_stats[p]["inter"].values())
        tot_union = sum(pair_layer_stats[p]["union"].values())
        j_overall = (tot_inter / tot_union) if tot_union > 0 else float("nan")
        overall_jaccards.append(j_overall)

    valid_overall = [v for v in overall_jaccards if not math.isnan(v)]
    replicate_scopes["overall"] = (
        float(np.mean(valid_overall)) if valid_overall else float("nan")
    )

    return replicate_scopes


def run_random_reference_simulation(
    df_ext1_pairs: pd.DataFrame,
    df_ext1_summary: pd.DataFrame,
    datasets: List[str] = EXPECTED_DATASETS,
    levels: List[int] = EXPECTED_LEVELS,
    replicates: int = DEFAULT_REPLICATES,
    base_seed: int = DEFAULT_BASE_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full Monte Carlo random reference simulation across all datasets, levels, and scopes."""
    conditions = validate_and_extract_ext1_data(df_pairs=df_ext1_pairs, df_summary=df_ext1_summary)

    sample_records = []
    summary_records = []
    scopes = ["overall"] + EXPECTED_LAYERS

    for ds in datasets:
        for lvl in levels:
            key = (ds, lvl)
            if key not in conditions:
                continue

            cond = conditions[key]
            unique_seeds = cond["unique_seeds"]
            pairs = cond["pairs"]
            n_pairs = cond["n_pairs"]
            layer_counts = cond["layer_counts"]
            weights_remaining = cond["weights_remaining"]
            pair_id_str = ";".join(f"{p[0]}-{p[1]}" for p in pairs)

            # Pre-allocate replicate values: scope -> array of length replicates
            rep_values = {scp: np.zeros(replicates, dtype=float) for scp in scopes}

            for r in range(replicates):
                rep_results = simulate_single_replicate(
                    dataset_name=ds,
                    level=lvl,
                    replicate_idx=r,
                    layer_counts=layer_counts,
                    unique_seeds=unique_seeds,
                    base_seed=base_seed,
                )
                for scp in scopes:
                    val = rep_results[scp]
                    rep_values[scp][r] = val
                    sample_records.append(
                        {
                            "dataset": ds,
                            "model": DEFAULT_MODEL,
                            "level": lvl,
                            "weights_remaining": weights_remaining,
                            "scope": scp,
                            "replicate": r,
                            "pair_mean_jaccard": val,
                            "n_pairs": n_pairs,
                            "seed_pairs": pair_id_str,
                        }
                    )

            # Compute summary and diagnostics per scope
            for scp in scopes:
                obs_row = df_ext1_summary[
                    (df_ext1_summary["dataset"] == ds)
                    & (df_ext1_summary["level"] == lvl)
                    & (df_ext1_summary["scope"] == scp)
                ]
                obs_mean = float(obs_row["mean"].iloc[0])
                obs_sd = (
                    float(obs_row["std"].iloc[0])
                    if pd.notnull(obs_row["std"].iloc[0])
                    else float("nan")
                )

                arr = rep_values[scp]
                valid_arr = arr[~np.isnan(arr)]

                if len(valid_arr) == 0:
                    ref_mean = float("nan")
                    ref_sd = float("nan")
                    ref_p025 = float("nan")
                    ref_p975 = float("nan")
                    ref_mcse = float("nan")
                    obs_diff = float("nan")
                    diagnostic_status = "not_evaluated"
                    is_stable = None
                    diag_mean_diff = float("nan")
                    diag_p025_diff = float("nan")
                    diag_p975_diff = float("nan")
                else:
                    ref_mean = float(np.mean(valid_arr))
                    ref_sd = float(np.std(valid_arr, ddof=1)) if len(valid_arr) > 1 else 0.0
                    ref_p025 = float(np.percentile(valid_arr, 2.5))
                    ref_p975 = float(np.percentile(valid_arr, 97.5))
                    ref_mcse = float(ref_sd / np.sqrt(len(valid_arr)))
                    obs_diff = float(obs_mean - ref_mean)

                    if replicates >= MIN_REPLICATES_FOR_DIAGNOSTIC:
                        diagnostic_status = "evaluated"
                        first_1000 = arr[:1000]
                        valid_1000 = first_1000[~np.isnan(first_1000)]
                        mean_1000 = float(np.mean(valid_1000))
                        p025_1000 = float(np.percentile(valid_1000, 2.5))
                        p975_1000 = float(np.percentile(valid_1000, 97.5))

                        diag_mean_diff = float(abs(ref_mean - mean_1000))
                        diag_p025_diff = float(abs(ref_p025 - p025_1000))
                        diag_p975_diff = float(abs(ref_p975 - p975_1000))

                        is_stable = bool(
                            diag_mean_diff <= DIAGNOSTIC_TOLERANCE_MEAN
                            and diag_p025_diff <= DIAGNOSTIC_TOLERANCE_QUANTILE
                            and diag_p975_diff <= DIAGNOSTIC_TOLERANCE_QUANTILE
                        )
                    else:
                        diagnostic_status = "not_evaluated"
                        is_stable = None
                        diag_mean_diff = float("nan")
                        diag_p025_diff = float("nan")
                        diag_p975_diff = float("nan")

                summary_records.append(
                    {
                        "dataset": ds,
                        "model": DEFAULT_MODEL,
                        "level": lvl,
                        "weights_remaining": weights_remaining,
                        "scope": scp,
                        "observed_mean": obs_mean,
                        "observed_pair_sd": obs_sd,
                        "n_pairs": n_pairs,
                        "reference_mean": ref_mean,
                        "reference_sd": ref_sd,
                        "reference_p025": ref_p025,
                        "reference_p975": ref_p975,
                        "reference_mean_mcse": ref_mcse,
                        "observed_minus_reference": obs_diff,
                        "n_replicates": replicates,
                        "count_source": "layerwise_exact",
                        "coverage_info": f"{len(unique_seeds)} seeds ({pair_id_str})",
                        "diagnostic_status": diagnostic_status,
                        "is_stable": is_stable,
                        "diag_mean_diff": diag_mean_diff,
                        "diag_p025_diff": diag_p025_diff,
                        "diag_p975_diff": diag_p975_diff,
                    }
                )

    df_samples = pd.DataFrame(sample_records)
    df_summary = pd.DataFrame(summary_records)

    scope_order = {"overall": 0, "fc1.weight": 1, "fc2.weight": 2, "fc3.weight": 3}
    df_summary["_order"] = df_summary["scope"].map(lambda x: scope_order.get(x, 99))
    df_summary = (
        df_summary.sort_values(by=["dataset", "_order", "level"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    return df_samples, df_summary


def verify_ext2_artifacts(
    ext2_dir: Path,
    expected_ext1_dir: Optional[Path] = None,
    runs_dir: Path = Path("runs"),
    results_dir: Optional[Path] = Path("results"),
) -> Tuple[bool, Dict[str, List[str]]]:
    """Verify integrity of an existing Extension 2 execution directory.

    Returns:
        (is_valid, categorized_errors_dict)
        Categorized errors distinguish:
        - 'inputs': underlying checkpoints/configs/ext1 source tables changed (requires rerun).
        - 'code': code files altered.
        - 'outputs': output artifacts missing or corrupted.
    """
    errors = {"inputs": [], "code": [], "outputs": []}

    if not ext2_dir.is_dir():
        errors["outputs"].append(f"Execution directory not found: {ext2_dir}")
        return False, errors

    meta_path = ext2_dir / "metadata.json"
    if not meta_path.is_file():
        errors["outputs"].append(f"metadata.json missing in {ext2_dir}")
        return False, errors

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        errors["outputs"].append(f"Failed to parse metadata.json in {ext2_dir}: {e}")
        return False, errors

    req_files = [
        ext2_dir / "tables" / "random_reference_samples.csv",
        ext2_dir / "tables" / "random_reference_summary.csv",
        ext2_dir / "figures" / "random_reference_grid.png",
        ext2_dir / "figures" / "random_reference_grid.pdf",
        ext2_dir / "figures" / "random_reference_diff.png",
        ext2_dir / "figures" / "random_reference_diff.pdf",
    ]
    for rf in req_files:
        if not rf.is_file():
            errors["outputs"].append(f"Missing required artifact: {rf}")

    # Check output artifacts SHA-256
    recorded_outputs = meta.get("outputs", {})
    for out_name, out_path_str in recorded_outputs.items():
        op = Path(out_path_str)
        if op.is_file():
            recorded_sha = meta.get("output_files_sha256", {}).get(str(op))
            if recorded_sha:
                curr_sha = compute_file_sha256(op)
                if curr_sha != recorded_sha:
                    errors["outputs"].append(
                        f"Output file {op.name} modified: recorded={recorded_sha} current={curr_sha}"
                    )

    # Check source Extension 1 directory & files
    ext1_source_path_str = meta.get("source_extension1_path")
    if expected_ext1_dir is not None:
        if ext1_source_path_str != str(expected_ext1_dir):
            errors["inputs"].append(
                f"Source extension 1 path mismatch: recorded={ext1_source_path_str} expected={expected_ext1_dir}"
            )

    ext1_dir = Path(ext1_source_path_str) if ext1_source_path_str else None
    if ext1_dir is None or not ext1_dir.is_dir():
        errors["inputs"].append(f"Source Extension 1 directory missing: {ext1_dir}")
    else:
        # Check Extension 1 files against recorded SHA-256
        recorded_ext1_hashes = meta.get("source_extension1_files_sha256", {})
        if not recorded_ext1_hashes:
            errors["inputs"].append("No source Extension 1 file hashes recorded in metadata.json")
        for ext1_file_str, expected_sha in recorded_ext1_hashes.items():
            efp = Path(ext1_file_str)
            if not efp.is_file():
                errors["inputs"].append(f"Extension 1 source file missing: {efp}")
            else:
                curr_sha = compute_file_sha256(efp)
                if curr_sha != expected_sha:
                    errors["inputs"].append(
                        f"Extension 1 source file {efp.name} modified: recorded={expected_sha} current={curr_sha}"
                    )

    # Check underlying checkpoints and config hashes
    recorded_inputs = meta.get("inputs", {})
    for p_str, info in recorded_inputs.items():
        expected_sha = info.get("sha256")
        if not expected_sha:
            continue
        p = Path(p_str)
        if not p.is_file():
            errors["inputs"].append(f"Underlying input file missing: {p}")
        else:
            actual_sha = compute_file_sha256(p)
            if actual_sha != expected_sha:
                errors["inputs"].append(
                    f"Underlying input file {p.name} modified: current={actual_sha} recorded={expected_sha}"
                )

    # Check code files SHA-256
    recorded_code = meta.get("code_files_sha256", {})
    for code_p_str, expected_sha in recorded_code.items():
        cp = Path(code_p_str)
        if not cp.is_file():
            errors["code"].append(f"Code file missing: {cp}")
        else:
            actual_sha = compute_file_sha256(cp)
            if actual_sha != expected_sha:
                errors["code"].append(
                    f"Code file {cp.name} modified: current={actual_sha} recorded={expected_sha}"
                )

    total_errors = sum(len(err_list) for err_list in errors.values())
    return (total_errors == 0), errors


def run_extension2_pipeline(
    ext1_execution_dir: str,
    runs_dir: str = "runs",
    results_dir: str = "results",
    execution_id: Optional[str] = None,
    out_base: str = "results/extensions",
    replicates: int = DEFAULT_REPLICATES,
    base_seed: int = DEFAULT_BASE_SEED,
) -> Dict[str, Any]:
    """Execute complete Extension 2 pipeline."""
    ext1_path = Path(ext1_execution_dir)
    runs_path = Path(runs_dir)
    results_path = Path(results_dir) if results_dir else None

    # 1. Strictly verify the provided Extension 1 artifacts
    ext1_valid, ext1_errors = verify_execution_artifacts(ext1_path, runs_path, results_path)
    if not ext1_valid:
        raise ValueError(
            f"Provided Extension 1 execution {ext1_path} failed integrity checks:\n"
            + "\n".join(ext1_errors)
        )

    ext1_meta_file = ext1_path / "metadata.json"
    ext1_pairs_file = ext1_path / "tables" / "mask_jaccard_pairs.csv"
    ext1_sum_file = ext1_path / "tables" / "mask_jaccard_summary.csv"
    ext1_inv_file = ext1_path / "artifact_inventory.csv"

    for f in [ext1_meta_file, ext1_pairs_file, ext1_sum_file, ext1_inv_file]:
        if not f.is_file():
            raise FileNotFoundError(f"Required Extension 1 file missing: {f}")

    # Compute Extension 1 files SHA-256
    ext1_files_sha256 = {
        str(ext1_meta_file): compute_file_sha256(ext1_meta_file),
        str(ext1_pairs_file): compute_file_sha256(ext1_pairs_file),
        str(ext1_sum_file): compute_file_sha256(ext1_sum_file),
        str(ext1_inv_file): compute_file_sha256(ext1_inv_file),
    }

    df_ext1_pairs = pd.read_csv(ext1_pairs_file)
    df_ext1_summary = pd.read_csv(ext1_sum_file)

    # 2. Setup output directory with unique ID and overwrite protection
    if not execution_id:
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        execution_id = f"ext2_{now_str}_{short_id}"

    out_dir = Path(out_base) / execution_id
    if out_dir.exists():
        raise FileExistsError(
            f"Execution directory already exists: {out_dir}. Overwriting an existing execution is forbidden."
        )

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=False)

    # 3. Run Monte Carlo simulation with per-layer SeedSequence
    df_samples, df_summary = run_random_reference_simulation(
        df_ext1_pairs=df_ext1_pairs,
        df_ext1_summary=df_ext1_summary,
        datasets=EXPECTED_DATASETS,
        levels=EXPECTED_LEVELS,
        replicates=replicates,
        base_seed=base_seed,
    )

    samples_path = tables_dir / "random_reference_samples.csv"
    summary_path = tables_dir / "random_reference_summary.csv"
    df_samples.to_csv(samples_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    # 4. Generate plots
    from extension_analysis.plots import (
        plot_random_reference_difference,
        plot_random_reference_grid,
    )

    grid_png = figures_dir / "random_reference_grid.png"
    grid_pdf = figures_dir / "random_reference_grid.pdf"
    plot_random_reference_grid(
        df_summary=df_summary,
        save_png=grid_png,
        save_pdf=grid_pdf,
    )

    diff_png = figures_dir / "random_reference_diff.png"
    diff_pdf = figures_dir / "random_reference_diff.pdf"
    plot_random_reference_difference(
        df_summary=df_summary,
        save_png=diff_png,
        save_pdf=diff_pdf,
    )

    # 5. Metadata JSON
    ext1_meta = json.loads(ext1_meta_file.read_text(encoding="utf-8"))

    # Record code file SHA-256
    code_files = [
        Path("extension_analysis/random_reference.py"),
        Path("extension_analysis/plots.py"),
    ]
    code_meta = {}
    for cf in code_files:
        if cf.is_file():
            code_meta[str(cf)] = compute_file_sha256(cf)

    # Record output files SHA-256
    output_files = [samples_path, summary_path, grid_png, grid_pdf, diff_png, diff_pdf]
    output_meta = {str(of): compute_file_sha256(of) for of in output_files}

    stable_conditions = (
        int(df_summary["is_stable"].sum())
        if df_summary["is_stable"].notnull().any()
        else 0
    )
    all_stable = (
        bool(df_summary["is_stable"].all())
        if replicates >= MIN_REPLICATES_FOR_DIAGNOSTIC
        else False
    )

    metadata = {
        "execution_id": execution_id,
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "source_extension1_path": str(ext1_path),
        "source_extension1_id": ext1_meta.get("execution_id"),
        "source_extension1_files_sha256": ext1_files_sha256,
        "git": get_git_info(),
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "simulation_parameters": {
            "algorithm": "joint_hypergeometric_partition_sampling",
            "base_seed": base_seed,
            "replicates": replicates,
            "seed_sequence_rule": "SeedSequence((base_seed, dataset_id, level, replicate_idx, layer_id))",
            "datasets": EXPECTED_DATASETS,
            "levels": EXPECTED_LEVELS,
            "diagnostic_tolerances": {
                "min_replicates_required": MIN_REPLICATES_FOR_DIAGNOSTIC,
                "mean_abs_diff": DIAGNOSTIC_TOLERANCE_MEAN,
                "quantile_abs_diff": DIAGNOSTIC_TOLERANCE_QUANTILE,
            },
        },
        "stability_diagnostics": {
            "diagnostic_status": "evaluated" if replicates >= MIN_REPLICATES_FOR_DIAGNOSTIC else "not_evaluated",
            "total_conditions": len(df_summary),
            "stable_conditions": stable_conditions,
            "all_conditions_stable": all_stable,
        },
        "outputs": {
            "samples_csv": str(samples_path),
            "summary_csv": str(summary_path),
            "grid_png": str(grid_png),
            "grid_pdf": str(grid_pdf),
            "diff_png": str(diff_png),
            "diff_pdf": str(diff_pdf),
        },
        "output_files_sha256": output_meta,
        "code_files_sha256": code_meta,
        "inputs": ext1_meta.get("inputs", {}),
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "execution_id": execution_id,
        "out_dir": out_dir,
        "df_samples": df_samples,
        "df_summary": df_summary,
        "metadata_path": meta_path,
        "grid_png": grid_png,
        "grid_pdf": grid_pdf,
        "diff_png": diff_png,
        "diff_pdf": diff_pdf,
    }
