"""Extension 4: Per-layer sparsity and surviving-weight allocation analysis.

This module analyzes:
1. Surviving connection counts, pruned counts, and sparsity per layer across IMP pruning levels.
2. Layerwise surviving weight share (% of total surviving network weights) across levels.
3. Actual per-round pruning rates vs. scheduled pruning rates (20% for FC1/FC2, 10% for FC3),
   evaluating the exact mathematical consequences of integer round() dynamics.
4. Empirical confirmation of identical structural allocations across seeds and datasets.
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from extension_analysis.mask_stability import (
    audit_all_runs,
    compute_file_sha256,
    get_git_info,
)

# Layer parameter constants for LeNet-300-100 (excluding biases)
LAYER_SHAPES: Dict[str, Tuple[int, int]] = {
    "fc1.weight": (300, 784),
    "fc2.weight": (100, 300),
    "fc3.weight": (10, 100),
}

LAYER_PARAMS: Dict[str, int] = {
    "fc1.weight": 235200,
    "fc2.weight": 30000,
    "fc3.weight": 1000,
}

TOTAL_PRUNABLE_PARAMS: int = 266200

SCHEDULED_PRUNE_RATES: Dict[str, float] = {
    "fc1.weight": 0.2,
    "fc2.weight": 0.2,
    "fc3.weight": 0.1,
}

LAYER_ORDER: List[str] = ["fc1.weight", "fc2.weight", "fc3.weight"]


def compute_layer_sparsity_records(
    store: Dict[Tuple[str, int, int], Dict[str, Any]],
    datasets: List[str] = ["mnist", "fashion_mnist"],
    seeds: List[int] = [1, 2, 42],
    levels: List[int] = list(range(16)),
    model: str = "lenet300100",
) -> pd.DataFrame:
    """Compute detailed per-layer sparsity and allocation metrics for all checkpoints.

    Args:
        store: Verified checkpoint store from audit_all_runs().
        datasets: List of dataset names.
        seeds: List of random seeds.
        levels: List of pruning levels (0 to 15).
        model: Model architecture name.

    Returns:
        DataFrame with detailed per-layer and overall records for each checkpoint.
    """
    records: List[Dict[str, Any]] = []

    for dataset in sorted(datasets):
        for seed in sorted(seeds):
            prev_surviving: Dict[str, int] = {}
            prev_total_surviving: Optional[int] = None

            for level in sorted(levels):
                key = (dataset, seed, level)
                if key not in store:
                    raise KeyError(f"Missing required checkpoint in store: {key}")

                ckpt_info = store[key]
                masks = ckpt_info["masks"]
                ckpt_path = str(ckpt_info.get("path", ""))
                ckpt_sha = str(ckpt_info.get("sha256", ""))

                # Compute surviving counts per layer
                surviving_counts: Dict[str, int] = {}
                for layer_name in LAYER_ORDER:
                    if layer_name not in masks:
                        raise KeyError(f"Mask {layer_name} not found in checkpoint {key}")
                    mask = masks[layer_name]
                    if mask.shape != LAYER_SHAPES[layer_name]:
                        raise ValueError(
                            f"Shape mismatch for {layer_name} in {key}: "
                            f"{mask.shape} vs {LAYER_SHAPES[layer_name]}"
                        )
                    surviving_counts[layer_name] = int(mask.sum().item())

                total_surviving = sum(surviving_counts.values())

                # Per-layer records
                for layer_name in LAYER_ORDER:
                    total_prunable = LAYER_PARAMS[layer_name]
                    surviving = surviving_counts[layer_name]
                    pruned = total_prunable - surviving
                    remaining_fraction = surviving / total_prunable
                    sparsity = 1.0 - remaining_fraction
                    remaining_percent = remaining_fraction * 100.0
                    sparsity_percent = sparsity * 100.0
                    surviving_share = surviving / total_surviving

                    if level == 0:
                        newly_pruned = float("nan")
                        actual_prune_rate = float("nan")
                        scheduled_rate = float("nan")
                        prune_rate_diff = float("nan")
                    else:
                        prev_cnt = prev_surviving[layer_name]
                        newly_pruned = float(prev_cnt - surviving)
                        actual_prune_rate = float(newly_pruned / prev_cnt)
                        scheduled_rate = float(SCHEDULED_PRUNE_RATES[layer_name])
                        prune_rate_diff = float(actual_prune_rate - scheduled_rate)

                    records.append({
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "level": level,
                        "layer": layer_name,
                        "total_prunable": total_prunable,
                        "surviving": surviving,
                        "pruned": pruned,
                        "remaining_fraction": remaining_fraction,
                        "sparsity": sparsity,
                        "remaining_percent": remaining_percent,
                        "sparsity_percent": sparsity_percent,
                        "surviving_share": surviving_share,
                        "newly_pruned": newly_pruned,
                        "actual_prune_rate": actual_prune_rate,
                        "scheduled_prune_rate": scheduled_rate,
                        "prune_rate_diff": prune_rate_diff,
                        "checkpoint_path": ckpt_path,
                        "checkpoint_sha256": ckpt_sha,
                    })

                # Overall network record
                overall_total_prunable = TOTAL_PRUNABLE_PARAMS
                overall_surviving = total_surviving
                overall_pruned = overall_total_prunable - overall_surviving
                overall_rem_frac = overall_surviving / overall_total_prunable
                overall_sparsity = 1.0 - overall_rem_frac
                overall_rem_pct = overall_rem_frac * 100.0
                overall_sparsity_pct = overall_sparsity * 100.0
                overall_share = 1.0

                if level == 0:
                    overall_newly_pruned = float("nan")
                    overall_act_rate = float("nan")
                    overall_sched_rate = float("nan")
                    overall_rate_diff = float("nan")
                else:
                    prev_tot = prev_total_surviving
                    assert prev_tot is not None
                    overall_newly_pruned = float(prev_tot - overall_surviving)
                    overall_act_rate = float(overall_newly_pruned / prev_tot)

                    # Scheduled overall newly pruned according to formula
                    sched_newly_pruned = sum(
                        round(prev_surviving[ly] * SCHEDULED_PRUNE_RATES[ly])
                        for ly in LAYER_ORDER
                    )
                    overall_sched_rate = float(sched_newly_pruned / prev_tot)
                    overall_rate_diff = float(overall_act_rate - overall_sched_rate)

                records.append({
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "level": level,
                    "layer": "overall",
                    "total_prunable": overall_total_prunable,
                    "surviving": overall_surviving,
                    "pruned": overall_pruned,
                    "remaining_fraction": overall_rem_frac,
                    "sparsity": overall_sparsity,
                    "remaining_percent": overall_rem_pct,
                    "sparsity_percent": overall_sparsity_pct,
                    "surviving_share": overall_share,
                    "newly_pruned": overall_newly_pruned,
                    "actual_prune_rate": overall_act_rate,
                    "scheduled_prune_rate": overall_sched_rate,
                    "prune_rate_diff": overall_rate_diff,
                    "checkpoint_path": ckpt_path,
                    "checkpoint_sha256": ckpt_sha,
                })

                # Update previous level tracking
                prev_surviving = {ly: surviving_counts[ly] for ly in LAYER_ORDER}
                prev_total_surviving = total_surviving

    df = pd.DataFrame.from_records(records)
    return df


def aggregate_layer_sparsity_summary(df_detailed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-layer sparsity metrics across seeds by dataset, level, and layer.

    Args:
        df_detailed: Output from compute_layer_sparsity_records().

    Returns:
        Summary DataFrame with aggregated descriptive statistics.
    """
    group_cols = ["dataset", "level", "layer"]
    summary_rows: List[Dict[str, Any]] = []

    grouped = df_detailed.groupby(group_cols, sort=True)
    for (dataset, level, layer), group in grouped:
        n_seeds = len(group["seed"].unique())
        total_prunable = int(group["total_prunable"].iloc[0])

        surviving_vals = group["surviving"].to_numpy()
        rem_frac_vals = group["remaining_fraction"].to_numpy()
        sparsity_vals = group["sparsity"].to_numpy()
        share_vals = group["surviving_share"].to_numpy()

        newly_pruned_vals = group["newly_pruned"].dropna().to_numpy()
        actual_rate_vals = group["actual_prune_rate"].dropna().to_numpy()
        sched_rate_vals = group["scheduled_prune_rate"].dropna().to_numpy()
        diff_vals = group["prune_rate_diff"].dropna().to_numpy()

        # Check identical counts
        counts_identical = bool(surviving_vals.min() == surviving_vals.max())

        summary_rows.append({
            "dataset": dataset,
            "level": level,
            "layer": layer,
            "n_seeds": n_seeds,
            "counts_identical": counts_identical,
            "total_prunable": total_prunable,
            "surviving_mean": float(np.mean(surviving_vals)),
            "surviving_sd": float(np.std(surviving_vals, ddof=1)) if n_seeds > 1 else 0.0,
            "surviving_min": int(np.min(surviving_vals)),
            "surviving_max": int(np.max(surviving_vals)),
            "remaining_fraction_mean": float(np.mean(rem_frac_vals)),
            "sparsity_mean": float(np.mean(sparsity_vals)),
            "surviving_share_mean": float(np.mean(share_vals)),
            "newly_pruned_mean": float(np.mean(newly_pruned_vals)) if len(newly_pruned_vals) > 0 else float("nan"),
            "actual_prune_rate_mean": float(np.mean(actual_rate_vals)) if len(actual_rate_vals) > 0 else float("nan"),
            "scheduled_prune_rate": float(sched_rate_vals[0]) if len(sched_rate_vals) > 0 else float("nan"),
            "prune_rate_diff_mean": float(np.mean(diff_vals)) if len(diff_vals) > 0 else float("nan"),
        })

    return pd.DataFrame.from_records(summary_rows)


def create_layer_allocation_summary(df_detailed: pd.DataFrame) -> pd.DataFrame:
    """Create a clean pivot table indexed by level showing surviving counts and shares.

    Because counts are identical across datasets and seeds, representative values
    are extracted for presentation.

    Args:
        df_detailed: Output from compute_layer_sparsity_records().

    Returns:
        DataFrame indexed by level with layerwise surviving counts, shares, and sparsity.
    """
    # Filter to one representative run (e.g. mnist, seed 1) to build presentation table
    rep = df_detailed[(df_detailed["dataset"] == "mnist") & (df_detailed["seed"] == 1)].copy()

    records: List[Dict[str, Any]] = []
    for level in sorted(rep["level"].unique()):
        lvl_data = rep[rep["level"] == level]

        fc1 = lvl_data[lvl_data["layer"] == "fc1.weight"].iloc[0]
        fc2 = lvl_data[lvl_data["layer"] == "fc2.weight"].iloc[0]
        fc3 = lvl_data[lvl_data["layer"] == "fc3.weight"].iloc[0]
        overall = lvl_data[lvl_data["layer"] == "overall"].iloc[0]

        records.append({
            "level": int(level),
            "overall_surviving": int(overall["surviving"]),
            "overall_pct_remaining": float(overall["remaining_fraction"] * 100.0),
            "overall_sparsity": float(overall["sparsity"] * 100.0),
            "overall_prune_rate": float(overall["actual_prune_rate"] * 100.0) if pd.notnull(overall["actual_prune_rate"]) else float("nan"),
            "fc1_surviving": int(fc1["surviving"]),
            "fc1_share": float(fc1["surviving_share"] * 100.0),
            "fc1_sparsity": float(fc1["sparsity"] * 100.0),
            "fc1_prune_rate": float(fc1["actual_prune_rate"] * 100.0) if pd.notnull(fc1["actual_prune_rate"]) else float("nan"),
            "fc2_surviving": int(fc2["surviving"]),
            "fc2_share": float(fc2["surviving_share"] * 100.0),
            "fc2_sparsity": float(fc2["sparsity"] * 100.0),
            "fc2_prune_rate": float(fc2["actual_prune_rate"] * 100.0) if pd.notnull(fc2["actual_prune_rate"]) else float("nan"),
            "fc3_surviving": int(fc3["surviving"]),
            "fc3_share": float(fc3["surviving_share"] * 100.0),
            "fc3_sparsity": float(fc3["sparsity"] * 100.0),
            "fc3_prune_rate": float(fc3["actual_prune_rate"] * 100.0) if pd.notnull(fc3["actual_prune_rate"]) else float("nan"),
        })

    return pd.DataFrame.from_records(records)


def verify_ext4_artifacts(
    execution_dir: Path,
    runs_dir: Path = Path("runs"),
    results_dir: Optional[Path] = Path("results"),
) -> Tuple[bool, Dict[str, List[str]]]:
    """Verify presence and structural validity of Extension 4 artifacts.

    Checks file existence and required table schemas without blocking on SHA-256 comparisons.

    Args:
        execution_dir: Directory containing Extension 4 results.
        runs_dir: Root runs directory.
        results_dir: Root results directory.

    Returns:
        Tuple of (is_valid, errors_dict)
    """
    errors: Dict[str, List[str]] = {
        "inputs": [],
        "code": [],
        "outputs": [],
    }

    if not execution_dir.is_dir():
        errors["outputs"].append(f"Execution directory does not exist: {execution_dir}")
        return False, errors

    meta_path = execution_dir / "metadata.json"
    if not meta_path.exists():
        errors["outputs"].append(f"metadata.json missing in {execution_dir}")
        return False, errors

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        errors["outputs"].append(f"Failed to read metadata.json: {e}")
        return False, errors

    # Verify input checkpoints exist
    ckpt_hashes = meta.get("input_checkpoint_hashes", {})
    for rel_path in ckpt_hashes.keys():
        p = Path(rel_path)
        if not p.exists():
            errors["inputs"].append(f"Input checkpoint {rel_path} does not exist")

    # Verify output tables exist and validate required columns
    required_table_cols = {
        "tables/layer_sparsity_detailed.csv": [
            "dataset",
            "model",
            "seed",
            "level",
            "layer",
            "surviving",
            "pruned",
            "sparsity",
            "surviving_share",
        ],
        "tables/layer_sparsity_summary.csv": [
            "dataset",
            "level",
            "layer",
            "counts_identical",
            "surviving_mean",
            "sparsity_mean",
            "surviving_share_mean",
        ],
        "tables/layer_allocation_summary.csv": [
            "level",
            "overall_surviving",
            "fc1_surviving",
            "fc1_share",
            "fc2_surviving",
            "fc2_share",
            "fc3_surviving",
            "fc3_share",
        ],
    }

    for rel_path, req_cols in required_table_cols.items():
        p = execution_dir / rel_path
        if not p.exists():
            errors["outputs"].append(f"Output table {rel_path} missing")
        else:
            try:
                df_head = pd.read_csv(p, nrows=5)
                missing_cols = [c for c in req_cols if c not in df_head.columns]
                if missing_cols:
                    errors["outputs"].append(f"Output table {rel_path} missing required columns: {missing_cols}")
            except Exception as exc:
                errors["outputs"].append(f"Failed to read output table {rel_path}: {exc}")

    # Verify output figures exist
    fig_hashes = meta.get("output_figure_hashes", {})
    for rel_path in fig_hashes.keys():
        p = execution_dir / rel_path
        if not p.exists():
            errors["outputs"].append(f"Output figure {rel_path} missing")

    is_valid = len(errors["inputs"]) == 0 and len(errors["code"]) == 0 and len(errors["outputs"]) == 0
    return is_valid, errors


def run_extension4_pipeline(
    runs_dir: Path | str = "runs",
    results_dir: Optional[Path | str] = "results",
    execution_id: Optional[str] = None,
    out_base: Path | str = "results/extensions",
) -> Path:
    """Execute the complete Extension 4 analysis pipeline.

    Args:
        runs_dir: Path to directory containing formal run checkpoints.
        results_dir: Path to original results directory.
        execution_id: Optional explicit execution identifier.
        out_base: Base directory for storing extension executions.

    Returns:
        Path to newly created execution directory.
    """
    from extension_analysis.plots import (
        plot_layer_sparsity_curves,
        plot_layer_share_evolution,
        plot_pruning_schedule_adherence,
    )

    runs_path = Path(runs_dir)
    results_path = Path(results_dir) if results_dir else None
    out_base_path = Path(out_base)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if execution_id is None:
        short_uuid = uuid.uuid4().hex[:6]
        execution_id = f"ext4_{timestamp}_{short_uuid}"

    out_dir = out_base_path / execution_id
    if out_dir.exists():
        raise FileExistsError(f"Execution directory already exists: {out_dir}")

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=False)

    # 1. Audit all formal checkpoints and load masks
    datasets = ["mnist", "fashion_mnist"]
    seeds = [1, 2, 42]
    levels = list(range(16))
    model = "lenet300100"

    df_audit, store = audit_all_runs(
        runs_dir=runs_path,
        results_dir=results_path,
        datasets=datasets,
        seeds=seeds,
        levels=levels,
        model=model,
    )

    # 2. Compute layer sparsity records
    df_detailed = compute_layer_sparsity_records(
        store=store,
        datasets=datasets,
        seeds=seeds,
        levels=levels,
        model=model,
    )

    # 3. Compute summaries
    df_summary = aggregate_layer_sparsity_summary(df_detailed)
    df_allocation = create_layer_allocation_summary(df_detailed)

    # Verify counts identical across seeds and datasets
    seed_variance_zero = bool(df_summary["counts_identical"].all())
    max_rate_diff = float(df_detailed["prune_rate_diff"].abs().dropna().max())

    # 4. Save tables
    detailed_csv = tables_dir / "layer_sparsity_detailed.csv"
    summary_csv = tables_dir / "layer_sparsity_summary.csv"
    allocation_csv = tables_dir / "layer_allocation_summary.csv"

    df_detailed.to_csv(detailed_csv, index=False)
    df_summary.to_csv(summary_csv, index=False)
    df_allocation.to_csv(allocation_csv, index=False)

    # 5. Generate and save figures
    fig_curves_path = figures_dir / "layer_sparsity_curves.png"
    fig_share_path = figures_dir / "layer_share_evolution.png"
    fig_sched_path = figures_dir / "pruning_schedule_adherence.png"

    plot_layer_sparsity_curves(df_summary, out_path=fig_curves_path)
    plot_layer_share_evolution(df_summary, out_path=fig_share_path)
    plot_pruning_schedule_adherence(df_summary, out_path=fig_sched_path)

    # 6. Build file hashes and inventory
    table_hashes: Dict[str, str] = {
        f"tables/{detailed_csv.name}": compute_file_sha256(detailed_csv),
        f"tables/{summary_csv.name}": compute_file_sha256(summary_csv),
        f"tables/{allocation_csv.name}": compute_file_sha256(allocation_csv),
    }

    fig_hashes: Dict[str, str] = {
        f"figures/{fig_curves_path.name}": compute_file_sha256(fig_curves_path),
        f"figures/{fig_share_path.name}": compute_file_sha256(fig_share_path),
        f"figures/{fig_sched_path.name}": compute_file_sha256(fig_sched_path),
    }

    inventory_path = out_dir / "artifact_inventory.csv"
    inventory_rows = [
        {"category": "table", "relpath": k, "sha256": v, "size_bytes": (out_dir / k).stat().st_size}
        for k, v in table_hashes.items()
    ] + [
        {"category": "figure", "relpath": k, "sha256": v, "size_bytes": (out_dir / k).stat().st_size}
        for k, v in fig_hashes.items()
    ]
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)

    # Track code file hashes
    code_files = [
        Path("extension_analysis/layer_sparsity.py"),
        Path("extension_analysis/plots.py"),
        Path("lth/models.py"),
        Path("lth/pruning.py"),
        Path("run_imp.py"),
    ]
    code_hashes: Dict[str, str] = {}
    for cf in code_files:
        if cf.exists():
            code_hashes[str(cf)] = compute_file_sha256(cf)

    # Track input checkpoint, config, and summary hashes
    ckpt_hashes: Dict[str, str] = {}
    config_hashes: Dict[str, str] = {}
    summary_hashes: Dict[str, str] = {}

    for d in datasets:
        for s in seeds:
            run_name = f"{d}_adam_seed{s}"
            cfg_p = runs_path / run_name / "config.json"
            if cfg_p.exists():
                config_hashes[run_name] = compute_file_sha256(cfg_p)
            if results_path:
                sum_p = results_path / run_name / "summary.csv"
                if sum_p.exists():
                    summary_hashes[run_name] = compute_file_sha256(sum_p)

            for l in levels:
                cp = runs_path / run_name / f"level_{l:02d}.pt"
                if cp.exists():
                    ckpt_hashes[str(cp)] = compute_file_sha256(cp)

    metadata: Dict[str, Any] = {
        "extension": "Extension 4: Per-Layer Sparsity and Surviving-Weight Allocation",
        "execution_id": execution_id,
        "timestamp": timestamp,
        "model": model,
        "layers": LAYER_ORDER,
        "layer_shapes": {k: list(v) for k, v in LAYER_SHAPES.items()},
        "layer_param_counts": LAYER_PARAMS,
        "total_prunable_parameters": TOTAL_PRUNABLE_PARAMS,
        "scheduled_prune_rates": SCHEDULED_PRUNE_RATES,
        "pruning_strategy": "layerwise",
        "supported_datasets": datasets,
        "seeds": seeds,
        "levels": levels,
        "git": get_git_info(),
        "config_hashes": config_hashes,
        "summary_hashes": summary_hashes,
        "input_checkpoint_hashes": ckpt_hashes,
        "code_file_hashes": code_hashes,
        "output_table_hashes": table_hashes,
        "output_figure_hashes": fig_hashes,
        "summary_statistics": {
            "n_checkpoints_evaluated": len(store),
            "layer_counts_identical_across_seeds": seed_variance_zero,
            "max_prune_rate_deviation": max_rate_diff,
        },
    }

    meta_file = out_dir / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_dir

