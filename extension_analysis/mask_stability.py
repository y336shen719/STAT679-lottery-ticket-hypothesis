"""Cross-seed mask stability analysis (Jaccard similarity) for IMP subnets."""

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
import torch

EXPECTED_LAYERS = ["fc1.weight", "fc2.weight", "fc3.weight"]
EXPECTED_SHAPES = {
    "fc1.weight": (300, 784),
    "fc2.weight": (100, 300),
    "fc3.weight": (10, 100),
}
EXPECTED_SEEDS = [1, 2, 42]
EXPECTED_DATASETS = ["mnist", "fashion_mnist"]
EXPECTED_LEVELS = list(range(16))
DEFAULT_MODEL = "lenet300100"

EXPECTED_COMMON_CONFIG = {
    "model": "lenet300100",
    "split_seed": 0,
    "rounds": 15,
    "steps": 50000,
    "batch_size": 60,
    "eval_every": 100,
    "lr": 0.0012,
    "optimizer": "adam",
    "pruning": "layerwise",
}


def compute_layer_jaccard(
    mask_a: torch.Tensor, mask_b: torch.Tensor
) -> Dict[str, Any]:
    """Compute surviving count, intersection, union, and Jaccard for two binary masks.

    Does not modify inputs.
    """
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(mask_a.shape)} vs {tuple(mask_b.shape)}"
        )
    # Ensure boolean tensors without in-place modification
    m_a = mask_a.detach().to(dtype=torch.bool, device="cpu")
    m_b = mask_b.detach().to(dtype=torch.bool, device="cpu")

    surv_a = int(m_a.sum().item())
    surv_b = int(m_b.sum().item())
    inter = int((m_a & m_b).sum().item())
    uni = int((m_a | m_b).sum().item())

    jaccard = float(inter / uni) if uni > 0 else float("nan")
    return {
        "surviving_a": surv_a,
        "surviving_b": surv_b,
        "intersection": inter,
        "union": uni,
        "jaccard": jaccard,
    }


def compute_overall_jaccard(
    layer_stats_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute overall Jaccard across layers: sum(intersections) / sum(unions).

    Never averages layerwise Jaccard scores.
    """
    tot_surv_a = sum(s["surviving_a"] for s in layer_stats_list)
    tot_surv_b = sum(s["surviving_b"] for s in layer_stats_list)
    tot_inter = sum(s["intersection"] for s in layer_stats_list)
    tot_union = sum(s["union"] for s in layer_stats_list)

    overall_jaccard = (
        float(tot_inter / tot_union) if tot_union > 0 else float("nan")
    )
    return {
        "surviving_a": tot_surv_a,
        "surviving_b": tot_surv_b,
        "intersection": tot_inter,
        "union": tot_union,
        "jaccard": overall_jaccard,
    }


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_masks(
    ckpt_path: Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Read-only load checkpoint using weights_only=True and extract masks."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "masks" not in ckpt:
        raise KeyError(f"'masks' key missing in checkpoint: {ckpt_path}")
    raw_masks = ckpt["masks"]

    # Filter only expected connection weights (no biases)
    extracted = {}
    for layer_name in EXPECTED_LAYERS:
        if layer_name not in raw_masks:
            raise KeyError(f"Expected layer {layer_name} not found in masks: {ckpt_path}")
        m = raw_masks[layer_name]
        if m.shape != EXPECTED_SHAPES[layer_name]:
            raise ValueError(
                f"Shape mismatch for {layer_name} in {ckpt_path}: {m.shape} vs {EXPECTED_SHAPES[layer_name]}"
            )
        # Verify binary / boolean values
        if m.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int32, torch.int64, torch.float32):
            raise TypeError(f"Unsupported dtype {m.dtype} for mask in {ckpt_path}")
        unique_vals = torch.unique(m)
        for u in unique_vals:
            if u.item() not in (0, 1, False, True):
                raise ValueError(f"Non-binary value {u.item()} found in mask {layer_name} in {ckpt_path}")

        extracted[layer_name] = m.detach().clone().bool().cpu()

    metrics = ckpt.get("metrics", {})
    return extracted, metrics


def validate_run_configuration(
    run_name: str,
    runs_dir: Path,
    results_dir: Optional[Path],
    dataset: str,
    seed: int,
) -> Tuple[bool, str]:
    """Verify that configuration exists and matches expected formal experiment settings."""
    cfg_path = runs_dir / run_name / "config.json"
    if not cfg_path.is_file():
        return False, f"Missing runs config: {cfg_path}"

    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"Failed to parse runs config.json: {e}"

    if cfg.get("dataset") != dataset:
        return False, f"Config dataset mismatch: {cfg.get('dataset')} != {dataset}"
    if cfg.get("seed") != seed:
        return False, f"Config seed mismatch: {cfg.get('seed')} != {seed}"

    for k, expected_v in EXPECTED_COMMON_CONFIG.items():
        if k not in cfg:
            return False, f"Missing key '{k}' in runs config: {cfg_path}"
        actual_v = cfg[k]
        if isinstance(expected_v, float):
            if abs(actual_v - expected_v) > 1e-6:
                return False, f"Config mismatch for '{k}': {actual_v} != {expected_v}"
        else:
            if actual_v != expected_v:
                return False, f"Config mismatch for '{k}': {actual_v} != {expected_v}"

    # Also verify results_dir config if provided
    if results_dir is not None:
        res_cfg_path = results_dir / run_name / "config.json"
        if not res_cfg_path.is_file():
            return False, f"Missing results config: {res_cfg_path}"
        try:
            res_cfg = json.loads(res_cfg_path.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"Failed to parse results config.json: {e}"

        for k, v in EXPECTED_COMMON_CONFIG.items():
            if k not in res_cfg:
                return False, f"Missing key '{k}' in results config: {res_cfg_path}"
            res_v = res_cfg[k]
            if isinstance(v, float):
                if abs(res_v - v) > 1e-6:
                    return False, f"Results config mismatch for '{k}': {res_v} != {v}"
            else:
                if res_v != v:
                    return False, f"Results config mismatch for '{k}': {res_v} != {v}"

    return True, ""


def load_and_verify_summary(
    summary_path: Path,
    expected_levels: List[int],
) -> Tuple[bool, str, Dict[int, float]]:
    """Load summary.csv and verify required columns, uniqueness, and valid numbers."""
    if not summary_path.is_file():
        return False, f"Missing summary: {summary_path}", {}

    try:
        df = pd.read_csv(summary_path)
    except Exception as e:
        return False, f"Failed to parse summary {summary_path}: {e}", {}

    if "level" not in df.columns or "weights_remaining" not in df.columns:
        return False, f"Missing required columns in {summary_path}", {}

    if df["level"].duplicated().any():
        dups = df["level"][df["level"].duplicated()].tolist()
        return False, f"Duplicate levels found in {summary_path}: {dups}", {}

    summary_map = {}
    for _, row in df.iterrows():
        try:
            lvl = int(row["level"])
            frac = float(row["weights_remaining"])
        except (ValueError, TypeError) as e:
            return False, f"Invalid level or weights_remaining in {summary_path}: {e}", {}

        if not math.isfinite(frac) or frac < 0.0 or frac > 1.0:
            return False, f"Non-finite or out-of-range weights_remaining {frac} at level {lvl} in {summary_path}", {}

        summary_map[lvl] = frac

    return True, "", summary_map


def audit_run_checkpoints(
    runs_dir: Path,
    results_dir: Optional[Path],
    dataset: str,
    seed: int,
    levels: List[int] = EXPECTED_LEVELS,
    model: str = DEFAULT_MODEL,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Audit checkpoints for a single run, checking nesting, shapes, and metrics."""
    run_name = f"{dataset}_adam_seed{seed}"
    run_path = runs_dir / run_name
    inventory_rows = []
    valid_data_by_level = {}

    # 1. Config verification
    config_valid, cfg_err = validate_run_configuration(
        run_name=run_name,
        runs_dir=runs_dir,
        results_dir=results_dir,
        dataset=dataset,
        seed=seed,
    )

    # 2. Runs summary verification
    runs_summary_valid, runs_sum_err, runs_summary_map = load_and_verify_summary(
        summary_path=run_path / "summary.csv",
        expected_levels=levels,
    )

    # 3. Results summary verification (if results_dir provided)
    results_summary_valid = True
    results_sum_err = ""
    results_summary_map = {}
    if results_dir is not None:
        results_summary_valid, results_sum_err, results_summary_map = load_and_verify_summary(
            summary_path=results_dir / run_name / "summary.csv",
            expected_levels=levels,
        )

    prev_masks = None
    for lvl in levels:
        ckpt_path = run_path / f"level_{lvl:02d}.pt"
        inv_item = {
            "dataset": dataset,
            "seed": seed,
            "level": lvl,
            "expected_path": str(ckpt_path),
            "exists": ckpt_path.is_file(),
            "sha256": None,
            "is_valid": False,
            "rejection_reason": "",
            "weights_remaining": None,
        }

        if not ckpt_path.is_file():
            inv_item["rejection_reason"] = "File not found"
            inventory_rows.append(inv_item)
            continue

        inv_item["sha256"] = compute_file_sha256(ckpt_path)

        if not config_valid:
            inv_item["rejection_reason"] = f"Invalid run config: {cfg_err}"
            inventory_rows.append(inv_item)
            continue

        if not runs_summary_valid:
            inv_item["rejection_reason"] = f"Invalid runs summary: {runs_sum_err}"
            inventory_rows.append(inv_item)
            continue

        if not results_summary_valid:
            inv_item["rejection_reason"] = f"Invalid results summary: {results_sum_err}"
            inventory_rows.append(inv_item)
            continue

        try:
            masks, metrics = load_checkpoint_masks(ckpt_path)
        except Exception as e:
            inv_item["rejection_reason"] = f"Failed to load checkpoint: {e}"
            inventory_rows.append(inv_item)
            continue

        # Check metrics dict
        if not isinstance(metrics, dict):
            inv_item["rejection_reason"] = f"metrics in {ckpt_path} is not a dict"
            inventory_rows.append(inv_item)
            continue

        if "level" not in metrics:
            inv_item["rejection_reason"] = f"Missing 'level' in metrics: {ckpt_path}"
            inventory_rows.append(inv_item)
            continue

        try:
            ckpt_level = int(metrics["level"])
        except (ValueError, TypeError):
            inv_item["rejection_reason"] = f"Non-integer level in metrics: {metrics['level']}"
            inventory_rows.append(inv_item)
            continue

        if ckpt_level != lvl:
            inv_item["rejection_reason"] = f"Metric level mismatch: {ckpt_level} != {lvl}"
            inventory_rows.append(inv_item)
            continue

        if "weights_remaining" not in metrics:
            inv_item["rejection_reason"] = f"Missing 'weights_remaining' in metrics: {ckpt_path}"
            inventory_rows.append(inv_item)
            continue

        try:
            metric_frac = float(metrics["weights_remaining"])
        except (ValueError, TypeError):
            inv_item["rejection_reason"] = f"Non-float weights_remaining in metrics: {metrics['weights_remaining']}"
            inventory_rows.append(inv_item)
            continue

        if not math.isfinite(metric_frac) or metric_frac < 0.0 or metric_frac > 1.0:
            inv_item["rejection_reason"] = f"Invalid non-finite or out-of-range weights_remaining in metrics: {metric_frac}"
            inventory_rows.append(inv_item)
            continue

        # Check weights remaining calculated from masks
        tot_surv = sum(int(m.sum().item()) for m in masks.values())
        tot_numel = sum(m.numel() for m in masks.values())
        calc_frac = tot_surv / tot_numel
        inv_item["weights_remaining"] = calc_frac

        # Verify dense baseline at level 0
        if lvl == 0:
            dense_violated = False
            for lyr, m in masks.items():
                if not torch.all(m):
                    dense_violated = True
                    break
            if dense_violated or abs(calc_frac - 1.0) > 1e-7 or abs(metric_frac - 1.0) > 1e-7:
                inv_item["rejection_reason"] = f"Dense baseline violated at level 0: frac={calc_frac}"
                inventory_rows.append(inv_item)
                continue

        if abs(calc_frac - metric_frac) > 1e-6:
            inv_item["rejection_reason"] = (
                f"Remaining fraction mismatch with metrics: {calc_frac} vs {metric_frac}"
            )
            inventory_rows.append(inv_item)
            continue

        # Verify matching with runs summary
        if lvl not in runs_summary_map:
            inv_item["rejection_reason"] = f"Level {lvl} missing in runs summary"
            inventory_rows.append(inv_item)
            continue

        runs_sum_frac = runs_summary_map[lvl]
        if abs(calc_frac - runs_sum_frac) > 1e-6:
            inv_item["rejection_reason"] = (
                f"Remaining fraction mismatch with runs summary: {calc_frac} vs {runs_sum_frac}"
            )
            inventory_rows.append(inv_item)
            continue

        # Verify matching with results summary
        if results_dir is not None:
            if lvl not in results_summary_map:
                inv_item["rejection_reason"] = f"Level {lvl} missing in results summary"
                inventory_rows.append(inv_item)
                continue
            res_sum_frac = results_summary_map[lvl]
            if abs(calc_frac - res_sum_frac) > 1e-6:
                inv_item["rejection_reason"] = (
                    f"Remaining fraction mismatch with results summary: {calc_frac} vs {res_sum_frac}"
                )
                inventory_rows.append(inv_item)
                continue

        # Check nesting
        if prev_masks is not None:
            nesting_violated = False
            for lyr in EXPECTED_LAYERS:
                if not torch.all(masks[lyr] <= prev_masks[lyr]):
                    nesting_violated = True
                    break
            if nesting_violated:
                inv_item["rejection_reason"] = "Mask nesting violated: pruned weights re-appeared"
                inventory_rows.append(inv_item)
                continue

        prev_masks = masks
        inv_item["is_valid"] = True
        inventory_rows.append(inv_item)
        valid_data_by_level[lvl] = {
            "masks": masks,
            "metrics": metrics,
            "weights_remaining": calc_frac,
            "sha256": inv_item["sha256"],
            "path": ckpt_path,
        }

    return inventory_rows, valid_data_by_level


def audit_all_runs(
    runs_dir: Path,
    results_dir: Optional[Path] = None,
    datasets: List[str] = EXPECTED_DATASETS,
    seeds: List[int] = EXPECTED_SEEDS,
    levels: List[int] = EXPECTED_LEVELS,
    model: str = DEFAULT_MODEL,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, int, int], Dict[str, Any]]]:
    """Audit all expected runs and build an inventory DataFrame and mask store."""
    all_inventory = []
    store = {}  # (dataset, seed, level) -> valid_data_by_level[lvl]

    for ds in datasets:
        for seed in seeds:
            inv_rows, valid_lvls = audit_run_checkpoints(
                runs_dir, results_dir, ds, seed, levels, model
            )
            all_inventory.extend(inv_rows)
            for lvl, data in valid_lvls.items():
                store[(ds, seed, lvl)] = data

    df_inventory = pd.DataFrame(all_inventory)
    return df_inventory, store


def compare_seed_pair_masks(
    dataset: str,
    model: str,
    level: int,
    weights_remaining: float,
    seed_a: int,
    seed_b: int,
    masks_a: Dict[str, torch.Tensor],
    masks_b: Dict[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    """Compare masks of two seeds at the same (dataset, model, level).

    Generates records for 'overall' and each individual layer.
    """
    if seed_a == seed_b:
        raise ValueError(f"Self-comparison not allowed: seed {seed_a} == seed {seed_b}")

    layer_stats = []
    layer_records = []

    for layer_name in EXPECTED_LAYERS:
        stats = compute_layer_jaccard(masks_a[layer_name], masks_b[layer_name])
        layer_stats.append(stats)
        rec = {
            "dataset": dataset,
            "model": model,
            "level": level,
            "weights_remaining": weights_remaining,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "scope": layer_name,
            **stats,
        }
        layer_records.append(rec)

    overall_stats = compute_overall_jaccard(layer_stats)
    overall_record = {
        "dataset": dataset,
        "model": model,
        "level": level,
        "weights_remaining": weights_remaining,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "scope": "overall",
        **overall_stats,
    }

    return [overall_record] + layer_records


def compute_all_pairs(
    store: Dict[Tuple[str, int, int], Dict[str, Any]],
    datasets: List[str] = EXPECTED_DATASETS,
    seeds: List[int] = EXPECTED_SEEDS,
    levels: List[int] = EXPECTED_LEVELS,
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Compute pairwise Jaccard across seed pairs (1,2), (1,42), (2,42)."""
    unique_seeds = sorted(set(seeds))
    seed_pairs = [
        (unique_seeds[i], unique_seeds[j])
        for i in range(len(unique_seeds))
        for j in range(i + 1, len(unique_seeds))
    ]

    pair_rows = []

    for ds in datasets:
        for lvl in levels:
            for s_a, s_b in seed_pairs:
                key_a = (ds, s_a, lvl)
                key_b = (ds, s_b, lvl)
                if key_a not in store or key_b not in store:
                    continue  # Missing checkpoint or invalid, skip pair

                data_a = store[key_a]
                data_b = store[key_b]

                # Check layerwise retained counts match across seeds
                for lyr in EXPECTED_LAYERS:
                    cnt_a = int(data_a["masks"][lyr].sum().item())
                    cnt_b = int(data_b["masks"][lyr].sum().item())
                    if cnt_a != cnt_b:
                        raise ValueError(
                            f"Retained count mismatch between seed {s_a} and {s_b} "
                            f"for {ds} lvl {lvl} layer {lyr}: {cnt_a} != {cnt_b}"
                        )

                frac = data_a["weights_remaining"]
                records = compare_seed_pair_masks(
                    dataset=ds,
                    model=model,
                    level=lvl,
                    weights_remaining=frac,
                    seed_a=s_a,
                    seed_b=s_b,
                    masks_a=data_a["masks"],
                    masks_b=data_b["masks"],
                )
                pair_rows.extend(records)

    columns = [
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
    df_pairs = pd.DataFrame(pair_rows, columns=columns)
    return df_pairs


def aggregate_jaccard_summary(
    df_pairs: pd.DataFrame,
    expected_pairs: int = 3,
) -> pd.DataFrame:
    """Aggregate pairwise results by (dataset, model, level, weights_remaining, scope).

    Computes mean, sample SD (ddof=1), min, max, n_pairs, expected_pairs.
    When n_pairs <= 1, SD is NaN.
    """
    if df_pairs.empty:
        return pd.DataFrame(
            columns=[
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
                "expected_pairs",
            ]
        )

    grouped = df_pairs.groupby(
        ["dataset", "model", "level", "weights_remaining", "scope"],
        as_index=False,
    )

    summary_rows = []
    for (ds, mdl, lvl, frac, scp), group in grouped:
        valid_jaccards = group["jaccard"].dropna()
        n = len(valid_jaccards)
        if n == 0:
            m = float("nan")
            s = float("nan")
            min_val = float("nan")
            max_val = float("nan")
        elif n == 1:
            m = float(valid_jaccards.iloc[0])
            s = float("nan")
            min_val = m
            max_val = m
        else:
            m = float(valid_jaccards.mean())
            s = float(valid_jaccards.std(ddof=1))
            min_val = float(valid_jaccards.min())
            max_val = float(valid_jaccards.max())

        summary_rows.append(
            {
                "dataset": ds,
                "model": mdl,
                "level": lvl,
                "weights_remaining": frac,
                "scope": scp,
                "mean": m,
                "std": s,
                "min": min_val,
                "max": max_val,
                "n_pairs": n,
                "expected_pairs": expected_pairs,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    scope_order = {"overall": 0, "fc1.weight": 1, "fc2.weight": 2, "fc3.weight": 3}
    df_summary["_order"] = df_summary["scope"].map(lambda x: scope_order.get(x, 99))
    df_summary = (
        df_summary.sort_values(by=["dataset", "_order", "level"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
    return df_summary


def get_git_info() -> Dict[str, Any]:
    """Retrieve git branch, commit hash, and dirty status."""
    try:
        rev = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        rev = "unknown"

    try:
        branch = (
            subprocess.check_output(
                ["git", "branch", "--show-current"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        branch = "unknown"

    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        is_dirty = len(status) > 0
    except Exception:
        is_dirty = True

    return {
        "commit": rev,
        "branch": branch,
        "is_dirty": is_dirty,
    }


def verify_execution_artifacts(
    execution_dir: Path,
    runs_dir: Path = Path("runs"),
    results_dir: Optional[Path] = Path("results"),
) -> Tuple[bool, List[str]]:
    """Verify integrity of an existing execution directory against current inputs.

    Returns (is_valid, list_of_error_strings).
    """
    errors = []
    if not execution_dir.is_dir():
        return False, [f"Execution directory not found: {execution_dir}"]

    meta_path = execution_dir / "metadata.json"
    if not meta_path.is_file():
        return False, [f"metadata.json missing in {execution_dir}"]

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, [f"Failed to parse metadata.json in {execution_dir}: {e}"]

    # Required artifact files
    req_files = [
        execution_dir / "artifact_inventory.csv",
        execution_dir / "tables" / "mask_jaccard_pairs.csv",
        execution_dir / "tables" / "mask_jaccard_summary.csv",
        execution_dir / "figures" / "mask_stability_grid.png",
        execution_dir / "figures" / "mask_stability_grid.pdf",
    ]
    for rf in req_files:
        if not rf.is_file():
            errors.append(f"Missing required artifact: {rf}")

    # Check input file SHA-256 integrity
    recorded_inputs = meta.get("inputs", {})
    if not recorded_inputs:
        errors.append("No inputs recorded in metadata.json")
    else:
        for file_path_str, info in recorded_inputs.items():
            expected_sha = info.get("sha256")
            if not expected_sha:
                continue
            p = Path(file_path_str)
            if not p.is_file():
                errors.append(f"Recorded input file no longer exists: {p}")
                continue
            actual_sha = compute_file_sha256(p)
            if actual_sha != expected_sha:
                errors.append(
                    f"SHA-256 hash mismatch for {p}: current={actual_sha} recorded={expected_sha}"
                )

    return (len(errors) == 0), errors


def run_extension1_pipeline(
    runs_dir: str = "runs",
    results_dir: str = "results",
    execution_id: Optional[str] = None,
    out_base: str = "results/extensions",
) -> Dict[str, Any]:
    """Execute complete Extension 1 analysis pipeline."""
    runs_path = Path(runs_dir)
    results_path = Path(results_dir) if results_dir else None

    # Generate unique execution ID if not provided
    if not execution_id:
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        execution_id = f"ext1_{now_str}_{short_id}"

    out_dir = Path(out_base) / execution_id
    if out_dir.exists():
        raise FileExistsError(
            f"Execution directory already exists: {out_dir}. Overwriting an existing execution is forbidden."
        )

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=False)

    # 1. Audit artifacts
    df_inventory, store = audit_all_runs(runs_path, results_path)
    inventory_path = out_dir / "artifact_inventory.csv"
    df_inventory.to_csv(inventory_path, index=False)

    # 2. Compute pairwise Jaccard
    df_pairs = compute_all_pairs(store)
    pairs_path = tables_dir / "mask_jaccard_pairs.csv"
    df_pairs.to_csv(pairs_path, index=False)

    # 3. Aggregate summary
    df_summary = aggregate_jaccard_summary(df_pairs, expected_pairs=3)
    summary_path = tables_dir / "mask_jaccard_summary.csv"
    df_summary.to_csv(summary_path, index=False)

    # 4. Generate plots
    from extension_analysis.plots import plot_mask_stability_grid

    fig_png = figures_dir / "mask_stability_grid.png"
    fig_pdf = figures_dir / "mask_stability_grid.pdf"
    plot_mask_stability_grid(
        df_summary=df_summary,
        df_pairs=df_pairs,
        save_png=fig_png,
        save_pdf=fig_pdf,
    )

    # 5. Metadata JSON
    inputs_meta = {}
    for item in df_inventory.to_dict(orient="records"):
        inputs_meta[item["expected_path"]] = {
            "exists": item["exists"],
            "is_valid": item["is_valid"],
            "sha256": item["sha256"],
            "rejection_reason": item["rejection_reason"],
        }

    # Also record configs and summaries in metadata
    for ds in EXPECTED_DATASETS:
        for seed in EXPECTED_SEEDS:
            run_name = f"{ds}_adam_seed{seed}"
            cfg_p = runs_path / run_name / "config.json"
            if cfg_p.is_file():
                inputs_meta[str(cfg_p)] = {
                    "sha256": compute_file_sha256(cfg_p),
                    "type": "runs_config",
                }
            sum_p = runs_path / run_name / "summary.csv"
            if sum_p.is_file():
                inputs_meta[str(sum_p)] = {
                    "sha256": compute_file_sha256(sum_p),
                    "type": "runs_summary",
                }
            if results_path is not None:
                res_cfg_p = results_path / run_name / "config.json"
                if res_cfg_p.is_file():
                    inputs_meta[str(res_cfg_p)] = {
                        "sha256": compute_file_sha256(res_cfg_p),
                        "type": "results_config",
                    }
                res_sum_p = results_path / run_name / "summary.csv"
                if res_sum_p.is_file():
                    inputs_meta[str(res_sum_p)] = {
                        "sha256": compute_file_sha256(res_sum_p),
                        "type": "results_summary",
                    }

    metadata = {
        "execution_id": execution_id,
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "git": get_git_info(),
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "parameters": {
            "datasets": EXPECTED_DATASETS,
            "seeds": EXPECTED_SEEDS,
            "levels": EXPECTED_LEVELS,
            "model": DEFAULT_MODEL,
            "layers": EXPECTED_LAYERS,
            "expected_common_config": EXPECTED_COMMON_CONFIG,
        },
        "coverage": {
            "total_expected_checkpoints": len(df_inventory),
            "total_valid_checkpoints": int(df_inventory["is_valid"].sum()),
            "total_pairs_computed": len(df_pairs[df_pairs["scope"] == "overall"]),
            "expected_pairs_per_level": 3,
        },
        "outputs": {
            "artifact_inventory": str(inventory_path),
            "mask_jaccard_pairs": str(pairs_path),
            "mask_jaccard_summary": str(summary_path),
            "figures_png": str(fig_png),
            "figures_pdf": str(fig_pdf),
        },
        "inputs": inputs_meta,
    }

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "execution_id": execution_id,
        "out_dir": out_dir,
        "df_inventory": df_inventory,
        "df_pairs": df_pairs,
        "df_summary": df_summary,
        "metadata_path": meta_path,
        "png_path": fig_png,
        "pdf_path": fig_pdf,
    }
