"""Extension 3: Cross-seed Prediction Agreement Analysis.

Investigates whether winning tickets found by IMP from different initialization seeds
make similar predictions on the test set, despite retaining different weight coordinates.
"""

import copy
import csv
import hashlib
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.datasets as torchvision_datasets

from lth.models import make_model
from extension_analysis.mask_stability import (
    EXPECTED_COMMON_CONFIG,
    EXPECTED_DATASETS,
    EXPECTED_LAYERS,
    EXPECTED_LEVELS,
    EXPECTED_SEEDS,
    EXPECTED_SHAPES,
    DEFAULT_MODEL,
    compute_file_sha256,
    validate_run_configuration,
    load_and_verify_summary,
)

SUPPORTED_DATASETS = ["mnist", "fashion_mnist"]
NUM_LEVELS = 16  # Levels 0 to 15
SEED_PAIRS = [(1, 2), (1, 42), (2, 42)]
DEFAULT_BATCH_SIZE = 512
DEFAULT_ACCURACY_TOLERANCE = 1e-4  # Allows at most 1 sample difference on 10,000 samples
PREPROCESSING_VERSION = "v1_div255_flatten"

EXPECTED_BIAS_SHAPES = {
    "fc1.bias": (300,),
    "fc2.bias": (100,),
    "fc3.bias": (10,),
}


class DatasetNotAvailableError(Exception):
    """Raised when a required dataset is not available locally and download=False."""

    def __init__(self, dataset_name: str, path: Path, message: str):
        super().__init__(f"Dataset '{dataset_name}' not available at {path}: {message}")
        self.dataset_name = dataset_name
        self.path = path
        self.message = message


class DatasetCorruptError(Exception):
    """Raised when a dataset is present locally but cannot be parsed or loaded."""

    def __init__(self, dataset_name: str, path: Path, message: str):
        super().__init__(f"Dataset '{dataset_name}' corrupted at {path}: {message}")
        self.dataset_name = dataset_name
        self.path = path
        self.message = message


class IncompatibleCheckpointError(Exception):
    """Raised when a checkpoint fails structure, parameter, or accuracy verification."""

    def __init__(self, ckpt_path: Path, reason: str):
        super().__init__(f"Checkpoint incompatible at {ckpt_path}: {reason}")
        self.ckpt_path = ckpt_path
        self.reason = reason


def compute_dataset_fingerprint(
    x: torch.Tensor,
    y: torch.Tensor,
    preprocessing_version: str = PREPROCESSING_VERSION,
) -> str:
    """Compute chunked SHA-256 fingerprint over complete tensor contents and metadata.

    Includes shape, dtype, preprocessing version string, and entire raw bytes in chunks.
    Ensures that any alteration (even to the final sample/pixel) alters the fingerprint.
    """
    h = hashlib.sha256()
    meta_str = (
        f"{list(x.shape)}|{str(x.dtype)}|{list(y.shape)}|{str(y.dtype)}|{preprocessing_version}\n"
    )
    h.update(meta_str.encode("utf-8"))

    # Chunked update for x (ensures full tensor coverage without truncation)
    x_bytes = memoryview(x.detach().cpu().contiguous().numpy())
    chunk_size = 65536
    for i in range(0, len(x_bytes), chunk_size):
        h.update(x_bytes[i : i + chunk_size])

    # Chunked update for y
    y_bytes = memoryview(y.detach().cpu().contiguous().numpy())
    for i in range(0, len(y_bytes), chunk_size):
        h.update(y_bytes[i : i + chunk_size])

    return h.hexdigest()


def load_test_dataset(
    dataset_name: str,
    data_root: Path = Path("data"),
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Load test dataset with download=False and compute integrity fingerprint.

    Returns:
        (x_test, y_test, data_fingerprint)
        where x_test is shape (N, 1, 28, 28) float32 in [0, 1],
        y_test is shape (N,) int64 targets.

    Raises:
        DatasetNotAvailableError if dataset files are missing locally.
        DatasetCorruptError if dataset files are corrupted or unreadable.
    """
    data_root = Path(data_root)
    if dataset_name == "mnist":
        cls = torchvision_datasets.MNIST
        raw_dir = data_root / "MNIST" / "raw"
    elif dataset_name == "fashion_mnist":
        cls = torchvision_datasets.FashionMNIST
        raw_dir = data_root / "FashionMNIST" / "raw"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if not raw_dir.exists():
        raise DatasetNotAvailableError(
            dataset_name=dataset_name,
            path=raw_dir,
            message="Raw dataset directory does not exist locally (download=False enforced).",
        )

    try:
        ds = cls(root=str(data_root), train=False, download=False)
    except RuntimeError as exc:
        raise DatasetNotAvailableError(
            dataset_name=dataset_name,
            path=raw_dir,
            message=str(exc),
        ) from exc
    except Exception as exc:
        raise DatasetCorruptError(
            dataset_name=dataset_name,
            path=raw_dir,
            message=str(exc),
        ) from exc

    # Tensor conversion matching lth.data loaders:
    # uint8 -> float32 / 255.0, shape (N, 1, 28, 28)
    try:
        x = ds.data.unsqueeze(1).float().div(255.0)
        y = ds.targets.to(dtype=torch.int64)
    except Exception as exc:
        raise DatasetCorruptError(
            dataset_name=dataset_name,
            path=raw_dir,
            message=f"Failed to convert dataset to normalized tensors: {exc}",
        ) from exc

    if x.ndim != 4 or x.shape[1:] != (1, 28, 28):
        raise DatasetCorruptError(
            dataset_name=dataset_name,
            path=raw_dir,
            message=f"Unexpected x tensor shape: {x.shape}",
        )
    if y.ndim != 1 or len(y) != len(x):
        raise DatasetCorruptError(
            dataset_name=dataset_name,
            path=raw_dir,
            message=f"Targets length mismatch: y {y.shape} vs x {x.shape}",
        )

    fingerprint = compute_dataset_fingerprint(x, y)
    return x, y, fingerprint


def verify_checkpoint_and_best_state(
    ckpt_path: Path,
    expected_dataset: str,
    expected_seed: int,
    expected_level: int,
    runs_dir: Path = Path("runs"),
    results_dir: Optional[Path] = Path("results"),
) -> Dict[str, Any]:
    """Strictly audit checkpoint integrity, best_state parameters, masks, and metadata.

    Reuses existing Extension 1 audit standards:
    - Runs and results config.json match EXPECTED_COMMON_CONFIG.
    - Summary.csv exists, valid, and matches weights_remaining.
    - Masks contain expected keys, shapes, binary values, and dense baseline at level 0.
    - best_state contains all connection weights and biases, finite, correct shape.
    - best_state pruned positions are strictly zero.
    """
    if not ckpt_path.is_file():
        raise IncompatibleCheckpointError(ckpt_path, "Checkpoint file does not exist")

    # 1. Config and summary audit
    run_name = f"{expected_dataset}_adam_seed{expected_seed}"
    cfg_ok, cfg_err = validate_run_configuration(
        run_name=run_name,
        runs_dir=runs_dir,
        results_dir=results_dir,
        dataset=expected_dataset,
        seed=expected_seed,
    )
    if not cfg_ok:
        raise IncompatibleCheckpointError(ckpt_path, f"Configuration invalid: {cfg_err}")

    # Check runs summary
    runs_summary_path = runs_dir / f"{expected_dataset}_adam_seed{expected_seed}" / "summary.csv"
    sum_ok, sum_err, sum_map = load_and_verify_summary(runs_summary_path, EXPECTED_LEVELS)
    if not sum_ok:
        raise IncompatibleCheckpointError(ckpt_path, f"Runs summary invalid: {sum_err}")

    # Read and strictly audit results summary if results_dir is provided
    res_summary_acc = None
    if results_dir is not None:
        res_sum_path = results_dir / f"{expected_dataset}_adam_seed{expected_seed}" / "summary.csv"
        if not res_sum_path.is_file():
            raise IncompatibleCheckpointError(ckpt_path, f"Missing results summary file: {res_sum_path}")

        try:
            res_df = pd.read_csv(res_sum_path)
        except Exception as exc:
            raise IncompatibleCheckpointError(ckpt_path, f"Failed to parse results summary {res_sum_path}: {exc}")

        if "level" not in res_df.columns or "best_test_accuracy" not in res_df.columns:
            raise IncompatibleCheckpointError(
                ckpt_path, f"Required columns ('level', 'best_test_accuracy') missing in {res_sum_path}"
            )

        if res_df["level"].duplicated().any():
            dups = res_df["level"][res_df["level"].duplicated()].tolist()
            raise IncompatibleCheckpointError(ckpt_path, f"Duplicate levels found in results summary {res_sum_path}: {dups}")

        match_row = res_df[res_df["level"] == expected_level]
        if match_row.empty:
            raise IncompatibleCheckpointError(ckpt_path, f"Target level {expected_level} not found in results summary {res_sum_path}")
        if len(match_row) > 1:
            raise IncompatibleCheckpointError(ckpt_path, f"Target level {expected_level} duplicated in results summary {res_sum_path}")

        raw_acc = match_row["best_test_accuracy"].iloc[0]
        try:
            res_summary_acc = float(raw_acc)
        except (ValueError, TypeError):
            raise IncompatibleCheckpointError(ckpt_path, f"Invalid best_test_accuracy in results summary {res_sum_path}: {raw_acc}")

        if not math.isfinite(res_summary_acc) or res_summary_acc < 0.0 or res_summary_acc > 1.0:
            raise IncompatibleCheckpointError(
                ckpt_path, f"Out-of-range or non-finite best_test_accuracy in results summary {res_sum_path}: {res_summary_acc}"
            )

    # 2. Checkpoint dictionary load
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise IncompatibleCheckpointError(ckpt_path, f"Failed to load checkpoint file: {exc}")

    for req_key in ["best_state", "masks", "metrics"]:
        if req_key not in ckpt:
            raise IncompatibleCheckpointError(ckpt_path, f"Missing required checkpoint key: '{req_key}'")

    best_state = ckpt["best_state"]
    masks = ckpt["masks"]
    metrics = ckpt["metrics"]

    # 3. Verify metrics
    level = metrics.get("level")
    if level != expected_level:
        raise IncompatibleCheckpointError(ckpt_path, f"Level mismatch: {level} != {expected_level}")

    weights_remaining = metrics.get("weights_remaining")
    if weights_remaining is None or not math.isfinite(weights_remaining):
        raise IncompatibleCheckpointError(ckpt_path, "weights_remaining missing or non-finite")

    if expected_level in sum_map and abs(weights_remaining - sum_map[expected_level]) > 1e-6:
        raise IncompatibleCheckpointError(
            ckpt_path,
            f"weights_remaining mismatch with summary: ckpt {weights_remaining} vs summary {sum_map[expected_level]}",
        )

    expected_acc = metrics.get("best_test_accuracy")
    if expected_acc is None or not math.isfinite(expected_acc):
        raise IncompatibleCheckpointError(ckpt_path, "best_test_accuracy missing or non-finite")

    if res_summary_acc is not None and abs(expected_acc - res_summary_acc) > 1e-6:
        raise IncompatibleCheckpointError(
            ckpt_path,
            f"Checkpoint best_test_accuracy ({expected_acc}) differs from results summary ({res_summary_acc})",
        )

    # 4. Verify masks
    for layer in EXPECTED_LAYERS:
        if layer not in masks:
            raise IncompatibleCheckpointError(ckpt_path, f"Mask key missing: {layer}")
        m = masks[layer]
        if tuple(m.shape) != EXPECTED_SHAPES[layer]:
            raise IncompatibleCheckpointError(
                ckpt_path, f"Mask shape mismatch for {layer}: {m.shape} vs {EXPECTED_SHAPES[layer]}"
            )
        u_vals = torch.unique(m)
        for u in u_vals:
            if u.item() not in (0, 1, False, True):
                raise IncompatibleCheckpointError(ckpt_path, f"Non-binary value in mask {layer}: {u.item()}")

    # Level 0 dense baseline check
    if expected_level == 0:
        for layer in EXPECTED_LAYERS:
            if not bool(torch.all(masks[layer] == 1)):
                raise IncompatibleCheckpointError(ckpt_path, f"Level 0 mask for {layer} is not all ones")

    # 5. Verify best_state parameters
    all_params_finite = True
    # Check weights
    for layer in EXPECTED_LAYERS:
        if layer not in best_state:
            raise IncompatibleCheckpointError(ckpt_path, f"best_state missing layer {layer}")
        w = best_state[layer]
        if tuple(w.shape) != EXPECTED_SHAPES[layer]:
            raise IncompatibleCheckpointError(
                ckpt_path, f"best_state weight shape mismatch for {layer}: {w.shape} vs {EXPECTED_SHAPES[layer]}"
            )
        if not bool(torch.all(torch.isfinite(w))):
            all_params_finite = False
            raise IncompatibleCheckpointError(ckpt_path, f"Non-finite weights found in best_state {layer}")

        # Check pruned positions are strictly zero
        m = masks[layer].bool()
        zero_mask = ~m
        if zero_mask.any():
            pruned_active = (w[zero_mask] != 0.0).sum().item()
            if pruned_active > 0:
                raise IncompatibleCheckpointError(
                    ckpt_path, f"Pruned weights non-zero in {layer}: {pruned_active} non-zero entries"
                )

    # Check biases
    for bias_name, b_shape in EXPECTED_BIAS_SHAPES.items():
        if bias_name not in best_state:
            raise IncompatibleCheckpointError(ckpt_path, f"best_state missing bias {bias_name}")
        b = best_state[bias_name]
        if tuple(b.shape) != b_shape:
            raise IncompatibleCheckpointError(
                ckpt_path, f"best_state bias shape mismatch for {bias_name}: {b.shape} vs {b_shape}"
            )
        if not bool(torch.all(torch.isfinite(b))):
            all_params_finite = False
            raise IncompatibleCheckpointError(ckpt_path, f"Non-finite bias found in best_state {bias_name}")

    return {
        "best_state": best_state,
        "masks": masks,
        "metrics": metrics,
        "level": level,
        "weights_remaining": float(weights_remaining),
        "checkpoint_accuracy": float(expected_acc),
        "results_summary_accuracy": res_summary_acc,
        "all_params_finite": all_params_finite,
        "pruned_weights_strictly_zero": True,
    }


def evaluate_checkpoint_predictions(
    ckpt_path: Path,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    expected_dataset: str,
    expected_seed: int,
    expected_level: int,
    runs_dir: Path = Path("runs"),
    results_dir: Optional[Path] = Path("results"),
    device: torch.device = torch.device("cpu"),
    batch_size: int = DEFAULT_BATCH_SIZE,
    accuracy_tolerance: float = DEFAULT_ACCURACY_TOLERANCE,
) -> Dict[str, Any]:
    """Run deterministic inference for a verified checkpoint.

    Computes accuracy as integer correct_count / total_samples.
    Enforces tolerance threshold (<= 1 sample difference on 10,000 samples).
    """
    audit_data = verify_checkpoint_and_best_state(
        ckpt_path=ckpt_path,
        expected_dataset=expected_dataset,
        expected_seed=expected_seed,
        expected_level=expected_level,
        runs_dir=runs_dir,
        results_dir=results_dir,
    )

    best_state = audit_data["best_state"]
    expected_acc = audit_data["checkpoint_accuracy"]

    model = make_model(DEFAULT_MODEL)
    model.load_state_dict(best_state, strict=True)
    model.to(device)
    model.eval()

    all_preds = []
    n_samples = len(y_test)

    with torch.inference_mode():
        for i in range(0, n_samples, batch_size):
            batch_x = x_test[i : i + batch_size].to(device)
            logits = model(batch_x)
            if not bool(torch.all(torch.isfinite(logits))):
                raise IncompatibleCheckpointError(ckpt_path, "Model produced non-finite logits")
            batch_preds = logits.argmax(dim=1).cpu()
            all_preds.append(batch_preds)

    preds_tensor = torch.cat(all_preds, dim=0).to(dtype=torch.int64)
    preds_np = preds_tensor.numpy()
    y_test_np = y_test.cpu().numpy()

    # Exact integer correct count calculation
    correct_count = int((preds_np == y_test_np).sum())
    recomputed_acc = float(correct_count / n_samples)
    acc_diff_ckpt = float(abs(recomputed_acc - expected_acc))
    res_summary_acc = audit_data["results_summary_accuracy"]
    acc_diff_summary = float(abs(recomputed_acc - res_summary_acc)) if res_summary_acc is not None else 0.0

    # Allow at most 1 sample divergence (1 / n_samples) with numerical slack against both checkpoint and summary
    allowed_boundary = accuracy_tolerance + 1e-6
    is_within_tol = bool(acc_diff_ckpt <= allowed_boundary and acc_diff_summary <= allowed_boundary)

    if not is_within_tol:
        raise IncompatibleCheckpointError(
            ckpt_path,
            f"Accuracy divergence exceeds 1-sample tolerance: recomputed {recomputed_acc:.6f} "
            f"vs checkpoint {expected_acc:.6f} (diff {acc_diff_ckpt:.2e}), "
            f"vs summary {res_summary_acc} (diff {acc_diff_summary:.2e})",
        )

    return {
        "preds": preds_np,
        "targets": y_test_np,
        "example_ids": np.arange(n_samples, dtype=np.int64),
        "recomputed_accuracy": recomputed_acc,
        "checkpoint_accuracy": expected_acc,
        "results_summary_accuracy": res_summary_acc,
        "accuracy_diff": acc_diff_ckpt,
        "accuracy_diff_checkpoint": acc_diff_ckpt,
        "accuracy_diff_summary": acc_diff_summary,
        "correct_count": correct_count,
        "total_samples": n_samples,
        "is_within_tolerance": is_within_tol,
        "tolerance": accuracy_tolerance,
        "level": audit_data["level"],
        "weights_remaining": audit_data["weights_remaining"],
        "pruned_weights_strictly_zero": audit_data["pruned_weights_strictly_zero"],
        "all_parameters_finite": audit_data["all_params_finite"],
    }


def compute_pair_agreement(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    targets: np.ndarray,
    example_ids: Optional[np.ndarray] = None,
    meta_a: Optional[Dict[str, Any]] = None,
    meta_b: Optional[Dict[str, Any]] = None,
    num_classes: int = 10,
) -> Dict[str, Any]:
    """Compute pairwise prediction agreement and exact accuracy decomposition.

    Strict validation:
    - Rejects non-1D shapes (eliminating (N,1) vs (N,) broadcasting risk).
    - Requires non-empty arrays with equal lengths.
    - Validates class labels in [0, num_classes - 1].
    - Validates matching targets, IDs, fingerprints, and conditions.
    - Validates identity: agreement == shared_correct + shared_wrong.
    """
    # 1. 1D shape check
    for name, arr in [("preds_a", preds_a), ("preds_b", preds_b), ("targets", targets)]:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray, got {type(arr)}")
        if arr.ndim != 1:
            raise ValueError(f"{name} must be strictly 1-dimensional, got shape {arr.shape}")
        if len(arr) == 0:
            raise ValueError(f"{name} must be non-empty")

    if example_ids is not None:
        if not isinstance(example_ids, np.ndarray) or example_ids.ndim != 1:
            raise ValueError(f"example_ids must be a 1D numpy array, got {getattr(example_ids, 'shape', None)}")
        if len(example_ids) != len(targets):
            raise ValueError(f"example_ids length {len(example_ids)} != targets length {len(targets)}")

    n = len(targets)
    if len(preds_a) != n or len(preds_b) != n:
        raise ValueError(f"Length mismatch: preds_a={len(preds_a)}, preds_b={len(preds_b)}, targets={n}")

    # 2. Metadata consistency check
    if meta_a is not None and meta_b is not None:
        for k in ["dataset", "model", "level"]:
            if meta_a.get(k) != meta_b.get(k):
                raise ValueError(f"Metadata mismatch on '{k}': {meta_a.get(k)} != {meta_b.get(k)}")
        if meta_a.get("checkpoint_kind") != "best_state" or meta_b.get("checkpoint_kind") != "best_state":
            raise ValueError("Both models must have checkpoint_kind='best_state'")
        if meta_a.get("data_fingerprint") != meta_b.get("data_fingerprint"):
            raise ValueError("Data fingerprint mismatch between models in pair")
        if "targets" in meta_a and not np.array_equal(meta_a["targets"], targets):
            raise ValueError("meta_a targets do not match passed targets")
        if "targets" in meta_b and not np.array_equal(meta_b["targets"], targets):
            raise ValueError("meta_b targets do not match passed targets")
        if "example_ids" in meta_a and "example_ids" in meta_b:
            if not np.array_equal(meta_a["example_ids"], meta_b["example_ids"]):
                raise ValueError("example_ids mismatch between meta_a and meta_b")

    # 3. Valid label ranges
    if bool(((preds_a < 0) | (preds_a >= num_classes)).any()):
        raise ValueError("preds_a contains out-of-range class labels")
    if bool(((preds_b < 0) | (preds_b >= num_classes)).any()):
        raise ValueError("preds_b contains out-of-range class labels")
    if bool(((targets < 0) | (targets >= num_classes)).any()):
        raise ValueError("targets contains out-of-range class labels")

    # 4. Exact rate calculations using integer counts
    agree_mask = preds_a == preds_b
    corr_a = preds_a == targets
    corr_b = preds_b == targets

    shared_correct_mask = agree_mask & corr_a
    shared_wrong_mask = agree_mask & (~corr_a)

    agreement = float(agree_mask.sum()) / n
    shared_correct = float(shared_correct_mask.sum()) / n
    shared_wrong = float(shared_wrong_mask.sum()) / n
    disagreement = float((~agree_mask).sum()) / n
    accuracy_a = float(corr_a.sum()) / n
    accuracy_b = float(corr_b.sum()) / n

    # 5. Verify decomposition identity
    diff = abs(agreement - (shared_correct + shared_wrong))
    if diff > 1e-12:
        raise ValueError(f"Decomposition failure: agreement - (sc + sw) = {diff}")

    return {
        "agreement": agreement,
        "shared_correct": shared_correct,
        "shared_wrong": shared_wrong,
        "disagreement": disagreement,
        "accuracy_a": accuracy_a,
        "accuracy_b": accuracy_b,
        "n_samples": n,
    }


def aggregate_agreement_summary(
    df_pairs: pd.DataFrame,
    df_accuracy: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate agreement metrics across seed pairs for each dataset and level.

    Statistical standards:
    - When n_pairs <= 1, pair sample standard deviations are NaN (not 0).
    - Accuracy is aggregated directly over unique evaluated models from df_accuracy,
      avoiding duplicate pair unfolding or unequal seed weighting under partial coverage.
    """
    records = []
    # Identify unique datasets and levels from df_accuracy
    combos = df_accuracy[["dataset", "model", "level", "weights_remaining"]].drop_duplicates()

    for _, row in combos.iterrows():
        ds = row["dataset"]
        model = row["model"]
        lvl = int(row["level"])
        wr = float(row["weights_remaining"])

        # 1. Unique model accuracy aggregation
        valid_models = df_accuracy[
            (df_accuracy["dataset"] == ds)
            & (df_accuracy["level"] == lvl)
            & (df_accuracy["audit_status"] == "evaluated")
        ]
        n_models = len(valid_models)
        if n_models > 0:
            acc_vals = valid_models["recomputed_accuracy"].to_numpy()
            acc_mean = float(np.mean(acc_vals))
            acc_sd = float(np.std(acc_vals, ddof=1)) if n_models > 1 else float("nan")
        else:
            acc_mean = float("nan")
            acc_sd = float("nan")

        # 2. Pair agreement aggregation
        sub_pairs = df_pairs[(df_pairs["dataset"] == ds) & (df_pairs["level"] == lvl)]
        n_pairs = len(sub_pairs)

        if n_pairs > 0:
            agrs = sub_pairs["agreement"].to_numpy()
            scs = sub_pairs["shared_correct"].to_numpy()
            sws = sub_pairs["shared_wrong"].to_numpy()

            agr_mean = float(np.mean(agrs))
            agr_sd = float(np.std(agrs, ddof=1)) if n_pairs > 1 else float("nan")
            agr_min = float(np.min(agrs))
            agr_max = float(np.max(agrs))

            sc_mean = float(np.mean(scs))
            sc_sd = float(np.std(scs, ddof=1)) if n_pairs > 1 else float("nan")

            sw_mean = float(np.mean(sws))
            sw_sd = float(np.std(sws, ddof=1)) if n_pairs > 1 else float("nan")
        else:
            agr_mean = float("nan")
            agr_sd = float("nan")
            agr_min = float("nan")
            agr_max = float("nan")
            sc_mean = float("nan")
            sc_sd = float("nan")
            sw_mean = float("nan")
            sw_sd = float("nan")

        records.append(
            {
                "dataset": ds,
                "model": model,
                "level": lvl,
                "weights_remaining": wr,
                "n_models": n_models,
                "accuracy_mean": acc_mean,
                "accuracy_sd": acc_sd,
                "n_pairs": n_pairs,
                "agreement_mean": agr_mean,
                "agreement_sd": agr_sd,
                "agreement_min": agr_min,
                "agreement_max": agr_max,
                "shared_correct_mean": sc_mean,
                "shared_correct_sd": sc_sd,
                "shared_wrong_mean": sw_mean,
                "shared_wrong_sd": sw_sd,
            }
        )

    df_summary = pd.DataFrame(records)
    return df_summary.sort_values(by=["dataset", "level"]).reset_index(drop=True)


def link_structure_and_function(
    df_agreement_pairs: pd.DataFrame,
    ext1_pairs_csv: Path,
) -> pd.DataFrame:
    """Join pairwise prediction agreement with Extension 1 overall mask Jaccard.

    Enforces strict one-to-one validation with key uniqueness checks.
    Rejects missing matches or duplicate inflation.
    """
    if df_agreement_pairs.empty:
        return pd.DataFrame()

    if not ext1_pairs_csv.is_file():
        raise FileNotFoundError(f"Extension 1 pairs file missing: {ext1_pairs_csv}")

    ext1_df = pd.read_csv(ext1_pairs_csv)
    ext1_overall = ext1_df[ext1_df["scope"] == "overall"][
        ["dataset", "model", "level", "weights_remaining", "seed_a", "seed_b", "jaccard"]
    ].rename(columns={"jaccard": "mask_jaccard"})

    key_cols = ["dataset", "model", "level", "seed_a", "seed_b"]

    # Verify key uniqueness before merging
    if ext1_overall.duplicated(subset=key_cols).any():
        dups = ext1_overall[ext1_overall.duplicated(subset=key_cols, keep=False)]
        raise ValueError(f"Duplicate keys found in Extension 1 pairs:\n{dups}")

    if df_agreement_pairs.duplicated(subset=key_cols).any():
        dups = df_agreement_pairs[df_agreement_pairs.duplicated(subset=key_cols, keep=False)]
        raise ValueError(f"Duplicate keys found in Extension 3 agreement pairs:\n{dups}")

    merged = pd.merge(
        df_agreement_pairs,
        ext1_overall,
        on=key_cols,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ext1"),
    )

    if "weights_remaining_ext1" in merged.columns:
        wr_diff = (merged["weights_remaining"] - merged["weights_remaining_ext1"]).abs().max()
        if wr_diff > 1e-6:
            raise ValueError(f"weights_remaining mismatch between Extension 1 and Extension 3: max diff {wr_diff}")
        merged = merged.drop(columns=["weights_remaining_ext1"])

    if len(merged) != len(df_agreement_pairs):
        raise ValueError(
            f"Structure-function join dropped pairs: expected {len(df_agreement_pairs)}, got {len(merged)}"
        )

    cols = [
        "dataset",
        "model",
        "level",
        "weights_remaining",
        "seed_a",
        "seed_b",
        "mask_jaccard",
        "agreement",
        "shared_correct",
        "shared_wrong",
        "accuracy_a",
        "accuracy_b",
    ]
    return merged[cols].sort_values(by=["dataset", "level", "seed_a", "seed_b"]).reset_index(drop=True)


def verify_ext3_artifacts(
    ext3_dir: Path,
    expected_ext1_dir: Optional[Path] = None,
    runs_dir: Path = Path("runs"),
    results_dir: Optional[Path] = Path("results"),
    data_root: Path = Path("data"),
) -> Tuple[bool, Dict[str, List[str]]]:
    """Verify integrity of Extension 3 execution directory, inputs, caches, and outputs.

    Returns:
        (is_valid, errors_by_category)
        where errors_by_category has keys: 'inputs', 'code', 'outputs'
    """
    errors: Dict[str, List[str]] = {"inputs": [], "code": [], "outputs": []}
    ext3_dir = Path(ext3_dir)
    metadata_path = ext3_dir / "metadata.json"

    if not metadata_path.is_file():
        errors["outputs"].append(f"Missing metadata.json in {ext3_dir}")
        return False, errors

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        errors["outputs"].append(f"Unreadable metadata.json: {exc}")
        return False, errors

    # Check input checkpoints
    ckpt_hashes = metadata.get("input_checkpoint_hashes", {})
    for ckpt_rel_path, expected_hash in ckpt_hashes.items():
        full_ckpt_path = Path(ckpt_rel_path)
        if not full_ckpt_path.is_file():
            errors["inputs"].append(f"Checkpoint file missing: {full_ckpt_path}")
        else:
            curr_hash = compute_file_sha256(full_ckpt_path)
            if curr_hash != expected_hash:
                errors["inputs"].append(
                    f"Checkpoint hash mismatch for {ckpt_rel_path}: current {curr_hash} vs recorded {expected_hash}"
                )

    # Check test dataset fingerprint for available datasets
    ds_fingerprints = metadata.get("test_dataset_fingerprints", {})
    for ds_name, expected_fp in ds_fingerprints.items():
        try:
            x, y, curr_fp = load_test_dataset(ds_name, data_root=data_root)
            if curr_fp != expected_fp:
                errors["inputs"].append(
                    f"Dataset fingerprint mismatch for {ds_name}: current {curr_fp} vs recorded {expected_fp}"
                )
        except DatasetNotAvailableError:
            errors["inputs"].append(f"Dataset {ds_name} was recorded as available but is now missing.")
        except Exception as exc:
            errors["inputs"].append(f"Dataset {ds_name} error: {exc}")

    # Check Extension 1 source artifacts
    ext1_meta = metadata.get("extension1_source", {})
    recorded_ext1_dir = ext1_meta.get("path")
    if expected_ext1_dir and recorded_ext1_dir != str(expected_ext1_dir):
        errors["inputs"].append(
            f"Extension 1 directory mismatch: expected {expected_ext1_dir} but recorded {recorded_ext1_dir}"
        )
    for rel_file, exp_hash in ext1_meta.get("table_hashes", {}).items():
        p = Path(recorded_ext1_dir) / rel_file if recorded_ext1_dir else None
        if not p or not p.is_file():
            errors["inputs"].append(f"Missing referenced Extension 1 table: {p}")
        else:
            curr_h = compute_file_sha256(p)
            if curr_h != exp_hash:
                errors["inputs"].append(f"Extension 1 table altered: {p}")

    # Check config and summary hashes
    for cfg_file, exp_hash in metadata.get("config_hashes", {}).items():
        p = Path(cfg_file)
        if not p.is_file() or compute_file_sha256(p) != exp_hash:
            errors["inputs"].append(f"Run config altered or missing: {cfg_file}")

    for sum_file, exp_hash in metadata.get("summary_hashes", {}).items():
        p = Path(sum_file)
        if not p.is_file() or compute_file_sha256(p) != exp_hash:
            errors["inputs"].append(f"Run summary altered or missing: {sum_file}")

    # Check code hashes
    for code_file, exp_hash in metadata.get("code_file_hashes", {}).items():
        p = Path(code_file)
        if p.is_file():
            curr_h = compute_file_sha256(p)
            if curr_h != exp_hash:
                errors["code"].append(f"Code file changed: {code_file}")
        else:
            errors["code"].append(f"Code file missing: {code_file}")

    # Check prediction caches
    pred_hashes = metadata.get("prediction_cache_hashes", {})
    for cache_rel, exp_hash in pred_hashes.items():
        p = ext3_dir / cache_rel
        if not p.is_file():
            errors["outputs"].append(f"Prediction cache missing: {cache_rel}")
        else:
            curr_h = compute_file_sha256(p)
            if curr_h != exp_hash:
                errors["outputs"].append(f"Prediction cache hash mismatch: {cache_rel}")

    # Check output tables and figures
    for tbl_rel, exp_hash in metadata.get("output_table_hashes", {}).items():
        p = ext3_dir / tbl_rel
        if not p.is_file():
            errors["outputs"].append(f"Output table missing: {tbl_rel}")
        else:
            curr_h = compute_file_sha256(p)
            if curr_h != exp_hash:
                errors["outputs"].append(f"Output table hash mismatch: {tbl_rel}")

    for fig_rel, exp_hash in metadata.get("output_figure_hashes", {}).items():
        p = ext3_dir / fig_rel
        if not p.is_file():
            errors["outputs"].append(f"Output figure missing: {fig_rel}")
        else:
            curr_h = compute_file_sha256(p)
            if curr_h != exp_hash:
                errors["outputs"].append(f"Output figure hash mismatch: {fig_rel}")

    is_valid = not (errors["inputs"] or errors["code"] or errors["outputs"])
    return is_valid, errors


def run_extension3_pipeline(
    ext1_dir: Path,
    runs_dir: Path = Path("runs"),
    results_dir: Path = Path("results"),
    data_root: Path = Path("data"),
    output_parent: Optional[Path] = None,
    execution_id: Optional[str] = None,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
    accuracy_tolerance: float = DEFAULT_ACCURACY_TOLERANCE,
) -> Path:
    """Execute complete Extension 3 analysis and produce immutable artifacts.

    Refuses to overwrite existing execution directories.
    Handles missing test datasets gracefully by recording blocked status.
    """
    ext1_dir = Path(ext1_dir)
    runs_dir = Path(runs_dir)
    results_dir = Path(results_dir)
    data_root = Path(data_root)

    # 1. Verify Extension 1 directory, tables, metadata, and all inputs
    from extension_analysis.mask_stability import verify_execution_artifacts

    ext1_valid, ext1_errors = verify_execution_artifacts(
        execution_dir=ext1_dir,
        runs_dir=runs_dir,
        results_dir=results_dir,
    )
    if not ext1_valid:
        raise ValueError(
            f"Extension 1 source execution verification failed for {ext1_dir}:\n"
            + "\n".join(ext1_errors)
        )

    ext1_pairs_csv = ext1_dir / "tables" / "mask_jaccard_pairs.csv"
    ext1_summary_csv = ext1_dir / "tables" / "mask_jaccard_summary.csv"
    ext1_inv_csv = ext1_dir / "artifact_inventory.csv"
    if not ext1_inv_csv.is_file():
        ext1_inv_csv = ext1_dir / "tables" / "checkpoint_inventory.csv"
    ext1_metadata = ext1_dir / "metadata.json"

    for required in [ext1_pairs_csv, ext1_summary_csv, ext1_inv_csv, ext1_metadata]:
        if not required.is_file():
            raise FileNotFoundError(f"Required Extension 1 file missing: {required}")

    df_ext1_inv = pd.read_csv(ext1_inv_csv)
    ext1_ckpt_map = {}
    for _, r in df_ext1_inv.iterrows():
        try:
            k = (str(r["dataset"]), int(r["seed"]), int(r["level"]))
            if pd.notnull(r.get("sha256")):
                ext1_ckpt_map[k] = str(r["sha256"])
        except (ValueError, TypeError):
            continue

    # Build unique execution directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if execution_id is None:
        execution_id = f"ext3_{timestamp}_{hashlib.sha256(timestamp.encode()).hexdigest()[:6]}"
    parent = output_parent or (results_dir / "extensions")
    out_dir = parent / execution_id

    if out_dir.exists():
        raise FileExistsError(f"Execution directory already exists: {out_dir}")

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    predictions_dir = out_dir / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=False)
    figures_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir.mkdir(parents=True, exist_ok=False)

    # 2. Dataset loading check & fingerprinting
    loaded_datasets: Dict[str, Tuple[torch.Tensor, torch.Tensor, str]] = {}
    dataset_status: Dict[str, Dict[str, Any]] = {}

    for ds_name in SUPPORTED_DATASETS:
        try:
            x, y, fp = load_test_dataset(ds_name, data_root=data_root)
            loaded_datasets[ds_name] = (x, y, fp)
            dataset_status[ds_name] = {
                "available": True,
                "n_samples": len(y),
                "fingerprint": fp,
                "status": "ready",
            }
        except DatasetNotAvailableError as exc:
            dataset_status[ds_name] = {
                "available": False,
                "n_samples": 0,
                "fingerprint": None,
                "status": f"blocked: {exc.message}",
            }
        except DatasetCorruptError as exc:
            dataset_status[ds_name] = {
                "available": False,
                "n_samples": 0,
                "fingerprint": None,
                "status": f"corrupt: {exc.message}",
            }

    # 3. Checkpoint inventory, verification, and prediction inference
    inventory_records = []
    accuracy_records = []
    checkpoint_hashes = {}
    cached_predictions: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    prediction_cache_hashes = {}
    dev = torch.device(device)

    # Software versions for metadata and npz headers
    software_versions_dict = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
    }
    software_versions_str = json.dumps(software_versions_dict)

    for ds_name in SUPPORTED_DATASETS:
        is_ds_avail = dataset_status[ds_name]["available"]
        if is_ds_avail:
            x_test, y_test, ds_fp = loaded_datasets[ds_name]

        for seed in EXPECTED_SEEDS:
            run_name = f"{ds_name}_adam_seed{seed}"
            run_folder = runs_dir / run_name

            for level in range(NUM_LEVELS):
                ckpt_path = run_folder / f"level_{level:02d}.pt"
                rel_ckpt_str = str(ckpt_path)

                if not ckpt_path.is_file():
                    inv_row = {
                        "dataset": ds_name,
                        "model": DEFAULT_MODEL,
                        "seed": seed,
                        "level": level,
                        "checkpoint_path": rel_ckpt_str,
                        "checkpoint_sha256": None,
                        "audit_status": "missing_checkpoint",
                        "checkpoint_accuracy": None,
                        "results_summary_accuracy": None,
                        "recomputed_accuracy": None,
                        "correct_count": -1,
                        "total_samples": 0,
                        "accuracy_diff": None,
                        "accuracy_diff_checkpoint": None,
                        "accuracy_diff_summary": None,
                        "tolerance": accuracy_tolerance,
                        "is_within_tolerance": False,
                        "pruned_weights_strictly_zero": None,
                        "all_parameters_finite": None,
                    }
                    inventory_records.append(inv_row)
                    accuracy_records.append(inv_row)
                    continue

                ckpt_hash = compute_file_sha256(ckpt_path)
                checkpoint_hashes[rel_ckpt_str] = ckpt_hash

                # Confirm checkpoint SHA-256 matches Extension 1 inventory
                ext1_key = (ds_name, seed, level)
                if ext1_key in ext1_ckpt_map:
                    if ext1_ckpt_map[ext1_key] != ckpt_hash:
                        raise ValueError(
                            f"Checkpoint SHA-256 mismatch against Extension 1 source for {ckpt_path}: "
                            f"current {ckpt_hash} vs ext1 {ext1_ckpt_map[ext1_key]}"
                        )
                else:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} missing in Extension 1 source inventory: {ext1_dir}"
                    )

                if not is_ds_avail:
                    ds_err_reason = dataset_status[ds_name]["status"]
                    inv_row = {
                        "dataset": ds_name,
                        "model": DEFAULT_MODEL,
                        "seed": seed,
                        "level": level,
                        "checkpoint_path": rel_ckpt_str,
                        "checkpoint_sha256": ckpt_hash,
                        "audit_status": f"blocked: test data unavailable ({ds_err_reason})",
                        "checkpoint_accuracy": None,
                        "results_summary_accuracy": None,
                        "recomputed_accuracy": None,
                        "correct_count": -1,
                        "total_samples": 0,
                        "accuracy_diff": None,
                        "accuracy_diff_checkpoint": None,
                        "accuracy_diff_summary": None,
                        "tolerance": accuracy_tolerance,
                        "is_within_tolerance": False,
                        "pruned_weights_strictly_zero": None,
                        "all_parameters_finite": None,
                    }
                    inventory_records.append(inv_row)
                    accuracy_records.append(inv_row)
                    continue

                # Run evaluation with specific exception catching
                try:
                    eval_res = evaluate_checkpoint_predictions(
                        ckpt_path=ckpt_path,
                        x_test=x_test,
                        y_test=y_test,
                        expected_dataset=ds_name,
                        expected_seed=seed,
                        expected_level=level,
                        runs_dir=runs_dir,
                        results_dir=results_dir,
                        device=dev,
                        batch_size=batch_size,
                        accuracy_tolerance=accuracy_tolerance,
                    )
                    status_str = "evaluated"
                except IncompatibleCheckpointError as exc:
                    status_str = f"incompatible_checkpoint: {exc.reason}"
                    eval_res = None
                except (ValueError, KeyError, RuntimeError, torch.jit.Error) as exc:
                    status_str = f"incompatible_checkpoint: {exc}"
                    eval_res = None

                if eval_res is not None:
                    preds = eval_res["preds"]
                    targets = eval_res["targets"]
                    example_ids = eval_res["example_ids"]

                    model_key = (ds_name, seed, level)
                    cached_predictions[model_key] = {
                        "preds": preds,
                        "targets": targets,
                        "example_ids": example_ids,
                        "dataset": ds_name,
                        "model": DEFAULT_MODEL,
                        "seed": seed,
                        "level": level,
                        "checkpoint_kind": "best_state",
                        "checkpoint_sha256": ckpt_hash,
                        "data_fingerprint": ds_fp,
                    }

                    # Save self-contained immutable prediction cache
                    cache_rel = f"predictions/{ds_name}_{DEFAULT_MODEL}_seed{seed}_level{level:02d}.npz"
                    cache_full_path = out_dir / cache_rel

                    np.savez_compressed(
                        cache_full_path,
                        predicted_labels=preds,
                        targets=targets,
                        example_ids=example_ids,
                        dataset=ds_name,
                        model=DEFAULT_MODEL,
                        seed=seed,
                        level=level,
                        checkpoint_kind="best_state",
                        checkpoint_sha256=ckpt_hash,
                        data_fingerprint=ds_fp,
                        shape=np.array(list(x_test.shape), dtype=np.int64),
                        dtype="float32",
                        preprocessing_version=PREPROCESSING_VERSION,
                        software_versions=software_versions_str,
                        device=str(dev),
                        batch_size=batch_size,
                        recomputed_accuracy=eval_res["recomputed_accuracy"],
                        checkpoint_accuracy=eval_res["checkpoint_accuracy"],
                        correct_count=eval_res["correct_count"],
                        total_samples=eval_res["total_samples"],
                    )
                    prediction_cache_hashes[cache_rel] = compute_file_sha256(cache_full_path)

                    acc_row = {
                        "dataset": ds_name,
                        "model": DEFAULT_MODEL,
                        "seed": seed,
                        "level": level,
                        "weights_remaining": eval_res["weights_remaining"],
                        "checkpoint_path": rel_ckpt_str,
                        "checkpoint_sha256": ckpt_hash,
                        "checkpoint_accuracy": eval_res["checkpoint_accuracy"],
                        "results_summary_accuracy": eval_res["results_summary_accuracy"],
                        "recomputed_accuracy": eval_res["recomputed_accuracy"],
                        "correct_count": eval_res["correct_count"],
                        "total_samples": eval_res["total_samples"],
                        "accuracy_diff": eval_res["accuracy_diff_checkpoint"],
                        "accuracy_diff_checkpoint": eval_res["accuracy_diff_checkpoint"],
                        "accuracy_diff_summary": eval_res["accuracy_diff_summary"],
                        "tolerance": accuracy_tolerance,
                        "is_within_tolerance": eval_res["is_within_tolerance"],
                        "pruned_weights_strictly_zero": eval_res["pruned_weights_strictly_zero"],
                        "all_parameters_finite": eval_res["all_parameters_finite"],
                        "audit_status": status_str,
                    }
                else:
                    acc_row = {
                        "dataset": ds_name,
                        "model": DEFAULT_MODEL,
                        "seed": seed,
                        "level": level,
                        "weights_remaining": None,
                        "checkpoint_path": rel_ckpt_str,
                        "checkpoint_sha256": ckpt_hash,
                        "checkpoint_accuracy": None,
                        "results_summary_accuracy": None,
                        "recomputed_accuracy": None,
                        "correct_count": -1,
                        "total_samples": 0,
                        "accuracy_diff": None,
                        "accuracy_diff_checkpoint": None,
                        "accuracy_diff_summary": None,
                        "tolerance": accuracy_tolerance,
                        "is_within_tolerance": False,
                        "pruned_weights_strictly_zero": None,
                        "all_parameters_finite": None,
                        "audit_status": status_str,
                    }

                inventory_records.append(acc_row)
                accuracy_records.append(acc_row)

    df_inventory = pd.DataFrame(inventory_records)
    df_accuracy = pd.DataFrame(accuracy_records)

    # 4. Pairwise prediction agreement calculation
    pair_records = []
    ext1_pairs_df = pd.read_csv(ext1_pairs_csv)
    weights_map = (
        ext1_pairs_df.groupby(["dataset", "level"])["weights_remaining"].first().to_dict()
    )

    for ds_name in SUPPORTED_DATASETS:
        if not dataset_status[ds_name]["available"]:
            continue
        _, y_test, ds_fp = loaded_datasets[ds_name]
        targets_np = y_test.cpu().numpy()

        for level in range(NUM_LEVELS):
            wr = weights_map.get((ds_name, level))
            if wr is None:
                raise ValueError(f"Missing weights_remaining in Extension 1 for {ds_name} level {level}")

            for seed_a, seed_b in SEED_PAIRS:
                key_a = (ds_name, seed_a, level)
                key_b = (ds_name, seed_b, level)

                # Both models must be evaluated and verified
                if key_a in cached_predictions and key_b in cached_predictions:
                    info_a = cached_predictions[key_a]
                    info_b = cached_predictions[key_b]

                    agr_res = compute_pair_agreement(
                        preds_a=info_a["preds"],
                        preds_b=info_b["preds"],
                        targets=targets_np,
                        example_ids=info_a["example_ids"],
                        meta_a=info_a,
                        meta_b=info_b,
                    )

                    pair_records.append(
                        {
                            "dataset": ds_name,
                            "model": DEFAULT_MODEL,
                            "level": level,
                            "weights_remaining": float(wr),
                            "seed_a": seed_a,
                            "seed_b": seed_b,
                            "agreement": agr_res["agreement"],
                            "shared_correct": agr_res["shared_correct"],
                            "shared_wrong": agr_res["shared_wrong"],
                            "disagreement": agr_res["disagreement"],
                            "accuracy_a": agr_res["accuracy_a"],
                            "accuracy_b": agr_res["accuracy_b"],
                            "n_samples": agr_res["n_samples"],
                        }
                    )

    df_pairs = pd.DataFrame(pair_records)

    # 5. Aggregate agreement summary using unique models for accuracy
    df_summary = aggregate_agreement_summary(df_pairs, df_accuracy)

    # 6. Join structure and function pairs
    df_sf_pairs = link_structure_and_function(df_pairs, ext1_pairs_csv)

    # 7. Write output tables and inventory
    path_inventory = tables_dir / "prediction_inventory.csv"
    path_root_inventory = out_dir / "artifact_inventory.csv"
    path_accuracy = tables_dir / "prediction_accuracy.csv"
    path_pairs = tables_dir / "prediction_agreement_pairs.csv"
    path_summary = tables_dir / "prediction_agreement_summary.csv"
    path_sf_pairs = tables_dir / "structure_function_pairs.csv"

    df_inventory.to_csv(path_inventory, index=False)
    df_inventory.to_csv(path_root_inventory, index=False)
    df_accuracy.to_csv(path_accuracy, index=False)
    df_pairs.to_csv(path_pairs, index=False)
    df_summary.to_csv(path_summary, index=False)
    df_sf_pairs.to_csv(path_sf_pairs, index=False)

    # 8. Generate Figures
    from extension_analysis.plots import (
        plot_prediction_agreement_curves,
        plot_structure_function_scatter,
    )

    fig_curves_png = figures_dir / "fig_ext3_prediction_agreement.png"
    fig_curves_pdf = figures_dir / "fig_ext3_prediction_agreement.pdf"
    plot_prediction_agreement_curves(
        df_summary=df_summary,
        df_pairs=df_pairs,
        dataset_status=dataset_status,
        save_png=fig_curves_png,
        save_pdf=fig_curves_pdf,
    )

    fig_sf_png = figures_dir / "fig_ext3_structure_function.png"
    fig_sf_pdf = figures_dir / "fig_ext3_structure_function.pdf"
    plot_structure_function_scatter(
        df_sf_pairs=df_sf_pairs,
        dataset_status=dataset_status,
        save_png=fig_sf_png,
        save_pdf=fig_sf_pdf,
    )

    # 9. Compute and record hashes
    output_table_hashes = {
        "tables/prediction_inventory.csv": compute_file_sha256(path_inventory),
        "artifact_inventory.csv": compute_file_sha256(path_root_inventory),
        "tables/prediction_accuracy.csv": compute_file_sha256(path_accuracy),
        "tables/prediction_agreement_pairs.csv": compute_file_sha256(path_pairs),
        "tables/prediction_agreement_summary.csv": compute_file_sha256(path_summary),
        "tables/structure_function_pairs.csv": compute_file_sha256(path_sf_pairs),
    }

    output_figure_hashes = {
        "figures/fig_ext3_prediction_agreement.png": compute_file_sha256(fig_curves_png),
        "figures/fig_ext3_prediction_agreement.pdf": compute_file_sha256(fig_curves_pdf),
        "figures/fig_ext3_structure_function.png": compute_file_sha256(fig_sf_png),
        "figures/fig_ext3_structure_function.pdf": compute_file_sha256(fig_sf_pdf),
    }

    # Record hashes of executed code, models, and data loading
    code_files = [
        Path("extension_analysis/prediction_agreement.py"),
        Path("extension_analysis/plots.py"),
        Path("extension_analysis/mask_stability.py"),
        Path("lth/models.py"),
        Path("lth/data.py"),
    ]
    code_hashes = {str(cf): compute_file_sha256(cf) for cf in code_files if cf.is_file()}

    # Record config and summary hashes from runs and results
    config_hashes = {}
    summary_hashes = {}
    for ds in SUPPORTED_DATASETS:
        for s in EXPECTED_SEEDS:
            r_name = f"{ds}_adam_seed{s}"
            for p_base in [runs_dir, results_dir]:
                if p_base is not None:
                    c_p = p_base / r_name / "config.json"
                    if c_p.is_file():
                        config_hashes[str(c_p)] = compute_file_sha256(c_p)
                    s_p = p_base / r_name / "summary.csv"
                    if s_p.is_file():
                        summary_hashes[str(s_p)] = compute_file_sha256(s_p)

    ext1_hashes = {
        "tables/mask_jaccard_pairs.csv": compute_file_sha256(ext1_pairs_csv),
        "tables/mask_jaccard_summary.csv": compute_file_sha256(ext1_summary_csv),
        "metadata.json": compute_file_sha256(ext1_metadata),
    }
    if ext1_inv_csv.is_file():
        ext1_hashes[str(ext1_inv_csv.relative_to(ext1_dir))] = compute_file_sha256(ext1_inv_csv)

    metadata = {
        "extension": "Extension 3: Cross-Seed Prediction Agreement Analysis",
        "execution_id": execution_id,
        "timestamp": timestamp,
        "model": DEFAULT_MODEL,
        "evaluated_state": "best_state",
        "accuracy_tolerance": accuracy_tolerance,
        "preprocessing_version": PREPROCESSING_VERSION,
        "software_versions": software_versions_dict,
        "supported_datasets": SUPPORTED_DATASETS,
        "dataset_status": dataset_status,
        "test_dataset_fingerprints": {
            k: v["fingerprint"] for k, v in dataset_status.items() if v["available"]
        },
        "extension1_source": {
            "path": str(ext1_dir),
            "table_hashes": ext1_hashes,
        },
        "config_hashes": config_hashes,
        "summary_hashes": summary_hashes,
        "input_checkpoint_hashes": checkpoint_hashes,
        "code_file_hashes": code_hashes,
        "prediction_cache_hashes": prediction_cache_hashes,
        "output_table_hashes": output_table_hashes,
        "output_figure_hashes": output_figure_hashes,
        "summary_statistics": {
            "n_checkpoints_total": len(df_inventory),
            "n_checkpoints_evaluated": int((df_inventory["audit_status"] == "evaluated").sum()),
            "n_checkpoints_blocked": int(
                df_inventory["audit_status"].str.startswith("blocked").sum()
            ),
            "n_checkpoints_incompatible": int(
                df_inventory["audit_status"].str.startswith("incompatible").sum()
            ),
            "n_pairs_evaluated": len(df_pairs),
        },
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return out_dir
