"""Tests for Extension 1: Cross-seed mask stability analysis."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import pytest
import torch

from extension_analysis.mask_stability import (
    EXPECTED_COMMON_CONFIG,
    EXPECTED_LAYERS,
    EXPECTED_SHAPES,
    aggregate_jaccard_summary,
    audit_all_runs,
    audit_run_checkpoints,
    compare_seed_pair_masks,
    compute_all_pairs,
    compute_layer_jaccard,
    compute_overall_jaccard,
    load_checkpoint_masks,
    run_extension1_pipeline,
    verify_execution_artifacts,
)


def _make_valid_config(dataset="mnist", seed=1):
    return {
        "dataset": dataset,
        "seed": seed,
        **EXPECTED_COMMON_CONFIG,
    }


def _make_dense_masks():
    return {
        lyr: torch.ones(EXPECTED_SHAPES[lyr], dtype=torch.bool) for lyr in EXPECTED_LAYERS
    }


def test_jaccard_identity_and_symmetry():
    """Identity: J(A, A) == 1.0; Symmetry: J(A, B) == J(B, A)."""
    t_a = torch.tensor([[True, False, True], [False, True, False]])
    t_b = torch.tensor([[True, True, False], [False, True, False]])

    res_aa = compute_layer_jaccard(t_a, t_a)
    assert res_aa["jaccard"] == 1.0
    assert res_aa["surviving_a"] == 3
    assert res_aa["surviving_b"] == 3
    assert res_aa["intersection"] == 3
    assert res_aa["union"] == 3

    res_ab = compute_layer_jaccard(t_a, t_b)
    res_ba = compute_layer_jaccard(t_b, t_a)
    assert res_ab["jaccard"] == res_ba["jaccard"]
    assert res_ab["intersection"] == res_ba["intersection"]
    assert res_ab["union"] == res_ba["union"]


def test_jaccard_known_values():
    """Exact arithmetic check on synthetic binary tensors."""
    t_a = torch.tensor([True, True, False, False])
    t_b = torch.tensor([False, True, True, False])
    res = compute_layer_jaccard(t_a, t_b)

    assert res["surviving_a"] == 2
    assert res["surviving_b"] == 2
    assert res["intersection"] == 1
    assert res["union"] == 3
    assert math.isclose(res["jaccard"], 1.0 / 3.0, rel_tol=1e-7)


def test_empty_union_returns_nan():
    """Empty masks (union == 0) must return NaN for Jaccard."""
    t_empty = torch.zeros((5, 5), dtype=torch.bool)
    res = compute_layer_jaccard(t_empty, t_empty)
    assert res["intersection"] == 0
    assert res["union"] == 0
    assert math.isnan(res["jaccard"])

    overall = compute_overall_jaccard([res, res])
    assert overall["union"] == 0
    assert math.isnan(overall["jaccard"])


def test_overall_jaccard_not_simple_layer_average():
    """Overall Jaccard must be sum(intersections) / sum(unions), NOT average of layer Jaccards."""
    l1 = {"surviving_a": 5, "surviving_b": 5, "intersection": 1, "union": 10, "jaccard": 0.1}
    l2 = {"surviving_a": 95, "surviving_b": 95, "intersection": 90, "union": 100, "jaccard": 0.9}

    overall = compute_overall_jaccard([l1, l2])
    simple_avg = (l1["jaccard"] + l2["jaccard"]) / 2.0

    assert not math.isclose(overall["jaccard"], simple_avg, rel_tol=1e-3)
    assert math.isclose(overall["jaccard"], 91.0 / 110.0, rel_tol=1e-7)


def test_inputs_are_not_modified():
    """Ensure tensors are strictly unmodified during comparison."""
    m_a = torch.tensor([[True, False], [True, True]])
    m_b = torch.tensor([[False, False], [True, True]])
    orig_a = m_a.clone()
    orig_b = m_b.clone()

    _ = compute_layer_jaccard(m_a, m_b)
    assert torch.equal(m_a, orig_a)
    assert torch.equal(m_b, orig_b)


def test_shape_mismatch_raises():
    """Shape mismatch between masks must raise ValueError."""
    m_a = torch.zeros((10, 20), dtype=torch.bool)
    m_b = torch.zeros((10, 25), dtype=torch.bool)
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_layer_jaccard(m_a, m_b)


def test_seed_pair_no_self_comparison():
    """Self comparison seed_a == seed_b must raise ValueError."""
    dummy_masks = _make_dense_masks()
    with pytest.raises(ValueError, match="Self-comparison not allowed"):
        compare_seed_pair_masks(
            dataset="mnist",
            model="lenet300100",
            level=1,
            weights_remaining=0.8,
            seed_a=1,
            seed_b=1,
            masks_a=dummy_masks,
            masks_b=dummy_masks,
        )


def test_aggregate_summary_handles_single_pair_and_multiple():
    """Summary handles single pair (std is NaN) and multiple pairs (ddof=1 sample std)."""
    df_single = pd.DataFrame([
        {
            "dataset": "mnist",
            "model": "lenet300100",
            "level": 1,
            "weights_remaining": 0.8,
            "seed_a": 1,
            "seed_b": 2,
            "scope": "overall",
            "jaccard": 0.65,
        }
    ])
    sum_single = aggregate_jaccard_summary(df_single, expected_pairs=3)
    assert len(sum_single) == 1
    assert sum_single.iloc[0]["n_pairs"] == 1
    assert math.isnan(sum_single.iloc[0]["std"])
    assert sum_single.iloc[0]["mean"] == 0.65

    df_three = pd.DataFrame([
        {
            "dataset": "mnist",
            "model": "lenet300100",
            "level": 1,
            "weights_remaining": 0.8,
            "seed_a": 1,
            "seed_b": 2,
            "scope": "overall",
            "jaccard": 0.60,
        },
        {
            "dataset": "mnist",
            "model": "lenet300100",
            "level": 1,
            "weights_remaining": 0.8,
            "seed_a": 1,
            "seed_b": 42,
            "scope": "overall",
            "jaccard": 0.62,
        },
        {
            "dataset": "mnist",
            "model": "lenet300100",
            "level": 1,
            "weights_remaining": 0.8,
            "seed_a": 2,
            "seed_b": 42,
            "scope": "overall",
            "jaccard": 0.64,
        },
    ])
    sum_three = aggregate_jaccard_summary(df_three, expected_pairs=3)
    assert len(sum_three) == 1
    assert sum_three.iloc[0]["n_pairs"] == 3
    assert math.isclose(sum_three.iloc[0]["mean"], 0.62, rel_tol=1e-5)
    expected_std = float(np.std([0.60, 0.62, 0.64], ddof=1))
    assert math.isclose(sum_three.iloc[0]["std"], expected_std, rel_tol=1e-5)


def test_config_and_pruning_mismatch_rejected(tmp_path: Path):
    """Mismatches in optimizer, lr, steps, or pruning strategy must reject checkpoints."""
    run_dir = tmp_path / "mnist_adam_seed1"
    run_dir.mkdir(parents=True)

    # Bad optimizer and pruning
    bad_cfg = _make_valid_config("mnist", 1)
    bad_cfg["optimizer"] = "sgd"
    bad_cfg["pruning"] = "global"
    (run_dir / "config.json").write_text(json.dumps(bad_cfg))
    (run_dir / "summary.csv").write_text("level,weights_remaining\n0,1.0\n")

    ckpt_0 = {
        "masks": _make_dense_masks(),
        "metrics": {"level": 0, "weights_remaining": 1.0},
    }
    torch.save(ckpt_0, run_dir / "level_00.pt")

    inv, valid = audit_run_checkpoints(tmp_path, None, "mnist", 1, levels=[0])
    assert not inv[0]["is_valid"]
    assert "Config mismatch" in inv[0]["rejection_reason"]


def test_summary_duplicate_level_or_mismatch_rejected(tmp_path: Path):
    """Corrupted summary, duplicate level, or fraction mismatch must be rejected."""
    run_dir = tmp_path / "mnist_adam_seed1"
    run_dir.mkdir(parents=True)
    cfg = _make_valid_config("mnist", 1)
    (run_dir / "config.json").write_text(json.dumps(cfg))

    # Duplicate level in summary
    (run_dir / "summary.csv").write_text("level,weights_remaining\n0,1.0\n0,0.8\n")
    ckpt_0 = {
        "masks": _make_dense_masks(),
        "metrics": {"level": 0, "weights_remaining": 1.0},
    }
    torch.save(ckpt_0, run_dir / "level_00.pt")

    inv, valid = audit_run_checkpoints(tmp_path, None, "mnist", 1, levels=[0])
    assert not inv[0]["is_valid"]
    assert "Duplicate levels" in inv[0]["rejection_reason"]


def test_checkpoint_metrics_missing_nan_or_level_mismatch(tmp_path: Path):
    """Metrics missing 'level', NaN fraction, or level mismatch must be rejected."""
    run_dir = tmp_path / "mnist_adam_seed1"
    run_dir.mkdir(parents=True)
    cfg = _make_valid_config("mnist", 1)
    (run_dir / "config.json").write_text(json.dumps(cfg))
    (run_dir / "summary.csv").write_text("level,weights_remaining\n0,1.0\n1,0.8\n")

    # Missing level in metrics
    ckpt_0 = {
        "masks": _make_dense_masks(),
        "metrics": {"weights_remaining": 1.0},  # missing level
    }
    torch.save(ckpt_0, run_dir / "level_00.pt")

    # NaN fraction in level 1
    masks_1 = _make_dense_masks()
    masks_1["fc1.weight"][0, 0] = False
    ckpt_1 = {
        "masks": masks_1,
        "metrics": {"level": 1, "weights_remaining": float("nan")},
    }
    torch.save(ckpt_1, run_dir / "level_01.pt")

    inv, valid = audit_run_checkpoints(tmp_path, None, "mnist", 1, levels=[0, 1])
    assert not inv[0]["is_valid"]
    assert "Missing 'level'" in inv[0]["rejection_reason"]
    assert not inv[1]["is_valid"]
    assert "non-finite" in inv[1]["rejection_reason"]


def test_dense_baseline_violated_at_level_0(tmp_path: Path):
    """Level 0 with missing connections or non-1.0 fraction must be rejected."""
    run_dir = tmp_path / "mnist_adam_seed1"
    run_dir.mkdir(parents=True)
    cfg = _make_valid_config("mnist", 1)
    (run_dir / "config.json").write_text(json.dumps(cfg))
    (run_dir / "summary.csv").write_text("level,weights_remaining\n0,0.999\n")

    masks_sparse_0 = _make_dense_masks()
    masks_sparse_0["fc1.weight"][0, 0] = False  # pruned in level 0!
    ckpt_0 = {
        "masks": masks_sparse_0,
        "metrics": {"level": 0, "weights_remaining": 0.999},
    }
    torch.save(ckpt_0, run_dir / "level_00.pt")

    inv, valid = audit_run_checkpoints(tmp_path, None, "mnist", 1, levels=[0])
    assert not inv[0]["is_valid"]
    assert "Dense baseline violated" in inv[0]["rejection_reason"]


def test_cross_seed_layerwise_count_mismatch_rejected():
    """Mismatched survivor counts across seeds at same level must raise ValueError."""
    m_a = _make_dense_masks()
    m_b = _make_dense_masks()
    m_b["fc1.weight"][0, 0] = False  # Different retained count

    store = {
        ("mnist", 1, 1): {"masks": m_a, "weights_remaining": 1.0},
        ("mnist", 2, 1): {"masks": m_b, "weights_remaining": 0.999},
    }
    with pytest.raises(ValueError, match="Retained count mismatch"):
        compute_all_pairs(store, datasets=["mnist"], seeds=[1, 2], levels=[1])


def test_execution_directory_no_overwrite(tmp_path: Path):
    """Pipeline must refuse to overwrite an existing execution directory."""
    existing_dir = tmp_path / "ext_test"
    existing_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        run_extension1_pipeline(
            runs_dir="runs",
            results_dir="results",
            execution_id="ext_test",
            out_base=str(tmp_path),
        )


def test_read_mode_detects_missing_or_corrupt_files(tmp_path: Path):
    """verify_execution_artifacts must detect missing artifacts or missing columns."""
    exec_dir = tmp_path / "execution_test"
    exec_dir.mkdir(parents=True)
    (exec_dir / "tables").mkdir()
    (exec_dir / "figures").mkdir()

    dummy_input = tmp_path / "input.pt"
    dummy_input.write_bytes(b"original_bytes")

    meta = {
        "inputs": {
            str(dummy_input): {"exists": True}
        }
    }
    (exec_dir / "metadata.json").write_text(json.dumps(meta))
    (exec_dir / "artifact_inventory.csv").write_text("dataset,seed,level,is_valid\nmnist,1,0,True\n")
    (exec_dir / "tables/mask_jaccard_pairs.csv").write_text(
        "dataset,model,level,seed_a,seed_b,scope,jaccard\nmnist,lenet,0,1,2,overall,1.0\n"
    )
    (exec_dir / "tables/mask_jaccard_summary.csv").write_text(
        "dataset,model,level,scope,mean\nmnist,lenet,0,overall,1.0\n"
    )
    for f in ["figures/mask_stability_grid.png", "figures/mask_stability_grid.pdf"]:
        (exec_dir / f).write_text("dummy")

    # Intact
    ok, errors = verify_execution_artifacts(exec_dir)
    assert ok
    assert len(errors) == 0

    # Missing artifact
    (exec_dir / "tables/mask_jaccard_pairs.csv").unlink()
    ok2, errors2 = verify_execution_artifacts(exec_dir)
    assert not ok2
    assert any("Missing required artifact" in e for e in errors2)


def test_partial_artifact_coverage_graceful(tmp_path: Path):
    """Missing runs/levels are accurately recorded in inventory, and valid ones remain usable."""
    run_dir = tmp_path / "mnist_adam_seed1"
    run_dir.mkdir(parents=True)
    cfg = _make_valid_config("mnist", 1)
    (run_dir / "config.json").write_text(json.dumps(cfg))
    (run_dir / "summary.csv").write_text("level,weights_remaining\n0,1.0\n")

    ckpt_0 = {
        "masks": _make_dense_masks(),
        "metrics": {"level": 0, "weights_remaining": 1.0},
    }
    torch.save(ckpt_0, run_dir / "level_00.pt")
    # level 1 is deliberately missing

    inv_df, store = audit_all_runs(
        runs_dir=tmp_path,
        results_dir=None,
        datasets=["mnist"],
        seeds=[1],
        levels=[0, 1],
    )
    assert len(inv_df) == 2
    assert inv_df.iloc[0]["is_valid"]
    assert not inv_df.iloc[1]["is_valid"]
    assert "File not found" in inv_df.iloc[1]["rejection_reason"]
    assert ("mnist", 1, 0) in store
    assert ("mnist", 1, 1) not in store


def test_original_notebook_cells_and_metadata_preserved():
    """Ensure original 21 cells structure (cell_type, source, order) matches baseline commit 247de41."""
    baseline_commit = "247de41"
    orig_text = subprocess.check_output(
        ["git", "show", f"{baseline_commit}:analysis.ipynb"],
        stderr=subprocess.DEVNULL,
    ).decode("utf-8")
    orig_nb = json.loads(orig_text)

    with open("analysis.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Verify nbformat
    assert nb.get("nbformat") == orig_nb.get("nbformat"), "nbformat altered!"
    assert nb.get("nbformat_minor") == orig_nb.get("nbformat_minor"), "nbformat_minor altered!"

    # Verify that original 21 cells are preserved in order, type, and source
    assert len(orig_nb["cells"]) == 21
    assert len(nb["cells"]) >= 21, "Current notebook has fewer cells than the original baseline!"

    for i in range(21):
        orig_c = orig_nb["cells"][i]
        curr_c = nb["cells"][i]
        assert curr_c.get("cell_type") == orig_c.get("cell_type"), f"Cell {i} cell_type mismatch!"
        assert curr_c.get("source") == orig_c.get("source"), f"Cell {i} source mismatch!"
