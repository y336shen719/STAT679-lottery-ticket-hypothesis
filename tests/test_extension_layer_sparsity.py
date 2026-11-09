"""Unit tests for Extension 4: Per-layer sparsity and surviving-weight allocation."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from extension_analysis.layer_sparsity import (
    LAYER_PARAMS,
    LAYER_SHAPES,
    SCHEDULED_PRUNE_RATES,
    TOTAL_PRUNABLE_PARAMS,
    aggregate_layer_sparsity_summary,
    compute_layer_sparsity_records,
    create_layer_allocation_summary,
    run_extension4_pipeline,
    verify_ext4_artifacts,
)


def test_layer_parameters_and_shapes_consistency():
    """Verify layer parameter counts sum to 266,200 and shapes match LeNet-300-100."""
    total = sum(LAYER_PARAMS.values())
    assert total == TOTAL_PRUNABLE_PARAMS == 266200
    assert LAYER_PARAMS["fc1.weight"] == 300 * 784 == 235200
    assert LAYER_PARAMS["fc2.weight"] == 100 * 300 == 30000
    assert LAYER_PARAMS["fc3.weight"] == 10 * 100 == 1000

    assert LAYER_SHAPES["fc1.weight"] == (300, 784)
    assert LAYER_SHAPES["fc2.weight"] == (100, 300)
    assert LAYER_SHAPES["fc3.weight"] == (10, 100)


def test_level_0_dense_baseline_invariants(tmp_path):
    """Verify that Level 0 has zero sparsity and newly_pruned/actual_rate are NaN."""
    # Create minimal mock store for Level 0 and Level 1
    store = {
        ("mnist", 1, 0): {
            "masks": {
                "fc1.weight": torch.ones(300, 784, dtype=torch.bool),
                "fc2.weight": torch.ones(100, 300, dtype=torch.bool),
                "fc3.weight": torch.ones(10, 100, dtype=torch.bool),
            },
            "weights_remaining": 1.0,
            "path": "dummy_l0.pt",
            "sha256": "dummy_sha_0",
        },
        ("mnist", 1, 1): {
            "masks": {
                "fc1.weight": torch.zeros(300, 784, dtype=torch.bool),
                "fc2.weight": torch.zeros(100, 300, dtype=torch.bool),
                "fc3.weight": torch.zeros(10, 100, dtype=torch.bool),
            },
            "weights_remaining": 0.8,
            "path": "dummy_l1.pt",
            "sha256": "dummy_sha_1",
        },
    }
    # Set surviving counts for level 1
    store[("mnist", 1, 1)]["masks"]["fc1.weight"][:188160 // 784, :] = True
    store[("mnist", 1, 1)]["masks"]["fc2.weight"][:24000 // 300, :] = True
    store[("mnist", 1, 1)]["masks"]["fc3.weight"][:900 // 100, :] = True

    df = compute_layer_sparsity_records(store, datasets=["mnist"], seeds=[1], levels=[0, 1])

    l0_rows = df[df["level"] == 0]
    for _, row in l0_rows.iterrows():
        assert row["surviving"] == row["total_prunable"]
        assert row["pruned"] == 0
        assert row["sparsity"] == 0.0
        assert np.isnan(row["newly_pruned"])
        assert np.isnan(row["actual_prune_rate"])

    l1_rows = df[df["level"] == 1]
    fc1_l1 = l1_rows[l1_rows["layer"] == "fc1.weight"].iloc[0]
    assert fc1_l1["surviving"] == 188160
    assert fc1_l1["newly_pruned"] == 47040
    assert pytest.approx(fc1_l1["actual_prune_rate"], 1e-6) == 0.20


def test_cross_seed_and_dataset_invariance_on_actual_runs():
    """Verify that surviving counts and shares are strictly identical across seeds and datasets."""
    from extension_analysis.mask_stability import audit_all_runs

    df_audit, store = audit_all_runs(Path("runs"), Path("results"))
    df_det = compute_layer_sparsity_records(store)
    df_sum = aggregate_layer_sparsity_summary(df_det)

    # Every layer at every level must have identical counts across seeds
    assert df_sum["counts_identical"].all(), "Surviving counts vary across random seeds!"
    assert (df_sum["surviving_sd"] == 0.0).all(), "Surviving SD is non-zero across seeds!"

    # Check across datasets: mnist vs fashion_mnist
    pivot = df_sum.pivot(index=["level", "layer"], columns="dataset", values="surviving_mean")
    assert (pivot["mnist"] == pivot["fashion_mnist"]).all(), "Surviving counts vary between MNIST and Fashion-MNIST!"


def test_layer_share_partition_unity():
    """Verify that FC1, FC2, and FC3 surviving shares sum to exactly 1.0 (100%) at every level."""
    from extension_analysis.mask_stability import audit_all_runs

    _, store = audit_all_runs(Path("runs"), Path("results"))
    df_det = compute_layer_sparsity_records(store)
    layers_only = df_det[df_det["layer"] != "overall"]

    share_sums = layers_only.groupby(["dataset", "seed", "level"])["surviving_share"].sum()
    for (ds, s, lvl), total_share in share_sums.items():
        assert pytest.approx(total_share, abs=1e-7) == 1.0, f"Shares do not sum to 1.0 at {ds} seed {s} L{lvl}"


def test_schedule_adherence_within_integer_rounding_bound():
    """Verify that actual prune rates match scheduled rates within theoretical round() bounds."""
    from extension_analysis.mask_stability import audit_all_runs

    _, store = audit_all_runs(Path("runs"), Path("results"))
    df_det = compute_layer_sparsity_records(store)
    sparse_layers = df_det[(df_det["level"] > 0) & (df_det["layer"] != "overall")]

    for _, row in sparse_layers.iterrows():
        act_rate = row["actual_prune_rate"]
        sched_rate = row["scheduled_prune_rate"]
        # previous surviving is surviving + newly_pruned
        prev_surviving = row["surviving"] + row["newly_pruned"]
        # integer round() discrepancy cannot exceed 0.5 / prev_surviving
        max_allowed_diff = 0.5 / prev_surviving
        actual_diff = abs(act_rate - sched_rate)
        assert actual_diff <= max_allowed_diff + 1e-9, (
            f"Prune rate discrepancy exceeds integer quantization bound at L{row['level']} {row['layer']}: "
            f"actual {act_rate}, scheduled {sched_rate}, max_allowed {max_allowed_diff}"
        )


def test_ext4_pipeline_overwrite_protection(tmp_path):
    """Verify that run_extension4_pipeline refuses to overwrite an existing directory."""
    existing_dir = tmp_path / "existing_ext4"
    existing_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        run_extension4_pipeline(
            runs_dir="runs",
            results_dir="results",
            execution_id="existing_ext4",
            out_base=tmp_path,
        )


def test_verify_ext4_artifacts_detects_tampering(tmp_path):
    """Verify that verify_ext4_artifacts catches missing files or tampered content."""
    # Run minimal pipeline into tmp_path
    out_dir = run_extension4_pipeline(
        runs_dir="runs",
        results_dir="results",
        out_base=tmp_path,
    )

    # 1. Clean verification must pass
    ok, errs = verify_ext4_artifacts(out_dir)
    assert ok, f"Clean verification failed: {errs}"

    # 2. Corrupting a table (missing required columns) must fail
    table_path = out_dir / "tables" / "layer_allocation_summary.csv"
    orig_text = table_path.read_text(encoding="utf-8")
    table_path.write_text("invalid,columns\n1,2\n", encoding="utf-8")

    ok_tampered, errs_tampered = verify_ext4_artifacts(out_dir)
    assert not ok_tampered, "Failed to detect corrupted table!"
    assert any("missing required columns" in e for e in errs_tampered["outputs"])

    # Restore table
    table_path.write_text(orig_text, encoding="utf-8")

    # 3. Deleting a figure must fail
    fig_path = out_dir / "figures" / "layer_sparsity_curves.png"
    fig_path.unlink()
    ok_deleted, errs_deleted = verify_ext4_artifacts(out_dir)
    assert not ok_deleted, "Failed to detect missing figure!"
    assert any("missing" in e for e in errs_deleted["outputs"])


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

    # Verify that original 21 cells exist in the same order with unchanged type and source
    assert len(orig_nb["cells"]) == 21
    assert len(nb["cells"]) >= 21

    for i in range(21):
        orig_c = orig_nb["cells"][i]
        curr_c = nb["cells"][i]
        assert curr_c.get("cell_type") == orig_c.get("cell_type"), f"Cell {i} cell_type mismatch!"
        assert curr_c.get("source") == orig_c.get("source"), f"Cell {i} source mismatch!"

