"""Tests for Extension 2: Density-matched random mask reference analysis."""

import copy
import json
import math
import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import pytest

from extension_analysis.random_reference import (
    DEFAULT_BASE_SEED,
    DIAGNOSTIC_TOLERANCE_MEAN,
    DIAGNOSTIC_TOLERANCE_QUANTILE,
    MIN_REPLICATES_FOR_DIAGNOSTIC,
    get_layer_rng,
    run_extension2_pipeline,
    run_random_reference_simulation,
    sample_explicit_masks_overlap,
    sample_joint_hypergeometric_overlaps,
    simulate_single_replicate,
    validate_and_extract_ext1_data,
    verify_ext2_artifacts,
)


def _make_valid_ext1_fixtures():
    """Create minimal valid ext1 pairs and summary DataFrames for testing."""
    layer_sizes = {"fc1.weight": 235200, "fc2.weight": 30000, "fc3.weight": 1000}
    total_params = sum(layer_sizes.values())

    # Level 1: 80% remaining -> 0.8 * 266200 = 212960
    # Surviving: fc1: 188160, fc2: 24000, fc3: 800 -> sum = 212960
    surv_counts = {"fc1.weight": 188160, "fc2.weight": 24000, "fc3.weight": 800}
    weights_remaining = sum(surv_counts.values()) / total_params

    pairs_list = [(1, 2), (1, 42), (2, 42)]
    pair_rows = []

    for s_a, s_b in pairs_list:
        # overall
        pair_rows.append({
            "dataset": "mnist",
            "model": "lenet300100",
            "level": 1,
            "weights_remaining": weights_remaining,
            "seed_a": s_a,
            "seed_b": s_b,
            "scope": "overall",
            "surviving_a": sum(surv_counts.values()),
            "surviving_b": sum(surv_counts.values()),
            "intersection": 170000,
            "union": 255920,
            "jaccard": 170000 / 255920,
        })
        for lyr in ["fc1.weight", "fc2.weight", "fc3.weight"]:
            k = surv_counts[lyr]
            inter = int(round(k * 0.8))
            uni = 2 * k - inter
            pair_rows.append({
                "dataset": "mnist",
                "model": "lenet300100",
                "level": 1,
                "weights_remaining": weights_remaining,
                "seed_a": s_a,
                "seed_b": s_b,
                "scope": lyr,
                "surviving_a": k,
                "surviving_b": k,
                "intersection": inter,
                "union": uni,
                "jaccard": inter / uni,
            })

    df_pairs = pd.DataFrame(pair_rows)

    summary_rows = []
    for scp in ["overall", "fc1.weight", "fc2.weight", "fc3.weight"]:
        sub = df_pairs[df_pairs["scope"] == scp]
        summary_rows.append({
            "dataset": "mnist",
            "model": "lenet300100",
            "level": 1,
            "weights_remaining": weights_remaining,
            "scope": scp,
            "mean": float(sub["jaccard"].mean()),
            "std": float(sub["jaccard"].std(ddof=1)),
            "min": float(sub["jaccard"].min()),
            "max": float(sub["jaccard"].max()),
            "n_pairs": len(sub),
        })
    df_summary = pd.DataFrame(summary_rows)

    return df_pairs, df_summary


def test_hypergeometric_boundary_and_range_conditions():
    """Verify exact behavior at boundary conditions (k=n, k=0, and valid mathematical ranges)."""
    rng = np.random.default_rng(12345)

    # 1. Full dense: k = n
    res_dense_3 = sample_joint_hypergeometric_overlaps(100, 100, 3, rng)
    for p in [(0, 1), (0, 2), (1, 2)]:
        inter, uni = res_dense_3[p]
        assert inter == 100
        assert uni == 100

    # 2. Empty: k = 0
    res_empty_3 = sample_joint_hypergeometric_overlaps(100, 0, 3, rng)
    for p in [(0, 1), (0, 2), (1, 2)]:
        inter, uni = res_empty_3[p]
        assert inter == 0
        assert uni == 0

    # 3. Intermediate: check mathematical range bounds
    n, k = 1000, 300
    for _ in range(50):
        res = sample_joint_hypergeometric_overlaps(n, k, 3, rng)
        for p, (inter, uni) in res.items():
            assert 0 <= inter <= k
            assert uni == 2 * k - inter
            assert k <= uni <= n


def test_three_seed_joint_hypergeometric_matches_explicit_simulation_moments():
    """Verify that joint 3-seed hypergeometric sampling matches explicit boolean mask simulation across all 3 pairs."""
    n, k = 1000, 300
    replicates = 8000

    rng_hyp = np.random.default_rng(42)
    rng_exp = np.random.default_rng(42)

    hyp_01, hyp_02, hyp_12 = [], [], []
    for _ in range(replicates):
        res = sample_joint_hypergeometric_overlaps(n, k, 3, rng_hyp)
        hyp_01.append(res[(0, 1)][0])
        hyp_02.append(res[(0, 2)][0])
        hyp_12.append(res[(1, 2)][0])

    exp_01, exp_02, exp_12 = [], [], []
    for _ in range(replicates):
        res = sample_explicit_masks_overlap(n, k, 3, rng_exp)
        exp_01.append(res[(0, 1)][0])
        exp_02.append(res[(0, 2)][0])
        exp_12.append(res[(1, 2)][0])

    # Theoretical expected intersection: k^2 / n = 300 * 300 / 1000 = 90.0
    for h_list, e_list in [(hyp_01, exp_01), (hyp_02, exp_02), (hyp_12, exp_12)]:
        mean_h = float(np.mean(h_list))
        mean_e = float(np.mean(e_list))
        var_h = float(np.var(h_list, ddof=1))
        var_e = float(np.var(e_list, ddof=1))

        assert math.isclose(mean_h, 90.0, abs_tol=0.35)
        assert math.isclose(mean_e, 90.0, abs_tol=0.35)
        assert math.isclose(mean_h, mean_e, abs_tol=0.35)
        assert math.isclose(var_h, var_e, rel_tol=0.08)

    # Check pair-mean statistic across the 3 pairs
    hyp_pair_means = (np.array(hyp_01) + np.array(hyp_02) + np.array(hyp_12)) / 3.0
    exp_pair_means = (np.array(exp_01) + np.array(exp_02) + np.array(exp_12)) / 3.0

    assert math.isclose(float(np.mean(hyp_pair_means)), float(np.mean(exp_pair_means)), abs_tol=0.25)
    assert math.isclose(float(np.var(hyp_pair_means, ddof=1)), float(np.var(exp_pair_means, ddof=1)), rel_tol=0.08)


def test_layer_traversal_order_invariance():
    """Simulating layers in reverse order must produce bit-for-bit identical results due to layer-independent RNG."""
    layer_counts = {
        "fc1.weight": (235200, 47040),
        "fc2.weight": (30000, 6000),
        "fc3.weight": (1000, 200),
    }

    # Standard order
    res_normal = simulate_single_replicate(
        dataset_name="mnist",
        level=1,
        replicate_idx=42,
        layer_counts=layer_counts,
        unique_seeds=[1, 2, 42],
        base_seed=DEFAULT_BASE_SEED,
    )

    # Reversed layer dict
    reversed_counts = {
        "fc3.weight": (1000, 200),
        "fc2.weight": (30000, 6000),
        "fc1.weight": (235200, 47040),
    }
    res_reversed = simulate_single_replicate(
        dataset_name="mnist",
        level=1,
        replicate_idx=42,
        layer_counts=reversed_counts,
        unique_seeds=[1, 2, 42],
        base_seed=DEFAULT_BASE_SEED,
    )

    for scp in ["overall", "fc1.weight", "fc2.weight", "fc3.weight"]:
        assert res_normal[scp] == res_reversed[scp], f"Layer order invariance violated for {scp}"


def test_replicate_prefix_invariance():
    """Replicates 0..49 simulated with total=100 must be bit-for-bit identical to total=50."""
    df_pairs, df_summary = _make_valid_ext1_fixtures()

    _, sum_50 = run_random_reference_simulation(
        df_ext1_pairs=df_pairs,
        df_ext1_summary=df_summary,
        datasets=["mnist"],
        levels=[1],
        replicates=50,
        base_seed=DEFAULT_BASE_SEED,
    )
    samples_100, sum_100 = run_random_reference_simulation(
        df_ext1_pairs=df_pairs,
        df_ext1_summary=df_summary,
        datasets=["mnist"],
        levels=[1],
        replicates=100,
        base_seed=DEFAULT_BASE_SEED,
    )

    # Re-slice first 50 replicates from samples_100
    for scp in ["overall", "fc1.weight", "fc2.weight", "fc3.weight"]:
        s50 = samples_100[(samples_100["scope"] == scp) & (samples_100["replicate"] < 50)]["pair_mean_jaccard"].values
        calc_mean_50 = float(np.mean(s50))
        assert math.isclose(sum_50[sum_50["scope"] == scp]["reference_mean"].iloc[0], calc_mean_50, abs_tol=1e-12)


def test_overall_calculation_pools_intersections_and_unions_in_actual_simulation():
    """Verify that simulate_single_replicate pools intersections and unions before dividing."""
    # When layers have different sizes and different densities:
    # Layer 1 has 200,000 params, Layer 3 has 1,000 params
    layer_counts = {
        "fc1.weight": (235200, 200000),
        "fc2.weight": (30000, 3000),
        "fc3.weight": (1000, 100),
    }
    res = simulate_single_replicate(
        dataset_name="mnist",
        level=1,
        replicate_idx=0,
        layer_counts=layer_counts,
        unique_seeds=[1, 2, 42],
        base_seed=DEFAULT_BASE_SEED,
    )

    arithmetic_avg = (res["fc1.weight"] + res["fc2.weight"] + res["fc3.weight"]) / 3.0
    # Overall must be dominated by fc1.weight, NOT the simple average of layer Jaccards
    assert not math.isclose(res["overall"], arithmetic_avg, abs_tol=0.01)
    assert math.isclose(res["overall"], res["fc1.weight"], abs_tol=0.02)


def test_ext1_validation_rejects_missing_scope_or_duplicate():
    """validate_and_extract_ext1_data must reject missing layer scopes or duplicated rows."""
    df_pairs, df_summary = _make_valid_ext1_fixtures()

    # 1. Drop fc3.weight
    bad_pairs = df_pairs[df_pairs["scope"] != "fc3.weight"].copy()
    with pytest.raises(ValueError, match="missing scopes"):
        validate_and_extract_ext1_data(bad_pairs, df_summary)

    # 2. Duplicate a row
    dup_pairs = pd.concat([df_pairs, df_pairs.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate records"):
        validate_and_extract_ext1_data(dup_pairs, df_summary)


def test_ext1_validation_rejects_summary_mismatch():
    """validate_and_extract_ext1_data must reject summary values that disagree with pairs table."""
    df_pairs, df_summary = _make_valid_ext1_fixtures()

    # Tamper with summary mean
    bad_summary = df_summary.copy()
    bad_summary.loc[0, "mean"] = 0.999999
    with pytest.raises(ValueError, match="Summary mean mismatch"):
        validate_and_extract_ext1_data(df_pairs, bad_summary)


def test_stability_diagnostic_status_under_2000_replicates():
    """Simulations with < 2000 replicates must mark diagnostic_status='not_evaluated' and is_stable=None."""
    df_pairs, df_summary = _make_valid_ext1_fixtures()

    _, df_sum = run_random_reference_simulation(
        df_ext1_pairs=df_pairs,
        df_ext1_summary=df_summary,
        datasets=["mnist"],
        levels=[1],
        replicates=500,
        base_seed=DEFAULT_BASE_SEED,
    )

    for _, row in df_sum.iterrows():
        assert row["diagnostic_status"] == "not_evaluated"
        assert pd.isna(row["is_stable"])
        assert math.isnan(row["diag_mean_diff"])
        assert math.isnan(row["diag_p025_diff"])
        assert math.isnan(row["diag_p975_diff"])


def test_stability_diagnostic_status_at_2000_replicates():
    """Simulations with >= 2000 replicates must evaluate stability diagnostics."""
    df_pairs, df_summary = _make_valid_ext1_fixtures()

    _, df_sum = run_random_reference_simulation(
        df_ext1_pairs=df_pairs,
        df_ext1_summary=df_summary,
        datasets=["mnist"],
        levels=[1],
        replicates=2000,
        base_seed=DEFAULT_BASE_SEED,
    )

    for _, row in df_sum.iterrows():
        assert row["diagnostic_status"] == "evaluated"
        assert row["is_stable"] in [True, False]
        assert not math.isnan(row["diag_mean_diff"])
        assert not math.isnan(row["diag_p025_diff"])
        assert not math.isnan(row["diag_p975_diff"])


def test_verify_ext2_artifacts_detects_all_categories_of_tampering(tmp_path: Path):
    """verify_ext2_artifacts must distinguish and detect missing inputs and corrupted outputs."""
    exec_dir = tmp_path / "ext2_test_audit"
    exec_dir.mkdir(parents=True)
    tables_dir = exec_dir / "tables"
    figures_dir = exec_dir / "figures"
    tables_dir.mkdir()
    figures_dir.mkdir()

    ext1_dir = tmp_path / "ext1_dummy"
    ext1_dir.mkdir()

    # Valid tables with required columns
    (tables_dir / "random_reference_samples.csv").write_text(
        "dataset,level,scope,pair_mean_jaccard\nmnist,1,overall,0.1\n"
    )
    (tables_dir / "random_reference_summary.csv").write_text(
        "dataset,model,level,scope,observed_mean,reference_mean\nmnist,lenet,1,overall,0.2,0.1\n"
    )
    for fig_name in [
        "random_reference_grid.png",
        "random_reference_grid.pdf",
        "random_reference_diff.png",
        "random_reference_diff.pdf",
    ]:
        (figures_dir / fig_name).write_text("fig_data")

    meta = {
        "source_extension1_path": str(ext1_dir),
        "inputs": {},
    }
    (exec_dir / "metadata.json").write_text(json.dumps(meta))

    # 1. Clean verify
    ok, errors = verify_ext2_artifacts(exec_dir, expected_ext1_dir=ext1_dir)
    assert ok
    assert not errors["inputs"] and not errors["code"] and not errors["outputs"]

    # 2. Source extension 1 path mismatch
    ok, errors = verify_ext2_artifacts(exec_dir, expected_ext1_dir=tmp_path / "nonexistent")
    assert not ok
    assert len(errors["inputs"]) > 0

    # 3. Missing output table
    (tables_dir / "random_reference_summary.csv").unlink()
    ok, errors = verify_ext2_artifacts(exec_dir, expected_ext1_dir=ext1_dir)
    assert not ok
    assert len(errors["outputs"]) > 0


def test_ext2_overwrite_protection(tmp_path: Path):
    """Pipeline must refuse to overwrite an existing extension 2 directory."""
    existing_dir = tmp_path / "ext2_test_existing"
    existing_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        run_extension2_pipeline(
            ext1_execution_dir="results/extensions/ext1_20261108_233054_662fcc",
            execution_id="ext2_test_existing",
            out_base=str(tmp_path),
        )
