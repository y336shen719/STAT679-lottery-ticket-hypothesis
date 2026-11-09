"""Unit tests for Extension 3: Cross-seed Prediction Agreement Analysis."""

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from extension_analysis.prediction_agreement import (
    compute_dataset_fingerprint,
    load_test_dataset,
    compute_pair_agreement,
    aggregate_agreement_summary,
    link_structure_and_function,
    verify_checkpoint_and_best_state,
    evaluate_checkpoint_predictions,
    verify_ext3_artifacts,
    run_extension3_pipeline,
    DatasetNotAvailableError,
    DatasetCorruptError,
    IncompatibleCheckpointError,
    DEFAULT_ACCURACY_TOLERANCE,
)
from extension_analysis.plots import (
    plot_prediction_agreement_curves,
    plot_structure_function_scatter,
)
from lth.models import make_model
from lth.pruning import initial_masks


def test_dataset_fingerprint_chunked_and_tail_sensitive():
    """Verify chunked fingerprint is reproducible and sensitive to alterations in the final sample."""
    x = torch.randn(100, 1, 28, 28, dtype=torch.float32)
    y = torch.randint(0, 10, (100,), dtype=torch.int64)

    fp1 = compute_dataset_fingerprint(x, y)
    fp2 = compute_dataset_fingerprint(x.clone(), y.clone())
    assert fp1 == fp2

    # Alter single value in the very last sample of x
    x_tail = x.clone()
    x_tail[-1, 0, -1, -1] += 0.001
    assert compute_dataset_fingerprint(x_tail, y) != fp1

    # Alter single label in the very last entry of y
    y_tail = y.clone()
    y_tail[-1] = (y_tail[-1] + 1) % 10
    assert compute_dataset_fingerprint(x, y_tail) != fp1


def test_missing_dataset_raises_dataset_not_available(tmp_path):
    """Enforces download=False: missing local dataset raises DatasetNotAvailableError."""
    with pytest.raises(DatasetNotAvailableError) as exc_info:
        load_test_dataset("fashion_mnist", data_root=tmp_path)
    assert "fashion_mnist" in str(exc_info.value)
    assert "download=False" in str(exc_info.value)


def test_broadcasting_risk_rejected_by_1d_check():
    """Verify that (N, 1) vs (N,) broadcasting is strictly rejected."""
    targets = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    preds_1d = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    preds_2d = preds_1d.reshape(-1, 1)

    # preds_a is (5, 1) instead of (5,)
    with pytest.raises(ValueError, match="strictly 1-dimensional"):
        compute_pair_agreement(preds_2d, preds_1d, targets)

    # preds_b is (5, 1)
    with pytest.raises(ValueError, match="strictly 1-dimensional"):
        compute_pair_agreement(preds_1d, preds_2d, targets)

    # targets is (5, 1)
    with pytest.raises(ValueError, match="strictly 1-dimensional"):
        compute_pair_agreement(preds_1d, preds_1d, preds_2d)


def test_mismatched_ids_targets_or_fingerprints_rejected():
    """Verify pair agreement rejects condition mismatches, target discrepancies, or fingerprint divergence."""
    preds_a = np.array([0, 1, 2], dtype=np.int64)
    preds_b = np.array([0, 1, 2], dtype=np.int64)
    targets = np.array([0, 1, 2], dtype=np.int64)
    example_ids = np.array([0, 1, 2], dtype=np.int64)

    meta_a = {
        "dataset": "mnist",
        "model": "lenet300100",
        "level": 1,
        "checkpoint_kind": "best_state",
        "data_fingerprint": "fp_alpha",
        "targets": targets,
        "example_ids": example_ids,
    }
    meta_b = dict(meta_a)

    # Clean comparison passes
    res = compute_pair_agreement(preds_a, preds_b, targets, example_ids=example_ids, meta_a=meta_a, meta_b=meta_b)
    assert res["agreement"] == 1.0

    # Fingerprint mismatch
    meta_b_bad_fp = dict(meta_a, data_fingerprint="fp_beta")
    with pytest.raises(ValueError, match="Data fingerprint mismatch"):
        compute_pair_agreement(preds_a, preds_b, targets, meta_a=meta_a, meta_b=meta_b_bad_fp)

    # Targets mismatch
    meta_b_bad_targets = dict(meta_a, targets=np.array([0, 1, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="targets do not match"):
        compute_pair_agreement(preds_a, preds_b, targets, meta_a=meta_a, meta_b=meta_b_bad_targets)

    # Example IDs mismatch
    meta_b_bad_ids = dict(meta_a, example_ids=np.array([10, 11, 12], dtype=np.int64))
    with pytest.raises(ValueError, match="example_ids mismatch"):
        compute_pair_agreement(preds_a, preds_b, targets, meta_a=meta_a, meta_b=meta_b_bad_ids)

    # Checkpoint kind mismatch
    meta_b_end = dict(meta_a, checkpoint_kind="end_state")
    with pytest.raises(ValueError, match="best_state"):
        compute_pair_agreement(preds_a, preds_b, targets, meta_a=meta_a, meta_b=meta_b_end)


def test_rejection_of_non_finite_weights_and_non_zero_pruned_weights(tmp_path):
    """Verify strict rejection of NaN/Inf in parameters and non-zero entries in pruned positions."""
    model = make_model("lenet300100")
    masks = initial_masks(model)

    run_dir = tmp_path / "mnist_adam_seed1"
    run_dir.mkdir(parents=True)
    cfg = {
        "dataset": "mnist",
        "seed": 1,
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
    (run_dir / "config.json").write_text(json.dumps(cfg))
    (run_dir / "summary.csv").write_text("level,weights_remaining\n0,1.0\n1,0.8\n")

    # 1. Non-finite weight
    bad_state_nan = model.state_dict()
    bad_state_nan["fc1.weight"][0, 0] = float("nan")
    ckpt_nan = {
        "best_state": bad_state_nan,
        "masks": masks,
        "metrics": {"level": 0, "weights_remaining": 1.0, "best_test_accuracy": 0.5},
    }
    ckpt_path_nan = run_dir / "nan_ckpt.pt"
    torch.save(ckpt_nan, ckpt_path_nan)

    x_test = torch.randn(10, 1, 28, 28)
    y_test = torch.randint(0, 10, (10,))

    with pytest.raises(IncompatibleCheckpointError, match="Non-finite"):
        evaluate_checkpoint_predictions(
            ckpt_path=ckpt_path_nan,
            x_test=x_test,
            y_test=y_test,
            expected_dataset="mnist",
            expected_seed=1,
            expected_level=0,
            runs_dir=tmp_path,
            results_dir=None,
        )

    # 2. Non-zero pruned weight
    masks_pruned = initial_masks(model)
    masks_pruned["fc1.weight"][0, 0] = False
    state_nonzero = model.state_dict()
    state_nonzero["fc1.weight"][0, 0] = 0.42

    ckpt_nonzero = {
        "best_state": state_nonzero,
        "masks": masks_pruned,
        "metrics": {"level": 1, "weights_remaining": 0.8, "best_test_accuracy": 0.5},
    }
    ckpt_path_nz = run_dir / "nz_ckpt.pt"
    torch.save(ckpt_nonzero, ckpt_path_nz)

    with pytest.raises(IncompatibleCheckpointError, match="Pruned weights non-zero"):
        evaluate_checkpoint_predictions(
            ckpt_path=ckpt_path_nz,
            x_test=x_test,
            y_test=y_test,
            expected_dataset="mnist",
            expected_seed=1,
            expected_level=1,
            runs_dir=tmp_path,
            results_dir=None,
        )


def test_one_sample_tolerance_boundary_handling(tmp_path):
    """Verify that recomputed accuracy allows <= 1 sample divergence and rejects > 1 sample divergence."""
    model = make_model("lenet300100")
    masks = initial_masks(model)

    x_test = torch.randn(10000, 1, 28, 28)
    y_test = torch.randint(0, 10, (10000,))

    model.eval()
    with torch.inference_mode():
        logits = model(x_test)
        preds = logits.argmax(dim=1)
        exact_acc = float((preds == y_test).float().mean().item())

    # Create dummy run config and summary in tmp_path
    run_dir = tmp_path / "mnist_adam_seed1"
    run_dir.mkdir(parents=True)
    cfg = {
        "dataset": "mnist",
        "seed": 1,
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
    (run_dir / "config.json").write_text(json.dumps(cfg))
    (run_dir / "summary.csv").write_text(f"level,weights_remaining\n0,1.0\n")

    # 1. Exactly 1 sample deviation (1e-4) -> allowed
    ckpt_1sample = {
        "best_state": model.state_dict(),
        "masks": masks,
        "metrics": {"level": 0, "weights_remaining": 1.0, "best_test_accuracy": exact_acc + 0.0001},
    }
    ckpt_path_1s = run_dir / "level_00.pt"
    torch.save(ckpt_1sample, ckpt_path_1s)

    res = evaluate_checkpoint_predictions(
        ckpt_path=ckpt_path_1s,
        x_test=x_test,
        y_test=y_test,
        expected_dataset="mnist",
        expected_seed=1,
        expected_level=0,
        runs_dir=tmp_path,
        results_dir=None,
        accuracy_tolerance=1e-4,
    )
    assert res["is_within_tolerance"]

    # 2. 2 sample deviation (2e-4) -> rejected
    ckpt_2sample = {
        "best_state": model.state_dict(),
        "masks": masks,
        "metrics": {"level": 0, "weights_remaining": 1.0, "best_test_accuracy": exact_acc + 0.0002},
    }
    torch.save(ckpt_2sample, ckpt_path_1s)

    with pytest.raises(IncompatibleCheckpointError, match="exceeds 1-sample tolerance"):
        evaluate_checkpoint_predictions(
            ckpt_path=ckpt_path_1s,
            x_test=x_test,
            y_test=y_test,
            expected_dataset="mnist",
            expected_seed=1,
            expected_level=0,
            runs_dir=tmp_path,
            results_dir=None,
            accuracy_tolerance=1e-4,
        )


def test_single_pair_sd_is_nan_and_unique_model_aggregation():
    """Verify that n_pairs=1 produces NaN sample SD, and model accuracy aggregates unique models."""
    # df_accuracy contains 3 unique models (seeds 1, 2, 42)
    df_accuracy = pd.DataFrame(
        [
            {"dataset": "mnist", "model": "lenet300100", "seed": 1, "level": 1, "weights_remaining": 0.8, "recomputed_accuracy": 0.98, "audit_status": "evaluated"},
            {"dataset": "mnist", "model": "lenet300100", "seed": 2, "level": 1, "weights_remaining": 0.8, "recomputed_accuracy": 0.96, "audit_status": "evaluated"},
            {"dataset": "mnist", "model": "lenet300100", "seed": 42, "level": 1, "weights_remaining": 0.8, "recomputed_accuracy": 0.94, "audit_status": "incompatible_checkpoint: corrupted"},
        ]
    )

    # df_pairs only has 1 pair because seed 42 was incompatible
    df_pairs = pd.DataFrame(
        [
            {
                "dataset": "mnist",
                "model": "lenet300100",
                "level": 1,
                "weights_remaining": 0.8,
                "seed_a": 1,
                "seed_b": 2,
                "agreement": 0.98,
                "shared_correct": 0.96,
                "shared_wrong": 0.02,
                "accuracy_a": 0.98,
                "accuracy_b": 0.96,
            }
        ]
    )

    summary = aggregate_agreement_summary(df_pairs, df_accuracy)
    assert len(summary) == 1
    row = summary.iloc[0]

    # Pair metrics: n_pairs=1 -> SD must be NaN
    assert row["n_pairs"] == 1
    assert row["agreement_mean"] == pytest.approx(0.98)
    assert math.isnan(row["agreement_sd"])
    assert math.isnan(row["shared_correct_sd"])
    assert math.isnan(row["shared_wrong_sd"])

    # Model accuracy: aggregated over the 2 evaluated models (seeds 1 and 2), ignoring incompatible seed 42
    assert row["n_models"] == 2
    assert row["accuracy_mean"] == pytest.approx(0.97)  # mean(0.98, 0.96)
    assert row["accuracy_sd"] == pytest.approx(float(np.std([0.98, 0.96], ddof=1)))


def test_structure_function_linking_rejects_duplicates_and_dropped_keys(tmp_path):
    """Verify link_structure_and_function validates one-to-one mapping and rejects duplicate keys."""
    ext1_pairs_file = tmp_path / "mask_jaccard_pairs.csv"
    ext1_data = pd.DataFrame(
        [
            {"dataset": "mnist", "model": "lenet300100", "level": 1, "weights_remaining": 0.8, "seed_a": 1, "seed_b": 2, "scope": "overall", "jaccard": 0.695},
            # Duplicate key in ext1
            {"dataset": "mnist", "model": "lenet300100", "level": 1, "weights_remaining": 0.8, "seed_a": 1, "seed_b": 2, "scope": "overall", "jaccard": 0.690},
        ]
    )
    ext1_data.to_csv(ext1_pairs_file, index=False)

    df_pairs = pd.DataFrame(
        [
            {"dataset": "mnist", "model": "lenet300100", "level": 1, "weights_remaining": 0.8, "seed_a": 1, "seed_b": 2, "agreement": 0.98, "shared_correct": 0.97, "shared_wrong": 0.01, "accuracy_a": 0.98, "accuracy_b": 0.97}
        ]
    )

    with pytest.raises(ValueError, match="Duplicate keys"):
        link_structure_and_function(df_pairs, ext1_pairs_file)


def test_pipeline_overwrite_protection(tmp_path):
    """Pipeline must refuse to overwrite an existing extension directory."""
    existing_dir = tmp_path / "ext3_mock_exist"
    existing_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        run_extension3_pipeline(
            ext1_dir=Path("results/extensions/ext1_20261108_233054_662fcc"),
            output_parent=tmp_path,
            execution_id="ext3_mock_exist",
        )


def test_ext3_artifacts_verification_and_tamper_detection(tmp_path):
    """Verify verify_ext3_artifacts validates integrity and categorizes errors correctly."""
    exec_dir = tmp_path / "ext3_mock_exec"
    tables_dir = exec_dir / "tables"
    figures_dir = exec_dir / "figures"
    predictions_dir = exec_dir / "predictions"
    tables_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    from extension_analysis.mask_stability import compute_file_sha256

    # Fake ext1 dir
    ext1_dir = tmp_path / "ext1_mock"
    ext1_tables = ext1_dir / "tables"
    ext1_tables.mkdir(parents=True)
    ext1_pairs = ext1_tables / "mask_jaccard_pairs.csv"
    ext1_sum = ext1_tables / "mask_jaccard_summary.csv"
    ext1_meta = ext1_dir / "metadata.json"
    ext1_pairs.write_text("ext1_pairs")
    ext1_sum.write_text("ext1_sum")
    ext1_meta.write_text("{}")

    ext1_hashes = {
        "tables/mask_jaccard_pairs.csv": compute_file_sha256(ext1_pairs),
        "tables/mask_jaccard_summary.csv": compute_file_sha256(ext1_sum),
        "metadata.json": compute_file_sha256(ext1_meta),
    }

    # Fake code file
    code_file = tmp_path / "fake_code.py"
    code_file.write_text("print('code')")
    code_hashes = {str(code_file): compute_file_sha256(code_file)}

    # Fake output tables
    t1 = tables_dir / "prediction_inventory.csv"
    t2 = tables_dir / "prediction_agreement_pairs.csv"
    t3 = tables_dir / "prediction_agreement_summary.csv"
    t4 = tables_dir / "structure_function_pairs.csv"
    t5 = tables_dir / "prediction_accuracy.csv"
    for t in [t1, t2, t3, t4, t5]:
        t.write_text("table_data")

    table_hashes = {
        f"tables/{t.name}": compute_file_sha256(t) for t in [t1, t2, t3, t4, t5]
    }

    # Fake output figures
    f1 = figures_dir / "fig_ext3_prediction_agreement.png"
    f2 = figures_dir / "fig_ext3_prediction_agreement.pdf"
    f3 = figures_dir / "fig_ext3_structure_function.png"
    f4 = figures_dir / "fig_ext3_structure_function.pdf"
    for f in [f1, f2, f3, f4]:
        f.write_text("fig_data")

    fig_hashes = {
        f"figures/{f.name}": compute_file_sha256(f) for f in [f1, f2, f3, f4]
    }

    # Fake prediction cache
    p1 = predictions_dir / "mnist_lenet300100_seed1_level00.npz"
    p1.write_text("npz_data")
    pred_hashes = {f"predictions/{p1.name}": compute_file_sha256(p1)}

    meta = {
        "extension1_source": {
            "path": str(ext1_dir),
            "table_hashes": ext1_hashes,
        },
        "input_checkpoint_hashes": {},
        "config_hashes": {},
        "summary_hashes": {},
        "code_file_hashes": code_hashes,
        "prediction_cache_hashes": pred_hashes,
        "output_table_hashes": table_hashes,
        "output_figure_hashes": fig_hashes,
    }
    with open(exec_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # 1. Clean verification
    ok, errors = verify_ext3_artifacts(exec_dir, expected_ext1_dir=ext1_dir)
    assert ok
    assert not errors["inputs"] and not errors["code"] and not errors["outputs"]

    # 2. Tampered output table -> outputs error
    t1.write_text("tampered_table")
    ok, errors = verify_ext3_artifacts(exec_dir, expected_ext1_dir=ext1_dir)
    assert not ok
    assert len(errors["outputs"]) > 0

    # 3. Tampered ext1 table -> inputs error
    ext1_pairs.write_text("tampered_ext1")
    ok, errors = verify_ext3_artifacts(exec_dir, expected_ext1_dir=ext1_dir)
    assert not ok
    assert len(errors["inputs"]) > 0


def test_results_summary_audit_strict_rejection(tmp_path):
    """Verify results summary strict parsing: duplicate level, invalid value, or ckpt mismatch."""
    model = make_model("lenet300100")
    masks = initial_masks(model)

    run_dir = tmp_path / "runs" / "mnist_adam_seed1"
    run_dir.mkdir(parents=True)
    res_dir = tmp_path / "results" / "mnist_adam_seed1"
    res_dir.mkdir(parents=True)

    cfg = {
        "dataset": "mnist",
        "seed": 1,
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
    (run_dir / "config.json").write_text(json.dumps(cfg))
    (res_dir / "config.json").write_text(json.dumps(cfg))
    (run_dir / "summary.csv").write_text("level,weights_remaining\n0,1.0\n")

    ckpt = {
        "best_state": model.state_dict(),
        "masks": masks,
        "metrics": {"level": 0, "weights_remaining": 1.0, "best_test_accuracy": 0.98},
    }
    ckpt_path = run_dir / "level_00.pt"
    torch.save(ckpt, ckpt_path)

    # 1. Duplicate level in results summary
    (res_dir / "summary.csv").write_text("level,best_test_accuracy\n0,0.98\n0,0.98\n")
    with pytest.raises(IncompatibleCheckpointError, match="Duplicate levels"):
        verify_checkpoint_and_best_state(
            ckpt_path=ckpt_path,
            expected_dataset="mnist",
            expected_seed=1,
            expected_level=0,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
        )

    # 2. Invalid / non-finite accuracy
    (res_dir / "summary.csv").write_text("level,best_test_accuracy\n0,NaN\n")
    with pytest.raises(IncompatibleCheckpointError, match="Out-of-range or non-finite"):
        verify_checkpoint_and_best_state(
            ckpt_path=ckpt_path,
            expected_dataset="mnist",
            expected_seed=1,
            expected_level=0,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
        )

    # 3. Checkpoint accuracy differs from results summary
    (res_dir / "summary.csv").write_text("level,best_test_accuracy\n0,0.95\n")
    with pytest.raises(IncompatibleCheckpointError, match="differs from results summary"):
        verify_checkpoint_and_best_state(
            ckpt_path=ckpt_path,
            expected_dataset="mnist",
            expected_seed=1,
            expected_level=0,
            runs_dir=tmp_path / "runs",
            results_dir=tmp_path / "results",
        )


def test_extension1_checkpoint_hash_mismatch_rejected(tmp_path):
    """Verify run_extension3_pipeline rejects Extension 1 sources with mismatched checkpoint hashes."""
    ext1_dir = tmp_path / "ext1_test"
    ext1_dir.mkdir(parents=True)
    ext1_tables = ext1_dir / "tables"
    ext1_tables.mkdir(parents=True)
    (ext1_tables / "mask_jaccard_pairs.csv").write_text("dataset,model,level,weights_remaining,seed_a,seed_b,scope,jaccard\n")
    (ext1_tables / "mask_jaccard_summary.csv").write_text("dataset,model,level,weights_remaining,scope,mean,std,min,max,n_pairs,expected_pairs\n")

    # Ext1 inventory with bogus hash
    (ext1_dir / "artifact_inventory.csv").write_text(
        "dataset,seed,level,expected_path,exists,sha256,is_valid\n"
        "mnist,1,0,runs/mnist_adam_seed1/level_00.pt,True,bogus_hash_123,True\n"
    )

    # Create dummy metadata for ext1
    meta = {
        "execution_id": "ext1_test",
        "inputs": {},
    }
    (ext1_dir / "metadata.json").write_text(json.dumps(meta))

    with pytest.raises(ValueError, match="Extension 1 source execution verification failed"):
        run_extension3_pipeline(
            ext1_dir=ext1_dir,
            output_parent=tmp_path / "ext3_out",
        )

