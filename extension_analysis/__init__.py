"""Extension 1: Cross-seed mask stability analysis for the Lottery Ticket Hypothesis."""

from extension_analysis.mask_stability import (
    compute_layer_jaccard,
    compute_overall_jaccard,
    compare_seed_pair_masks,
    aggregate_jaccard_summary,
    audit_all_runs,
    verify_execution_artifacts,
    run_extension1_pipeline,
)
from extension_analysis.plots import (
    plot_mask_stability_grid,
    plot_random_reference_grid,
    plot_random_reference_difference,
    plot_prediction_agreement_curves,
    plot_structure_function_scatter,
)
from extension_analysis.random_reference import (
    sample_joint_hypergeometric_overlaps,
    sample_explicit_masks_overlap,
    run_random_reference_simulation,
    verify_ext2_artifacts,
    run_extension2_pipeline,
)
from extension_analysis.prediction_agreement import (
    load_test_dataset,
    compute_dataset_fingerprint,
    evaluate_checkpoint_predictions,
    compute_pair_agreement,
    aggregate_agreement_summary,
    link_structure_and_function,
    verify_ext3_artifacts,
    run_extension3_pipeline,
)

__all__ = [
    "compute_layer_jaccard",
    "compute_overall_jaccard",
    "compare_seed_pair_masks",
    "aggregate_jaccard_summary",
    "audit_all_runs",
    "verify_execution_artifacts",
    "run_extension1_pipeline",
    "plot_mask_stability_grid",
    "plot_random_reference_grid",
    "plot_random_reference_difference",
    "sample_joint_hypergeometric_overlaps",
    "sample_explicit_masks_overlap",
    "run_random_reference_simulation",
    "verify_ext2_artifacts",
    "run_extension2_pipeline",
    "load_test_dataset",
    "compute_dataset_fingerprint",
    "evaluate_checkpoint_predictions",
    "compute_pair_agreement",
    "aggregate_agreement_summary",
    "link_structure_and_function",
    "verify_ext3_artifacts",
    "run_extension3_pipeline",
    "plot_prediction_agreement_curves",
    "plot_structure_function_scatter",
]

