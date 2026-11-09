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
)
from extension_analysis.random_reference import (
    sample_joint_hypergeometric_overlaps,
    sample_explicit_masks_overlap,
    run_random_reference_simulation,
    verify_ext2_artifacts,
    run_extension2_pipeline,
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
]

