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
    plot_layer_sparsity_curves,
    plot_layer_share_evolution,
    plot_pruning_schedule_adherence,
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
from extension_analysis.layer_sparsity import (
    compute_layer_sparsity_records,
    aggregate_layer_sparsity_summary,
    create_layer_allocation_summary,
    verify_ext4_artifacts,
    run_extension4_pipeline,
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
    "compute_layer_sparsity_records",
    "aggregate_layer_sparsity_summary",
    "create_layer_allocation_summary",
    "verify_ext4_artifacts",
    "run_extension4_pipeline",
    "plot_layer_sparsity_curves",
    "plot_layer_share_evolution",
    "plot_pruning_schedule_adherence",
]

