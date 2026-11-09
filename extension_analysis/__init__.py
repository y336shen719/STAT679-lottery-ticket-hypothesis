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
from extension_analysis.plots import plot_mask_stability_grid

__all__ = [
    "compute_layer_jaccard",
    "compute_overall_jaccard",
    "compare_seed_pair_masks",
    "aggregate_jaccard_summary",
    "audit_all_runs",
    "verify_execution_artifacts",
    "run_extension1_pipeline",
    "plot_mask_stability_grid",
]

