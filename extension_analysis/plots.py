"""Plotting routines for mask stability analysis."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Styling constants aligned with analysis.ipynb
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

PAIR_COLORS = {
    (1, 2): "#7da0fa",
    (1, 42): "#f7a379",
    (2, 42): "#7bd4b4",
}
PAIR_LABELS = {
    (1, 2): "Seed 1 vs 2",
    (1, 42): "Seed 1 vs 42",
    (2, 42): "Seed 2 vs 42",
}

SCOPE_TITLES = {
    "overall": "Overall (All Layers)",
    "fc1.weight": "FC1 (300 × 784)",
    "fc2.weight": "FC2 (100 × 300)",
    "fc3.weight": "FC3 (10 × 100)",
}

DATASET_TITLES = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
}


def plot_mask_stability_grid(
    df_summary: pd.DataFrame,
    df_pairs: pd.DataFrame,
    save_png: Optional[Path] = None,
    save_pdf: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Generate 2x4 grid plot of Jaccard similarity vs weights remaining.

    Row 0: MNIST
    Row 1: Fashion-MNIST
    Col 0: Overall, Col 1: FC1, Col 2: FC2, Col 3: FC3
    Excludes Level 0 (dense, Jaccard=1.0) from the main sparse stability curves.
    """
    rc_params = {
        "figure.dpi": 110,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    }

    datasets = ["mnist", "fashion_mnist"]
    scopes = ["overall", "fc1.weight", "fc2.weight", "fc3.weight"]

    # Filter out Level 0 for visualization
    pairs_sub = df_pairs[df_pairs["level"] > 0].copy()
    summary_sub = df_summary[df_summary["level"] > 0].copy()

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=4,
            figsize=(14, 6.5),
            sharey=True,
            sharex=False,
        )

        for row_idx, ds in enumerate(datasets):
            ds_name = DATASET_TITLES.get(ds, ds)
            for col_idx, scp in enumerate(scopes):
                ax = axes[row_idx, col_idx]

                # 1. Plot individual pairs (pale curves)
                pair_group = pairs_sub[
                    (pairs_sub["dataset"] == ds) & (pairs_sub["scope"] == scp)
                ]
                for (s_a, s_b), grp in pair_group.groupby(["seed_a", "seed_b"]):
                    grp_sorted = grp.sort_values(by="weights_remaining", ascending=False)
                    pair_key = (min(s_a, s_b), max(s_a, s_b))
                    p_color = PAIR_COLORS.get(pair_key, "#a0a0a0")
                    p_label = PAIR_LABELS.get(pair_key, f"Seeds {s_a} vs {s_b}")
                    ax.plot(
                        grp_sorted["weights_remaining"] * 100,
                        grp_sorted["jaccard"],
                        color=p_color,
                        linewidth=1.2,
                        alpha=0.7,
                        linestyle="--",
                        marker=".",
                        markersize=3.5,
                        label=p_label if (row_idx == 0 and col_idx == 0) else None,
                    )

                # 2. Plot mean curve (prominent solid line)
                sum_group = summary_sub[
                    (summary_sub["dataset"] == ds) & (summary_sub["scope"] == scp)
                ]
                sum_sorted = sum_group.sort_values(
                    by="weights_remaining", ascending=False
                )
                ax.plot(
                    sum_sorted["weights_remaining"] * 100,
                    sum_sorted["mean"],
                    color=INK,
                    linewidth=2.0,
                    marker="o",
                    markersize=4.0,
                    label="Mean across pairs" if (row_idx == 0 and col_idx == 0) else None,
                    zorder=4,
                )

                # Configure axes
                ax.set_xscale("log")
                ax.invert_xaxis()
                ax.set_ylim(-0.02, 1.02)

                # Set custom x-ticks for percentage
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda y, _: f"{y:g}%")
                )
                ticks = [100, 50, 20, 10, 5]
                ax.set_xticks(ticks)
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))

                # Titles and labels
                if row_idx == 0:
                    ax.set_title(SCOPE_TITLES.get(scp, scp), pad=8, weight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"{ds_name}\nJaccard Similarity", color=INK)
                if row_idx == 1:
                    ax.set_xlabel("Weights remaining (log scale)")

                # Annotation of coverage
                n_pairs_actual = (
                    sum_sorted["n_pairs"].min() if not sum_sorted.empty else 0
                )
                exp_pairs = (
                    sum_sorted["expected_pairs"].iloc[0]
                    if not sum_sorted.empty
                    else 3
                )
                ax.text(
                    0.05,
                    0.05,
                    f"Coverage: {n_pairs_actual}/{exp_pairs} pairs",
                    transform=ax.transAxes,
                    fontsize=7.5,
                    color=MUTED,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor=GRID,
                        alpha=0.8,
                    ),
                )

        # Single legend for the entire figure at top
        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=4,
            frameon=False,
            fontsize=9,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_png:
            Path(save_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_png, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            Path(save_pdf).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_pdf, bbox_inches="tight")

    return fig

