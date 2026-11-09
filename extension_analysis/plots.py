"""Plotting routines for mask stability and random reference analysis."""

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
ACCENT_BLUE = "#2a78d6"
ACCENT_ORANGE = "#eb6834"

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


def _get_base_rc_params() -> dict:
    return {
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
    rc_params = _get_base_rc_params()
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


def plot_random_reference_grid(
    df_summary: pd.DataFrame,
    save_png: Optional[Path] = None,
    save_pdf: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Generate 2x4 grid plot comparing observed Jaccard vs. density-matched random reference.

    Row 0: MNIST, Row 1: Fashion-MNIST.
    Col 0: Overall, Col 1: FC1, Col 2: FC2, Col 3: FC3.
    Level 0 is excluded from sparse curves.
    """
    rc_params = _get_base_rc_params()
    datasets = ["mnist", "fashion_mnist"]
    scopes = ["overall", "fc1.weight", "fc2.weight", "fc3.weight"]

    # Filter out Level 0
    sub_df = df_summary[df_summary["level"] > 0].copy()

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=4,
            figsize=(14, 6.8),
            sharey=True,
            sharex=False,
        )

        for row_idx, ds in enumerate(datasets):
            ds_name = DATASET_TITLES.get(ds, ds)
            for col_idx, scp in enumerate(scopes):
                ax = axes[row_idx, col_idx]

                grp = sub_df[(sub_df["dataset"] == ds) & (sub_df["scope"] == scp)]
                grp_sorted = grp.sort_values(by="weights_remaining", ascending=False)
                x_vals = grp_sorted["weights_remaining"] * 100

                # 1. 95% Reference Interval (Pointwise Monte Carlo reference interval)
                ax.fill_between(
                    x_vals,
                    grp_sorted["reference_p025"],
                    grp_sorted["reference_p975"],
                    color=ACCENT_BLUE,
                    alpha=0.25,
                    label="Random 95% reference interval" if (row_idx == 0 and col_idx == 0) else None,
                    linewidth=0,
                )

                # 2. Random reference mean (dashed curve)
                ax.plot(
                    x_vals,
                    grp_sorted["reference_mean"],
                    color=ACCENT_BLUE,
                    linestyle="--",
                    linewidth=1.8,
                    label="Random reference mean" if (row_idx == 0 and col_idx == 0) else None,
                    zorder=3,
                )

                # 3. Observed IMP mean (solid black curve)
                ax.plot(
                    x_vals,
                    grp_sorted["observed_mean"],
                    color=INK,
                    linestyle="-",
                    linewidth=2.0,
                    marker="o",
                    markersize=4.0,
                    label="Observed ticket mean" if (row_idx == 0 and col_idx == 0) else None,
                    zorder=4,
                )

                # Axis configuration
                ax.set_xscale("log")
                ax.invert_xaxis()
                ax.set_ylim(-0.02, 1.02)

                ticks = [100, 50, 20, 10, 5]
                ax.set_xticks(ticks)
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))

                if row_idx == 0:
                    ax.set_title(SCOPE_TITLES.get(scp, scp), pad=8, weight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"{ds_name}\nJaccard Similarity", color=INK)
                if row_idx == 1:
                    ax.set_xlabel("Weights remaining (log scale)")

                # Annotation of coverage
                n_pairs = int(grp_sorted["n_pairs"].iloc[0]) if not grp_sorted.empty else 0
                n_reps = int(grp_sorted["n_replicates"].iloc[0]) if not grp_sorted.empty else 0
                ax.text(
                    0.05,
                    0.05,
                    f"Pairs: {n_pairs} | R: {n_reps}",
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

        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
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


def plot_random_reference_difference(
    df_summary: pd.DataFrame,
    save_png: Optional[Path] = None,
    save_pdf: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Generate concise 1x2 plot of (Observed - Reference) overall Jaccard difference.

    Panel 0: MNIST, Panel 1: Fashion-MNIST.
    """
    rc_params = _get_base_rc_params()
    datasets = ["mnist", "fashion_mnist"]
    sub_df = df_summary[(df_summary["level"] > 0) & (df_summary["scope"] == "overall")].copy()

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)

        for idx, ds in enumerate(datasets):
            ax = axes[idx]
            ds_name = DATASET_TITLES.get(ds, ds)
            grp = sub_df[sub_df["dataset"] == ds].sort_values(by="weights_remaining", ascending=False)
            x_vals = grp["weights_remaining"] * 100
            diff_vals = grp["observed_minus_reference"]

            # Zero reference line
            ax.axhline(0, color=MUTED, linestyle="--", linewidth=1.0, alpha=0.7)

            ax.plot(
                x_vals,
                diff_vals,
                color=INK if ds == "mnist" else ACCENT_ORANGE,
                linewidth=2.0,
                marker="o",
                markersize=4.5,
                label=f"{ds_name} overall difference",
            )

            ax.set_xscale("log")
            ax.invert_xaxis()
            ticks = [100, 50, 20, 10, 5]
            ax.set_xticks(ticks)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))

            ax.set_title(f"{ds_name} — Excess Overlap", pad=8, weight="bold")
            ax.set_xlabel("Weights remaining (log scale)")
            if idx == 0:
                ax.set_ylabel("Observed − Reference Jaccard", color=INK)

            ax.legend(loc="upper right", frameon=False)

        plt.tight_layout()

        if save_png:
            Path(save_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_png, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            Path(save_pdf).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_pdf, bbox_inches="tight")

    return fig


def plot_prediction_agreement_curves(
    df_summary: pd.DataFrame,
    df_pairs: pd.DataFrame,
    dataset_status: Optional[dict] = None,
    save_png: Optional[Path] = None,
    save_pdf: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Plot cross-seed prediction agreement vs remaining weights.

    Shows agreement decomposed into shared correct predictions and shared errors,
    as well as individual seed pair trajectories.
    """
    rc_params = _get_base_rc_params()
    datasets = ["mnist", "fashion_mnist"]

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)

        for idx, ds in enumerate(datasets):
            ax = axes[idx]
            ds_name = DATASET_TITLES.get(ds, ds)
            is_avail = dataset_status[ds]["available"] if dataset_status and ds in dataset_status else (not df_summary[df_summary["dataset"] == ds].empty)

            if not is_avail:
                ax.text(
                    0.5,
                    0.5,
                    f"{ds_name} Blocked\n\nLocal test data not found\n(download=False enforced per experiment protocol)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9.5,
                    color=MUTED,
                    bbox=dict(
                        boxstyle="round,pad=0.6",
                        facecolor="#fbfbf9",
                        edgecolor=GRID,
                        linewidth=1.0,
                    ),
                )
                ax.set_title(f"{ds_name} — Blocked", pad=8, weight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            sub_sum = df_summary[df_summary["dataset"] == ds].sort_values(by="weights_remaining", ascending=False)
            sub_pairs = df_pairs[df_pairs["dataset"] == ds]

            x_vals = sub_sum["weights_remaining"] * 100

            # 1. Plot individual seed pairs in light colors
            for (s_a, s_b), color in PAIR_COLORS.items():
                pair_data = sub_pairs[
                    (sub_pairs["seed_a"] == s_a) & (sub_pairs["seed_b"] == s_b)
                ].sort_values(by="weights_remaining", ascending=False)
                if not pair_data.empty:
                    ax.plot(
                        pair_data["weights_remaining"] * 100,
                        pair_data["agreement"] * 100,
                        color=color,
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.8,
                        label=f"{PAIR_LABELS.get((s_a, s_b), f'{s_a} vs {s_b}')}",
                    )

            # 2. Plot Mean Agreement
            ax.plot(
                x_vals,
                sub_sum["agreement_mean"] * 100,
                color=INK,
                linewidth=2.2,
                marker="o",
                markersize=4.5,
                label="Mean Prediction Agreement",
                zorder=4,
            )

            # 3. Plot Mean Test Accuracy
            ax.plot(
                x_vals,
                sub_sum["accuracy_mean"] * 100,
                color=ACCENT_BLUE,
                linewidth=1.6,
                linestyle="--",
                marker="s",
                markersize=3.5,
                label="Mean Test Accuracy",
                zorder=3,
            )

            # 4. Plot Shared Correct
            ax.plot(
                x_vals,
                sub_sum["shared_correct_mean"] * 100,
                color="#2e8b57",
                linewidth=1.5,
                linestyle="-.",
                label="Shared Correct Predictions",
                zorder=3,
            )

            # 5. Shared Errors on secondary axis or bottom note
            ax.plot(
                x_vals,
                sub_sum["shared_wrong_mean"] * 100,
                color=ACCENT_ORANGE,
                linewidth=1.4,
                linestyle="-.",
                label="Shared Errors (Wrong)",
                zorder=3,
            )

            ax.set_xscale("log")
            ax.invert_xaxis()
            ticks = [100, 50, 20, 10, 5]
            ax.set_xticks(ticks)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))

            ax.set_title(f"{ds_name} — Prediction Agreement & Accuracy", pad=8, weight="bold")
            ax.set_xlabel("Weights remaining (log scale)")
            ax.set_ylabel("Rate (%)")
            ax.set_ylim(-1, 102)

            ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.45), fontsize=8.0, frameon=False)

        plt.tight_layout()

        if save_png:
            Path(save_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_png, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            Path(save_pdf).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_pdf, bbox_inches="tight")

    return fig


def plot_structure_function_scatter(
    df_sf_pairs: pd.DataFrame,
    dataset_status: Optional[dict] = None,
    save_png: Optional[Path] = None,
    save_pdf: Optional[Path] = None,
    dpi: int = 200,
) -> plt.Figure:
    """Plot Structure vs Function: Mask Jaccard vs Prediction Agreement.

    Reveals that winning tickets with very low coordinate overlap (low Jaccard)
    still exhibit very high functional agreement on test predictions.
    """
    rc_params = _get_base_rc_params()
    datasets = ["mnist", "fashion_mnist"]
    pair_markers = {(1, 2): "o", (1, 42): "^", (2, 42): "s"}

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)

        for idx, ds in enumerate(datasets):
            ax = axes[idx]
            ds_name = DATASET_TITLES.get(ds, ds)
            is_avail = dataset_status[ds]["available"] if dataset_status and ds in dataset_status else (not df_sf_pairs[df_sf_pairs["dataset"] == ds].empty)

            if not is_avail:
                ax.text(
                    0.5,
                    0.5,
                    f"{ds_name} Blocked\n\nLocal test data not found\n(download=False enforced)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9.5,
                    color=MUTED,
                    bbox=dict(
                        boxstyle="round,pad=0.6",
                        facecolor="#fbfbf9",
                        edgecolor=GRID,
                        linewidth=1.0,
                    ),
                )
                ax.set_title(f"{ds_name} — Blocked", pad=8, weight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            sub_df = df_sf_pairs[df_sf_pairs["dataset"] == ds].copy()

            # Separate dense (Level 0) and sparse (Levels 1-15)
            sparse_df = sub_df[sub_df["level"] > 0]
            dense_df = sub_df[sub_df["level"] == 0]

            cmap = plt.cm.viridis
            norm = plt.Normalize(vmin=1, vmax=15)

            sc = None
            for (s_a, s_b), marker in pair_markers.items():
                p_data = sparse_df[(sparse_df["seed_a"] == s_a) & (sparse_df["seed_b"] == s_b)]
                if not p_data.empty:
                    sc = ax.scatter(
                        p_data["mask_jaccard"],
                        p_data["agreement"] * 100,
                        c=p_data["level"],
                        cmap=cmap,
                        norm=norm,
                        marker=marker,
                        s=40,
                        alpha=0.85,
                        edgecolors="none",
                        label=f"{PAIR_LABELS.get((s_a, s_b), f'{s_a} vs {s_b}')}",
                    )

            # Plot dense baseline (Level 0)
            if not dense_df.empty:
                ax.scatter(
                    dense_df["mask_jaccard"],
                    dense_df["agreement"] * 100,
                    color=INK,
                    marker="*",
                    s=90,
                    zorder=5,
                    label="Level 0 (Dense Baseline)",
                )

            if sc is not None:
                cbar = plt.colorbar(sc, ax=ax, pad=0.03, aspect=25)
                cbar.set_label("Pruning Level (1 = least sparse, 15 = most sparse)", fontsize=8.5)
                cbar.ax.tick_params(labelsize=8)

            ax.set_title(f"{ds_name} — Mask Overlap vs Prediction Agreement", pad=8, weight="bold")
            ax.set_xlabel("Overall Mask Jaccard (Structural Overlap)")
            ax.set_ylabel("Prediction Agreement (%) (Functional Overlap)")
            ax.set_ylim(95.0, 100.2)
            ax.set_xlim(-0.03, 1.05)

            # Add illustrative annotation highlighting key finding
            ax.annotate(
                "Level 15 winning tickets:\nJaccard ~ 0.026, Agreement > 98%",
                xy=(0.0256, 98.23),
                xytext=(0.18, 96.2),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0),
                fontsize=8.5,
                color=INK,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fbfbf9", edgecolor=GRID),
            )

            ax.legend(loc="lower right", fontsize=8.0, frameon=False)

        plt.tight_layout()

        if save_png:
            Path(save_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_png, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            Path(save_pdf).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_pdf, bbox_inches="tight")

    return fig
