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

                # Distinct, staggered markers keep near-coincident pairs visible
                # without shifting any observed coordinates.
                pair_group = pairs_sub[
                    (pairs_sub["dataset"] == ds) & (pairs_sub["scope"] == scp)
                ]
                for pair_idx, ((s_a, s_b), grp) in enumerate(
                    pair_group.groupby(["seed_a", "seed_b"])
                ):
                    grp_sorted = grp.sort_values(by="weights_remaining", ascending=False)
                    pair_key = (min(s_a, s_b), max(s_a, s_b))
                    p_color = {
                        (1, 2): "#648bd0",
                        (1, 42): "#df9665",
                        (2, 42): "#64aa94",
                    }.get(pair_key, "#707070")
                    p_label = PAIR_LABELS.get(pair_key, f"Seeds {s_a} vs {s_b}")
                    ax.plot(
                        grp_sorted["weights_remaining"] * 100,
                        grp_sorted["jaccard"],
                        color=p_color,
                        linewidth=0.8,
                        alpha=0.75,
                        linestyle=["-", "--", ":"][pair_idx % 3],
                        marker=["o", "s", "^"][pair_idx % 3],
                        markevery=(pair_idx % 3, 3),
                        markersize=4.0,
                        markerfacecolor="white",
                        markeredgewidth=0.8,
                        zorder=3,
                        label=p_label if (row_idx == 0 and col_idx == 0) else None,
                    )

                # Keep the mean behind the pair markers; hollow diamonds and
                # a fine dashed line avoid masking the individual trajectories.
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
                    linewidth=1.0,
                    linestyle=(0, (5, 4)),
                    alpha=0.45,
                    marker="D",
                    markevery=(1, 3),
                    markersize=6.5,
                    markerfacecolor="none",
                    markeredgewidth=0.9,
                    label="Mean across pairs" if (row_idx == 0 and col_idx == 0) else None,
                    zorder=2,
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
                    alpha=0.15,
                    label="Random 95% reference interval" if (row_idx == 0 and col_idx == 0) else None,
                    linewidth=0,
                )

                # 2. Random reference mean (dashed curve)
                ax.plot(
                    x_vals,
                    grp_sorted["reference_mean"],
                    color="#648bd0",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.75,
                    label="Random reference mean" if (row_idx == 0 and col_idx == 0) else None,
                    zorder=3,
                )

                # 3. Observed IMP mean (solid black curve)
                ax.plot(
                    x_vals,
                    grp_sorted["observed_mean"],
                    color=INK,
                    linestyle="-",
                    linewidth=1.0,
                    alpha=0.45,
                    marker="o",
                    markersize=4.0,
                    markerfacecolor="none",
                    markeredgewidth=0.8,
                    label="Observed ticket mean" if (row_idx == 0 and col_idx == 0) else None,
                    zorder=2,
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
            ax.axhline(0, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.45)

            ax.plot(
                x_vals,
                diff_vals,
                color=INK if ds == "mnist" else "#df9665",
                linewidth=0.9,
                alpha=0.6,
                marker="o",
                markersize=4.0,
                markerfacecolor="none",
                markeredgewidth=0.8,
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
    """Focused accuracy/agreement panels and separate shared-error panels."""
    with plt.rc_context(_get_base_rc_params()):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        for col, ds in enumerate(["mnist", "fashion_mnist"]):
            top, accuracy_ax, bottom = axes[:, col]
            summary = df_summary[df_summary["dataset"] == ds].sort_values("weights_remaining", ascending=False)
            pairs = df_pairs[df_pairs["dataset"] == ds]
            if summary.empty:
                for ax in (top, accuracy_ax, bottom):
                    ax.text(.5, .5, "Saved results unavailable", ha="center", transform=ax.transAxes)
                continue
            x = summary["weights_remaining"] * 100
            for idx, (pair, color) in enumerate(PAIR_COLORS.items()):
                d = pairs[(pairs.seed_a == pair[0]) & (pairs.seed_b == pair[1])].sort_values("weights_remaining", ascending=False)
                top.plot(d.weights_remaining * 100, d.agreement * 100, color=color,
                         linewidth=.8, alpha=.8, linestyle=["-", "--", ":"][idx], label=PAIR_LABELS[pair], zorder=3)
            for key, label, color, style in [
                ("agreement_mean", "Mean agreement", INK, "--"),
                ("accuracy_mean", "Mean test accuracy", "#648bd0", "-."),
                ("shared_correct_mean", "Shared correct", "#64aa94", ":"),
            ]:
                target_ax = top if key == "agreement_mean" else accuracy_ax
                target_ax.plot(x, summary[key] * 100, color=color, linestyle=style,
                         linewidth=1, alpha=.55, label=label, zorder=2)
            values = pairs.agreement.to_numpy() * 100
            padding = max(.25, np.ptp(values) * .12)
            top.set_ylim(max(0, values.min() - padding), min(100, values.max() + padding))
            top.set_title(f"{DATASET_TITLES[ds]} — Prediction Agreement", weight="bold")
            top.set_ylabel("Rate (%) — truncated y-axis")
            top.legend(fontsize=7.5, ncol=2, loc="lower center")
            accuracy_values = summary[["accuracy_mean", "shared_correct_mean"]].to_numpy() * 100
            accuracy_padding = max(.25, np.ptp(accuracy_values) * .12)
            accuracy_ax.set_ylim(max(0, accuracy_values.min() - accuracy_padding), min(100, accuracy_values.max() + accuracy_padding))
            accuracy_ax.set_title(f"{DATASET_TITLES[ds]} — Accuracy and Joint Correctness")
            accuracy_ax.set_ylabel("Rate (%) — truncated y-axis")
            accuracy_ax.legend(fontsize=8, loc="lower center")
            bottom.plot(x, summary.shared_wrong_mean * 100, color="#df9665", marker="o",
                        markerfacecolor="none", markersize=3.5, linewidth=.9, alpha=.8, label="Shared wrong (same class)")
            bottom.set_ylim(0, max(1, summary.shared_wrong_mean.max() * 115))
            bottom.set_title(f"{DATASET_TITLES[ds]} — Shared Incorrect Predictions")
            bottom.set_ylabel("All test examples (%)")
            bottom.legend(fontsize=8)
            for ax in (top, accuracy_ax, bottom):
                ax.set_xscale("log")
                ax.invert_xaxis()
                ax.set_xticks([100, 50, 20, 10, 5])
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%g%%"))
                ax.set_xlabel("Weights remaining (log scale)")
        fig.tight_layout()
        for path in (save_png, save_pdf):
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
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
                        s=28,
                        alpha=0.65,
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
                    s=65,
                    alpha=0.55,
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
            agreement_pct = sub_df["agreement"] * 100
            ax.set_ylim(agreement_pct.min() - 0.8, agreement_pct.max() + 0.8)
            ax.set_xlim(-0.03, 1.05)

            # Annotate each dataset using its saved Level 15 results.
            last_level = sub_df[sub_df["level"] == 15]
            if not last_level.empty:
                mean_j = last_level["mask_jaccard"].mean()
                mean_a = last_level["agreement"].mean() * 100
                ax.annotate(
                    f"Level 15 mean:\nJaccard = {mean_j:.4f}, Agreement = {mean_a:.2f}%",
                    xy=(mean_j, mean_a),
                    xytext=(0.05, 0.08),
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8),
                    fontsize=8.0,
                    color=MUTED,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fbfbf9", edgecolor=GRID),
                )

            ax.legend(loc="upper right", fontsize=8.0, frameon=False)

        plt.tight_layout()

        if save_png:
            Path(save_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_png, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            Path(save_pdf).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_pdf, bbox_inches="tight")

    return fig


LAYER_COLORS = {
    "fc1.weight": "#648bd0",
    "fc2.weight": "#df9665",
    "fc3.weight": "#64aa94",
    "overall": "#666666",
}

LAYER_DISPLAY_NAMES = {
    "fc1.weight": "FC1 (300 × 784, 20%/lvl)",
    "fc2.weight": "FC2 (100 × 300, 20%/lvl)",
    "fc3.weight": "FC3 (10 × 100, 10%/lvl)",
    "overall": "Overall Network",
}


def plot_layer_sparsity_curves(
    df_summary: pd.DataFrame,
    out_path: Optional[Path] = None,
    save_png: Optional[Path] = None,
    save_pdf: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot layerwise sparsity and surviving weights across pruning levels."""
    target_png = save_png or out_path
    df_plot = df_summary[df_summary["dataset"] == "mnist"].copy() if "dataset" in df_summary.columns else df_summary.copy()

    with plt.rc_context(_get_base_rc_params()):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=dpi)

        ax1 = axes[0]
        for layer_name in ["fc1.weight", "fc2.weight", "fc3.weight", "overall"]:
            sub = df_plot[df_plot["layer"] == layer_name].sort_values("level")
            ax1.plot(
                sub["level"],
                sub["sparsity_mean"] * 100.0,
                marker="o",
                markersize=3.5,
                linewidth=0.9,
                alpha=0.7,
                label=LAYER_DISPLAY_NAMES[layer_name],
                color=LAYER_COLORS[layer_name],
            )

        ax1.set_title("Layer Sparsity across Pruning Levels", pad=8, weight="bold")
        ax1.set_xlabel("Pruning Level")
        ax1.set_ylabel("Sparsity (% pruned)")
        ax1.set_ylim(-2.0, 102.0)
        ax1.set_xticks(range(0, 16, 2))
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        ax1.legend(loc="lower right", fontsize=8.0)

        ax2 = axes[1]
        for layer_name in ["fc1.weight", "fc2.weight", "fc3.weight", "overall"]:
            sub = df_plot[df_plot["layer"] == layer_name].sort_values("level")
            ax2.plot(
                sub["level"],
                sub["surviving_mean"],
                marker="s",
                markersize=3.5,
                linewidth=0.9,
                alpha=0.7,
                label=LAYER_DISPLAY_NAMES[layer_name],
                color=LAYER_COLORS[layer_name],
            )

        ax2.set_title("Surviving Connection Count (Log Scale)", pad=8, weight="bold")
        ax2.set_xlabel("Pruning Level")
        ax2.set_ylabel("Surviving Connections (Log Scale)")
        ax2.set_yscale("log")
        ax2.set_xticks(range(0, 16, 2))
        ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax2.legend(loc="upper right", fontsize=8.0)

        plt.tight_layout()

        if target_png:
            Path(target_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_png, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            Path(save_pdf).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_pdf, bbox_inches="tight")

    return fig


def plot_layer_share_evolution(
    df_summary: pd.DataFrame,
    out_path: Optional[Path] = None,
    save_png: Optional[Path] = None,
    save_pdf: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot the evolution of each layer's share of total remaining network parameters."""
    target_png = save_png or out_path
    df_plot = df_summary[df_summary["dataset"] == "mnist"].copy() if "dataset" in df_summary.columns else df_summary.copy()

    with plt.rc_context(_get_base_rc_params()):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=dpi)

        ax1 = axes[0]
        for layer_name in ["fc1.weight", "fc2.weight"]:
            sub = df_plot[df_plot["layer"] == layer_name].sort_values("level")
            ax1.plot(
                sub["level"],
                sub["surviving_share_mean"] * 100.0,
                marker="o",
                markersize=3.5,
                linewidth=0.9,
                alpha=0.7,
                label=LAYER_DISPLAY_NAMES[layer_name],
                color=LAYER_COLORS[layer_name],
            )

        ax1.set_title("Surviving Parameter Share: FC1 & FC2", pad=8, weight="bold")
        ax1.set_xlabel("Pruning Level")
        ax1.set_ylabel("Share of Remaining Weights (%)")
        ax1.set_xticks(range(0, 16, 2))
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        ax1.legend(loc="center right", fontsize=8.0)

        ax2 = axes[1]
        sub_fc3 = df_plot[df_plot["layer"] == "fc3.weight"].sort_values("level")
        ax2.plot(
            sub_fc3["level"],
            sub_fc3["surviving_share_mean"] * 100.0,
            marker="^",
            markersize=4.0,
            linewidth=0.9,
            alpha=0.7,
            label="FC3 (10%/lvl rate vs 20%/lvl for FC1/FC2)",
            color=LAYER_COLORS["fc3.weight"],
        )

        ax2.set_title("Surviving Parameter Share: FC3 (Relative Expansion)", pad=8, weight="bold")
        ax2.set_xlabel("Pruning Level")
        ax2.set_ylabel("Share of Remaining Weights (%)")
        ax2.set_xticks(range(0, 16, 2))
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

        ax2.annotate(
            "Share expands 5.7×:\n0.38% (L0) → 2.16% (L15)",
            xy=(15, 2.16),
            xytext=(9, 1.4),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0),
            fontsize=8.5,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fbfbf9", edgecolor=GRID),
        )

        ax2.legend(loc="upper left", fontsize=8.0)

        plt.tight_layout()

        if target_png:
            Path(target_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_png, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            Path(save_pdf).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_pdf, bbox_inches="tight")

    return fig


def plot_pruning_schedule_adherence(
    df_summary: pd.DataFrame,
    out_path: Optional[Path] = None,
    save_png: Optional[Path] = None,
    save_pdf: Optional[Path] = None,
    dpi: int = 300,
) -> plt.Figure:
    """Plot actual per-round pruning rate vs scheduled rate and rounding discrepancy."""
    target_png = save_png or out_path
    df_plot = df_summary[df_summary["dataset"] == "mnist"].copy() if "dataset" in df_summary.columns else df_summary.copy()
    sparse_data = df_plot[df_plot["level"] > 0].copy()

    with plt.rc_context(_get_base_rc_params()):
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), dpi=dpi, sharex=True)

        ax1 = axes[0]
        for layer_name in ["fc1.weight", "fc2.weight", "fc3.weight", "overall"]:
            sub = sparse_data[sparse_data["layer"] == layer_name].sort_values("level")
            ax1.plot(
                sub["level"],
                sub["actual_prune_rate_mean"] * 100.0,
                marker="o",
                markersize=4.0,
                linewidth=0.9,
                alpha=0.7,
                label=LAYER_DISPLAY_NAMES[layer_name],
                color=LAYER_COLORS[layer_name],
            )

        ax1.set_title("Per-Round Actual Pruning Rate vs Pruning Schedule", pad=8, weight="bold")
        ax1.set_ylabel("Actual Prune Rate (%)")
        ax1.set_ylim(8.0, 22.0)
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
        ax1.legend(loc="center right", fontsize=8.0)

        ax2 = axes[1]
        for layer_name in ["fc1.weight", "fc2.weight", "fc3.weight", "overall"]:
            sub = sparse_data[sparse_data["layer"] == layer_name].sort_values("level")
            ax2.plot(
                sub["level"],
                sub["prune_rate_diff_mean"] * 100.0,
                marker="s",
                markersize=4.0,
                linewidth=0.9,
                alpha=0.7,
                label=LAYER_DISPLAY_NAMES[layer_name],
                color=LAYER_COLORS[layer_name],
            )

        ax2.axhline(0.0, color=MUTED, linestyle="--", linewidth=0.8)
        ax2.set_title("Rounding Discrepancy: (Actual - Scheduled Rate) from integer round()", pad=8, weight="bold")
        ax2.set_xlabel("Pruning Level")
        ax2.set_ylabel("Discrepancy (percentage points)")
        ax2.set_xticks(range(1, 16))
        ax2.set_ylim(-0.20, 0.20)
        ax2.legend(loc="lower left", fontsize=8.0)

        plt.tight_layout()

        if target_png:
            Path(target_png).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target_png, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            Path(save_pdf).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_pdf, bbox_inches="tight")

    return fig


def plot_excess_jaccard_bars(df_summary: pd.DataFrame) -> plt.Figure:
    """Compare observed-minus-reference overall Jaccard at selected levels."""
    levels = [1, 2, 3, 5, 8, 11, 15]
    selected = df_summary[(df_summary["scope"] == "overall") & df_summary["level"].isin(levels)].copy()
    if selected.duplicated(["dataset", "level"]).any():
        raise ValueError("Expected one overall summary per dataset and level")
    selected["excess"] = selected["observed_mean"] - selected["reference_mean"]
    table = selected.pivot(index="level", columns="dataset", values="excess").reindex(levels)
    if table.reindex(columns=["mnist", "fashion_mnist"]).isna().any().any():
        raise ValueError("Missing selected dataset/level results")
    with plt.rc_context(_get_base_rc_params()):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(levels))
        for offset, ds, color in [(-.19, "mnist", "#648bd0"), (.19, "fashion_mnist", "#df9665")]:
            bars = ax.bar(x + offset, table[ds], width=.36, color=color, alpha=.85,
                          label=DATASET_TITLES[ds], zorder=3)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.axhline(0, color=MUTED, linewidth=.8)
        ax.set_ylim(min(0, table.min().min() * 1.2), max(.001, table.max().max() * 1.22))
        ax.set_xticks(x, [f"L{level}" for level in levels])
        ax.set_xlabel("Pruning Level")
        ax.set_ylabel("Excess Jaccard (observed − random reference)")
        ax.set_title("MNIST vs Fashion-MNIST, selected pruning levels", fontsize=10, pad=10)
        fig.suptitle("Extension 2 — Observed Excess Jaccard", fontsize=13, weight="bold")
        ax.xaxis.grid(False)
        ax.legend()
        fig.tight_layout()
        return fig


def plot_conditional_error_agreement(cea_summary: pd.DataFrame) -> plt.Figure:
    """Mean per-pair CEA with descriptive ±1 sample SD (not a CI)."""
    with plt.rc_context(_get_base_rc_params()):
        fig, ax = plt.subplots(figsize=(10, 5))
        lower_bounds, upper_bounds = [], []
        for ds, color, marker in [
            ("mnist", "#648bd0", "o"),
            ("fashion_mnist", "#df9665", "s"),
        ]:
            group = cea_summary[cea_summary["dataset"] == ds].sort_values("level")
            if group["level"].duplicated().any():
                raise ValueError("CEA chart expects one model per dataset and level")
            x = group["level"].to_numpy()
            mean = group["cea_mean"].to_numpy(dtype=float) * 100
            sd = group["cea_sd"].to_numpy(dtype=float) * 100
            ax.plot(x, mean, color=color, marker=marker, markersize=4,
                    markerfacecolor="white", markeredgewidth=.8,
                    linewidth=1.1, label=DATASET_TITLES[ds])
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.14,
                            linewidth=0)
            lower_bounds.extend((mean - sd)[np.isfinite(mean - sd)])
            upper_bounds.extend((mean + sd)[np.isfinite(mean + sd)])
        if lower_bounds:
            span = max(1, max(upper_bounds) - min(lower_bounds))
            low = max(0, min(lower_bounds) - .1 * span)
            high = min(100, max(upper_bounds) + .1 * span)
            ax.set_ylim(low, high)
            ax.set_ylabel("Conditional Error Agreement (%)" +
                          (" — truncated y-axis" if low > 0 or high < 100 else ""))
        else:
            ax.set_ylabel("Conditional Error Agreement (%)")
        ax.set_xticks(range(16))
        ax.set_xlabel("Pruning Level")
        ax.set_title("Conditional Error Agreement Across Pruning Levels", weight="bold", pad=12)
        ax.legend(title="Mean across seed pairs")
        fig.text(.5, .02, "Shading: ±1 sample SD across shared-seed pairs (descriptive variation, not a confidence interval).",
                 ha="center", fontsize=8, color=MUTED)
        fig.tight_layout(rect=[0, .05, 1, 1])
        return fig
