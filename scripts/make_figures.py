"""Draw the paper's box plots from the CSVs written by run_experiments.py.

    python scripts/make_figures.py

Writes PDF (for LaTeX) and PNG (for previews) files to results/figures/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import MODELS, RESULTS

FIGURES = RESULTS / "figures"
BLUE, ORANGE = "#2a78d6", "#eb6834"   # colour-blind-safe categorical pair
INK, MUTED, GRID = "#222222", "#555555", "#dddddd"
LABELS = {"Ridge": "Ridge\nRegression", "Random Forest": "Random\nForest", "XGBoost": "XGBoost"}
METRICS = [("R2", "$R^2$"), ("MAE", "MAE (kg/ha)"), ("RMSE", "RMSE (kg/ha)")]
WIDTH_IN = 6.3  # just under the sn-jnl text width

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.linewidth": 0.6, "pdf.fonttype": 42,
})
MEAN_MARKER = dict(marker="D", markersize=3.5, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.7)


def draw_boxes(ax, data, positions, color, width):
    ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showmeans=True, whis=1.5,
               manage_ticks=False,
               boxprops=dict(facecolor=color + "40", edgecolor=color, linewidth=1.0),
               medianprops=dict(color=color, linewidth=1.6),
               whiskerprops=dict(color=color, linewidth=0.9),
               capprops=dict(color=color, linewidth=0.9),
               flierprops=dict(marker="o", markersize=3, markerfacecolor="white",
                               markeredgecolor=color, markeredgewidth=0.8),
               meanprops=MEAN_MARKER)


def box_figure(df, fname, group_col=None, groups=(None,), colors=(BLUE,), titles=None, zero_line=False):
    """One panel per metric, models on the x axis, one box per group."""
    fig, axes = plt.subplots(1, len(METRICS), figsize=(WIDTH_IN, 2.45))
    width = 0.62 / len(groups)
    for ax, (metric, title) in zip(axes, METRICS):
        for gi, (group, color) in enumerate(zip(groups, colors)):
            offset = (gi - (len(groups) - 1) / 2) * (width + 0.06)
            subset = df if group_col is None else df[df[group_col] == group]
            data = [subset[subset.model == m][metric].values for m in MODELS]
            draw_boxes(ax, data, [i + offset for i in range(len(MODELS))], color, width)
        if zero_line:
            ax.axhline(0, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)
        ax.set_xticks(range(len(MODELS)), [LABELS[m] for m in MODELS])
        ax.set_title((titles or {}).get(metric, title))
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=0.6)
    handles = [Patch(facecolor=c + "40", edgecolor=c, label=g) for g, c in zip(groups, colors) if g]
    handles.append(Line2D([], [], linestyle="none", label="Mean", **MEAN_MARKER))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{fname}.{ext}", bbox_inches="tight", dpi=170)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Figure 1: random split (30 splits) vs walk-forward (14 yearly test folds)
    random_label, walk_label = "Random 80:20 split (30 splits)", "Walk-forward (14 yearly test folds)"
    validation = pd.concat([
        pd.read_csv(RESULTS / "random_split_runs.csv").assign(validation=random_label),
        pd.read_csv(RESULTS / "walk_forward_years.csv").assign(validation=walk_label),
    ])
    box_figure(validation, "fig_validation_boxplots", "validation", (random_label, walk_label), (BLUE, ORANGE))

    # Figure 2: irrigation, paired change (with minus without) across 30 random splits
    box_figure(pd.read_csv(RESULTS / "irrigation_random_change.csv"), "fig_irrigation_change_boxplots",
               zero_line=True, titles={"R2": "Change in $R^2$", "MAE": "Change in MAE (kg/ha)",
                                       "RMSE": "Change in RMSE (kg/ha)"})

    # Figure 3: PMFBY, without vs with across 30 paired random splits
    box_figure(pd.read_csv(RESULTS / "pmfby_runs.csv"), "fig_pmfby_boxplots", "condition",
               ("Without PMFBY", "With PMFBY"), (BLUE, ORANGE))
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
