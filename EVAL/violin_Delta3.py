import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import numpy as np

# ==========================================
# Color and style settings
# ==========================================
whisker_color = "#d62728"  # Red whiskers
mean_color = "#006400"  # Dark green
scatter_color = "#1f77b4"  # Unified scatter color (blue)

# Boxplot color settings (for legend)
box_face_color_rgba = (1, 1, 1, 0.1)  # White, alpha 0.1 (frosted glass effect)

box_palette = {
    "clean": "#8FD19E",
    "SLMY": "#F28E8C",
}

# Unified style
sns.set(style="white")


# ==========================================
# Custom legend handler ("I"-shaped)
# ==========================================
class HandlerWhisker(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        # Change: make legend line width thinner as well (linewidth=0.8)
        vline = Line2D([x0 + width / 2.0, x0 + width / 2.0], [y0, y0 + height],
                       color=whisker_color, linewidth=0.8)
        hline_top = Line2D([x0 + 0.2 * width, x0 + 0.8 * width], [y0 + height, y0 + height],
                           color=whisker_color, linewidth=0.8)
        hline_bottom = Line2D([x0 + 0.2 * width, x0 + 0.8 * width], [y0, y0],
                              color=whisker_color, linewidth=0.8)
        for artist in (vline, hline_top, hline_bottom):
            artist.set_transform(trans)
        return [vline, hline_top, hline_bottom]


whisker_dummy = object()

# ==========================================
# Data loading
# ==========================================
csv_path = "/home/zgc/datawork/boundary_metrics_all.csv"
try:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Delta_G", "Delta_C"])
    print("Total samples:", len(df))
except FileNotFoundError:
    print("Warning: file not found, using simulated data for demonstration.")
    # Generate high-density simulated data to demonstrate the effect
    df = pd.DataFrame({
        'group': ['clean'] * 300 + ['SLMY'] * 300,
        'Delta_G': np.concatenate([np.random.normal(0, 1, 300), np.random.normal(0.5, 1.2, 300)]),
        'Delta_C': np.concatenate([np.random.normal(0, 1, 300), np.random.normal(0.5, 1.2, 300)])
    })

group_order = ["clean", "SLMY"]

# ==========================================
# Statistical test calculations
# ==========================================
clean = df[df["group"] == "clean"]
slmy = df[df["group"] == "SLMY"]

# Delta G
t_stat_g, p_val_g = stats.ttest_ind(clean["Delta_G"], slmy["Delta_G"], equal_var=False, nan_policy="omit")
u_stat_g, p_u_g = stats.mannwhitneyu(clean["Delta_G"], slmy["Delta_G"], alternative="two-sided")

# Delta C
t_stat_c, p_val_c = stats.ttest_ind(clean["Delta_C"], slmy["Delta_C"], equal_var=False, nan_policy="omit")
u_stat_c, p_u_c = stats.mannwhitneyu(clean["Delta_C"], slmy["Delta_C"], alternative="two-sided")


# ==========================================
# Plot helper functions
# ==========================================
def get_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = (series < lower) | (series > upper)
    return outliers, ~outliers


def plot_scatter_with_outliers(ax, df_group, y_col, x_pos, jitter_amount=0.15):
    """
    Plot scatter points: Z-order set to 2, above Violin(0) but below Box(10)
    """
    outliers_mask, non_outliers_mask = get_outliers(df_group[y_col])

    np.random.seed(42)
    vals = df_group[y_col].values
    jitter_x = x_pos + np.random.uniform(-jitter_amount, jitter_amount, size=len(vals))

    # 1. Non-outliers
    ax.scatter(
        jitter_x[non_outliers_mask.values], vals[non_outliers_mask.values],
        marker='o', s=1, color=scatter_color, alpha=0.6,
        zorder=2,
        label='Non-Outliers'
    )

    # 2. Outliers
    ax.scatter(
        jitter_x[outliers_mask.values], vals[outliers_mask.values],
        marker='o', facecolors='none', edgecolors=scatter_color, linewidths=0.8, s=15,
        zorder=2,
        label='Outliers'
    )


def add_significance_bar(ax, p_val, y_buffer=0.12):
    """Add a significance bar"""
    x1, x2 = 0, 1
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    y = ymax + 0.02 * yrange
    h = 0.03 * yrange

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0)

    label = "ns" if p_val >= 0.05 else ("*" if p_val >= 0.01 else ("**" if p_val >= 0.001 else "***"))
    ax.text((x1 + x2) * 0.5, y + h, label, ha="center", va="bottom", fontsize=9)
    ax.set_ylim(ymin, ymax + y_buffer * yrange)


def create_legend(ax):
    """Create legend"""
    legend_elements = [
        # Change: set Patch linewidth to 0.8
        Patch(facecolor=box_face_color_rgba, edgecolor="black", linewidth=0.8, label="25%–75%"),
        whisker_dummy,
        # Change: set median line legend linewidth=0.8
        Line2D([0], [0], color="black", linewidth=0.8, label="Median"),
        # Change: mean point legend markersize remains small (2)
        Line2D([0], [0], marker="s", color=mean_color, linestyle="None", markersize=2, label="Mean"),
        Line2D([0], [0], marker="o", color=scatter_color, linestyle="None", markersize=1.5, label="Non-outliers"),
        Line2D([0], [0], marker="o", markerfacecolor='none', markeredgecolor=scatter_color, markeredgewidth=0.8,
               linestyle="None", markersize=4, label="Outliers"),
    ]
    ax.legend(
        handles=legend_elements,
        labels=["25%–75%", "Whisker (1.5×IQR)", "Median", "Mean", "Non-outliers", "Outliers"],
        handler_map={whisker_dummy: HandlerWhisker()},
        loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, frameon=True, fontsize=8
    )


# ==========================================
# Plotting logic
# ==========================================
metrics = [
    {"col": "Delta_G", "title": "Boundary Gradient Difference ΔG", "p_val": p_val_g, "fname": "combo_Delta_G.png"},
    {"col": "Delta_C", "title": "Boundary RMS Contrast Difference ΔC", "p_val": p_val_c, "fname": "combo_Delta_C.png"}
]

for m in metrics:
    plt.figure(figsize=(6, 5))

    # 1. Violin plot
    sns.violinplot(
        x="group", y=m["col"], hue="group", data=df, order=group_order,
        palette=box_palette, cut=0, inner=None, linewidth=0.6, legend=False,
        zorder=0
    )

    ax = plt.gca()

    # 2. Scatter plot
    for i, grp in enumerate(group_order):
        df_group = df[df["group"] == grp]
        plot_scatter_with_outliers(ax, df_group, m["col"], i)

    # 3. Box plot
    sns.boxplot(
        x="group", y=m["col"], data=df, order=group_order, width=0.25,
        showcaps=True, showfliers=False,
        linewidth=0.8,  # Change: make box outline thinner
        boxprops=dict(
            facecolor=box_face_color_rgba,
            edgecolor="black",
            zorder=10
        ),
        # Change: make median line thinner (linewidth=0.8)
        medianprops=dict(color="black", linewidth=0.8, zorder=11),
        # Change: make whiskers thinner (linewidth=0.8)
        whiskerprops=dict(linewidth=0.8, color=whisker_color, zorder=11),
        # Change: make caps thinner (linewidth=0.8)
        capprops=dict(linewidth=0.8, color=whisker_color, zorder=11)
    )

    # 4. Mean points
    for i, grp in enumerate(group_order):
        mean_val = df[df["group"] == grp][m["col"]].mean()
        # Change: s=8 (was 20), smaller mean point
        ax.scatter(i, mean_val, marker="s", color=mean_color, s=8, zorder=12, edgecolors='white', linewidths=0.5)

    # Style adjustments
    ax.grid(False)
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(["clean", "SLGP-DA"])
    ax.tick_params(axis="y", left=True, labelleft=True, right=False)
    plt.title(f"{m['title']} by Group")

    # Add legend and significance
    create_legend(ax)
    # add_significance_bar(ax, m["p_val"])

    plt.tight_layout()
    plt.savefig(m["fname"], dpi=600)
    plt.close()
    print(f"Saved {m['fname']}")

print("Plotting completed. Lines have been thinned, and mean points have been reduced in size.")
