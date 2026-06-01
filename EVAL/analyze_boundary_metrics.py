import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ==== 0. Plot color settings ====
# Boxplot colors for each group (based on values in the 'group' column)
# box_palette = {
#     "clean": "#4C72B0",  # Dark blue
#     "SLMY":  "#55A868",  # Green
# }
# box_palette = {
#     "clean": "#1f77b4",  # Blue (matplotlib default blue)
#     "SLMY":  "#ff7f0e",  # Orange
# }
box_palette = {
    "clean": "#55A868",  # Green
    "SLMY":  "#C44E52",  # Dark red
}


# Scatter color (stripplot)
# Common color examples:
#   "#000000"  Black
#   "#1f77b4"  Blue (matplotlib default blue)
#   "#d62728"  Red
#   "#2ca02c"  Green
#   "#ff7f0e"  Orange
scatter_color = "#1f77b4"  # Default is black here; change to any of the above if needed

# Boxplot outlier style (note: these are boxplot 'fliers', different from stripplot points)
flier_props = dict(
    marker="o",
    markerfacecolor="#1f77b4",   # Outlier fill color
    markeredgecolor="#1f77b4",   # Outlier edge color
    markersize=3,
    linestyle="none"
)

# ==== 1. Read CSV ====
csv_path = "/home/zgc/datawork/boundary_metrics_all.csv"  # Change to your path
df = pd.read_csv(csv_path)

# Optional: drop rows with NaN (if alignment fails or masks are abnormal)
df = df.dropna(subset=["Delta_G", "Delta_C"])

print("Total samples:", len(df))
print(df["group"].value_counts())

# ==== 2. Boxplot: ΔG by group ====
plt.figure(figsize=(6, 5))

sns.boxplot(
    x="group",
    y="Delta_G",
    data=df,
    showfliers=True,
    palette=box_palette,   # Use the per-group box colors set above
    flierprops=flier_props # Control the boxplot's own outlier (flier) style
)

# stripplot overlays all sample points on top of the boxplot
sns.stripplot(
    x="group",
    y="Delta_G",
    data=df,
    color=scatter_color,   # Scatter color (black / blue / red, etc.)
    alpha=0.3,
    size=2
)

# Only change x-axis tick labels: keep 'clean' unchanged, display 'SLMY' as 'SLGP-DA'
plt.gca().set_xticklabels(["clean", "SLGP-DA"])
plt.title("Boundary Gradient Difference ΔG by Group")
plt.tight_layout()
plt.savefig("boxplot_Delta_G.png", dpi=300)
plt.close()

# ==== 3. Boxplot: ΔC by group ====
plt.figure(figsize=(6, 5))

sns.boxplot(
    x="group",
    y="Delta_C",
    data=df,
    showfliers=True,
    palette=box_palette,
    flierprops=flier_props
)

sns.stripplot(
    x="group",
    y="Delta_C",
    data=df,
    color=scatter_color,
    alpha=0.3,
    size=2
)

plt.gca().set_xticklabels(["clean", "SLGP-DA"])
plt.title("Boundary RMS Contrast Difference ΔC by Group")
plt.tight_layout()
plt.savefig("boxplot_Delta_C.png", dpi=300)
plt.close()

print("Boxplots saved as boxplot_Delta_G.png / boxplot_Delta_C.png")

# ==== 4. Statistical tests ====
# Split clean and SLMY
clean = df[df["group"] == "clean"]
slmy = df[df["group"] == "SLMY"]

print("\n=== Sample counts ===")
print("clean  samples:", len(clean))
print("SLMY   samples:", len(slmy))

# --- 4.1 t-test & Mann-Whitney U for ΔG ---
dg_clean = clean["Delta_G"].values
dg_slmy = slmy["Delta_G"].values

# Two-sample t-test (allow unequal variances)
t_stat_g, p_val_g = stats.ttest_ind(
    dg_clean, dg_slmy,
    equal_var=False,
    nan_policy="omit"
)

# Mann-Whitney U test (non-parametric; does not require normality)
u_stat_g, p_u_g = stats.mannwhitneyu(
    dg_clean, dg_slmy,
    alternative="two-sided"
)

print("\n=== ΔG statistical tests ===")
print(f"t-test:        t = {t_stat_g:.4f}, p = {p_val_g:.4e}")
print(f"Mann-WhitneyU: U = {u_stat_g:.4f}, p = {p_u_g:.4e}")

# --- 4.2 t-test & Mann-Whitney U for ΔC ---
dc_clean = clean["Delta_C"].values
dc_slmy = slmy["Delta_C"].values

t_stat_c, p_val_c = stats.ttest_ind(
    dc_clean, dc_slmy,
    equal_var=False,
    nan_policy="omit"
)
u_stat_c, p_u_c = stats.mannwhitneyu(
    dc_clean, dc_slmy,
    alternative="two-sided"
)

print("\n=== ΔC statistical tests ===")
print(f"t-test:        t = {t_stat_c:.4f}, p = {p_val_c:.4e}")
print(f"Mann-WhitneyU: U = {u_stat_c:.4f}, p = {p_u_c:.4e}")
