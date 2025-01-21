import pandas as pd
from rlberry.manager.plotting import plot_smoothed_curves
import matplotlib.pyplot as plt

# Create figure with three subplots sharing y-axis
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

def load_and_process_data(filename, model_filter=None, depth_filter=None, max_depth=None, benchmark=None):
    df = pd.read_csv(filename)
    
    # Apply filters from exp_expected_tests.py
    if max_depth:
        df = df[df["model__max_depth"] == max_depth]
    if benchmark:
        df = df[df["benchmark"] == benchmark]
    if model_filter:
        df = df[df["model_name"] == model_filter]
    if depth_filter:
        df = df[df["mean_depth_scores"] <= depth_filter]
    
    # Get hyperparameter columns (model parameters)
    model_params = [col for col in df.columns if col.startswith('model__') or col == "mean_depth_scores"]
    model_params = [col for col in model_params if not df[col].isna().all()]
    model_params = [col for col in model_params if not col == "model__random_state"]
    df = df[model_params + ["mean_time"] + ['mean_test_score']].dropna()
    
    return df

# Load data for all plots with consistent preprocessing
max_depth = 5
benchmark = "numerical_classification_medium"

df_streed = load_and_process_data("results_streed.csv", max_depth=max_depth, benchmark=benchmark)
df_dpdt = load_and_process_data("results_cart_dpdt.csv", "dpdt_c", max_depth=max_depth, benchmark=benchmark)
df_cart = load_and_process_data("results_cart_dpdt.csv", "cart_c", 32, max_depth=max_depth, benchmark=benchmark)

# Plot 1: Tree Depth
grouped_stats = df_streed.groupby('mean_depth_scores')['mean_test_score'].mean().reset_index()
grouped_stats_dpdt = df_dpdt.groupby('mean_depth_scores')['mean_test_score'].mean().reset_index()
grouped_stats_cart = df_cart.groupby('mean_depth_scores')['mean_test_score'].mean().reset_index()

grouped_stats["name"] = ["PySTreeD"] * len(grouped_stats)
grouped_stats_dpdt["name"] = ["DPDT"] * len(grouped_stats_dpdt)
grouped_stats_cart["name"] = ["CART"] * len(grouped_stats_cart)
grouped_stats["n_simu"] = [0] * len(grouped_stats)
grouped_stats_dpdt["n_simu"] = [0] * len(grouped_stats_dpdt)
grouped_stats_cart["n_simu"] = [0] * len(grouped_stats_cart)

df_depth = pd.concat([grouped_stats, grouped_stats_cart, grouped_stats_dpdt])

# Plot 2: Number of Nodes
grouped_stats = df_streed.groupby('mean_nodes_scores')['mean_test_score'].mean().reset_index()
grouped_stats_dpdt = df_dpdt.groupby('mean_nodes_scores')['mean_test_score'].mean().reset_index()
grouped_stats_cart = df_cart.groupby('mean_nodes_scores')['mean_test_score'].mean().reset_index()

grouped_stats["name"] = ["PySTreeD"] * len(grouped_stats)
grouped_stats_dpdt["name"] = ["DPDT"] * len(grouped_stats_dpdt)
grouped_stats_cart["name"] = ["CART"] * len(grouped_stats_cart)
grouped_stats["n_simu"] = [0] * len(grouped_stats)
grouped_stats_dpdt["n_simu"] = [0] * len(grouped_stats_dpdt)
grouped_stats_cart["n_simu"] = [0] * len(grouped_stats_cart)

df_nodes = pd.concat([grouped_stats, grouped_stats_cart, grouped_stats_dpdt])

# Plot 3: Expected Number of Tests
grouped_stats = df_streed.groupby('mean_expected_tests_scores')['mean_test_score'].mean().reset_index()
grouped_stats_dpdt = df_dpdt.groupby('mean_expected_tests_scores')['mean_test_score'].mean().reset_index()
grouped_stats_cart = df_cart.groupby('mean_expected_tests_scores')['mean_test_score'].mean().reset_index()

grouped_stats["name"] = ["PySTreeD"] * len(grouped_stats)
grouped_stats_dpdt["name"] = ["DPDT"] * len(grouped_stats_dpdt)
grouped_stats_cart["name"] = ["CART"] * len(grouped_stats_cart)
grouped_stats["n_simu"] = [0] * len(grouped_stats)
grouped_stats_dpdt["n_simu"] = [0] * len(grouped_stats_dpdt)
grouped_stats_cart["n_simu"] = [0] * len(grouped_stats_cart)

df_tests = pd.concat([grouped_stats, grouped_stats_cart, grouped_stats_dpdt])

# Create the three plots
plot_smoothed_curves(
    data=df_depth,
    x="mean_depth_scores",
    y="mean_test_score",
    smoothing_bandwidth=0.2,
    level=0.95,
    error_representation="raw_curves",
    show=False,
    ax=ax1
)
ax1.set_xlabel('Tree Depth', fontsize=12)
ax1.set_ylabel('Test Score', fontsize=12)

plot_smoothed_curves(
    data=df_nodes,
    x="mean_nodes_scores",
    y="mean_test_score",
    smoothing_bandwidth=0.2,
    level=0.95,
    error_representation="raw_curves",
    show=False,
    ax=ax2
)
ax2.set_xlabel('Tree Nodes', fontsize=12)

plot_smoothed_curves(
    data=df_tests,
    x="mean_expected_tests_scores",
    y="mean_test_score",
    smoothing_bandwidth=0.2,
    level=0.95,
    error_representation="raw_curves",
    show=False,
    ax=ax3
)
ax3.set_xlabel('Expected Tests', fontsize=12)

# Adjust layout and appearance
plt.tight_layout()
fig.subplots_adjust(wspace=0.1)

# Make axis labels and ticks larger
for ax in [ax1, ax2, ax3]:
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3)

plt.show()