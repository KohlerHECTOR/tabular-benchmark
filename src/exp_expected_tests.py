import pandas as pd
from rlberry.manager.plotting import plot_smoothed_curves
import matplotlib.pyplot as plt

df = pd.read_csv("results_streed.csv")
# df = df[df["mean_nodes_scores"] >= 1]
df = df[df["model__max_depth"] == 5]
# df = df[df["data__keyword"]=="Higgs"]

# df = df[(df["model_name"] == "dpdt_c")]
# Get hyperparameter columns (model parameters)
model_params = [col for col in df.columns if col.startswith('model__') or col == "mean_nodes_scores"]
model_params = [col for col in model_params if not df[col].isna().all()]
model_params = [col for col in model_params if not col == "model__random_state"]
df = df[model_params+["mean_time"]+['mean_test_score']].dropna()
# Group by expected_tests_scores and get max test_score
grouped_stats = df.groupby('mean_nodes_scores')['mean_test_score'].max().reset_index()

df_dpdt = pd.read_csv("results_cart_dpdt.csv")
# df_dpdt = df_dpdt[df_dpdt["mean_nodes_scores"] >= 1]
df_dpdt = df_dpdt[df_dpdt["model__max_depth"] == 5]
# df_dpdt = df_dpdt[df_dpdt["data__keyword"]=="Higgs"]



df_dpdt = df_dpdt[(df_dpdt["model_name"] == "dpdt_c")]
# Get hyperparameter columns (model parameters)
model_params_dpdt = [col for col in df_dpdt.columns if col.startswith('model__') or col == "mean_nodes_scores"]
model_params_dpdt = [col for col in model_params_dpdt if not df_dpdt[col].isna().all()]
model_params_dpdt = [col for col in model_params_dpdt if not col == "model__random_state"]
df_dpdt = df_dpdt[model_params_dpdt+["mean_time"]+['mean_test_score']].dropna()
# Group by expected_tests_scores and get max test_score
grouped_stats_dpdt = df_dpdt.groupby('mean_nodes_scores')['mean_test_score'].max().reset_index()


df_cart = pd.read_csv("results_cart_dpdt.csv")
# df_cart = df_cart[df_cart["mean_nodes_scores"] >= 1]
df_cart = df_cart[df_cart["model__max_depth"] == 5]
df_cart = df_cart[df_cart["mean_nodes_scores"] <= 32]

# df_cart = df_cart[df_cart["data__keyword"]=="Higgs"]



df_cart = df_cart[(df_cart["model_name"] == "cart_c")]
# Get hyperparameter columns (model parameters)
model_params_cart = [col for col in df_cart.columns if col.startswith('model__') or col == "mean_nodes_scores"]
model_params_cart = [col for col in model_params_cart if not df_cart[col].isna().all()]
model_params_cart = [col for col in model_params_cart if not col == "model__random_state"]
df_cart = df_cart[model_params_cart+["mean_time"]+['mean_test_score']].dropna()
# Group by expected_tests_scores and get max test_score
grouped_stats_cart = df_cart.groupby('mean_nodes_scores')['mean_test_score'].max().reset_index()


grouped_stats_cart["name"] = ["CART"] * len(grouped_stats_cart)
grouped_stats_cart["n_simu"] = [0] * len(grouped_stats_cart)
grouped_stats_dpdt["name"] = ["DPDT"] * len(grouped_stats_dpdt)
grouped_stats_dpdt["n_simu"] = [0] * len(grouped_stats_dpdt)
grouped_stats["name"] = ["PySTreeD"] * len(grouped_stats)
grouped_stats["n_simu"] = [0] * len(grouped_stats)
df = pd.concat([grouped_stats, grouped_stats_cart, grouped_stats_dpdt])
# Create the plot
plot_smoothed_curves(
        data=df,
        x="mean_nodes_scores",
        y="mean_test_score",
        smoothing_bandwidth=1,
        level=0.95,
        error_representation="raw_curves",
        show=False,  # Don't show individual plots
        savefig_fname=None,  # Don't save individual plots
        # legend=False  # Add this parameter to prevent the function from creating a legend
    )
plt.tight_layout()
plt.savefig("res_nodes.pdf")