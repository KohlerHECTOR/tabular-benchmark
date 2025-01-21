import pandas as pd
from rlberry.manager.plotting import plot_smoothed_curves
import matplotlib.pyplot as plt

def create_combined_plot(metrics, output_filename):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    axes = [ax1, ax2, ax3]

    # Set font sizes and line width
    plt.rcParams.update({'font.size': 18})  # Increase font size
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'axes.labelsize': 18})
    plt.rcParams.update({'xtick.labelsize': 16})
    plt.rcParams.update({'ytick.labelsize': 16})
    plt.rcParams.update({'lines.linewidth': 3})  # Increase line width
    plt.rcParams.update({'legend.fontsize': 18})  # Increase legend font size

    for (metric_name, label), ax in zip(metrics, axes):
        df = pd.read_csv("results_streed.csv")
        df = df[df["model__max_depth"] == 5]
        df = df[df["benchmark"] == "numerical_classification_medium"]
        
        # Get hyperparameter columns (model parameters)
        model_params = [col for col in df.columns if col.startswith('model__') or col == metric_name]
        model_params = [col for col in model_params if not df[col].isna().all()]
        model_params = [col for col in model_params if not col == "model__random_state"]
        df = df[model_params+["mean_time"]+['mean_test_score']].dropna()
        # Group by metric and get mean test_score
        grouped_stats = df.groupby(metric_name)['mean_test_score'].mean().reset_index()

        df_dpdt = pd.read_csv("results_cart_dpdt.csv")
        df_dpdt = df_dpdt[df_dpdt["model__max_depth"] == 5]
        df_dpdt = df_dpdt[df_dpdt["benchmark"] == "numerical_classification_medium"]
        df_dpdt = df_dpdt[(df_dpdt["model_name"] == "dpdt_c")]
        
        model_params_dpdt = [col for col in df_dpdt.columns if col.startswith('model__') or col == metric_name]
        model_params_dpdt = [col for col in model_params_dpdt if not df_dpdt[col].isna().all()]
        model_params_dpdt = [col for col in model_params_dpdt if not col == "model__random_state"]
        df_dpdt = df_dpdt[model_params_dpdt+["mean_time"]+['mean_test_score']].dropna()
        grouped_stats_dpdt = df_dpdt.groupby(metric_name)['mean_test_score'].mean().reset_index()

        df_cart = pd.read_csv("results_cart_dpdt.csv")
        df_cart = df_cart[df_cart["model__max_depth"] == 5]
        if metric_name == "mean_nodes_scores":
            df_cart = df_cart[df_cart["mean_nodes_scores"] <= 32]
        df_cart = df_cart[df_cart["benchmark"] == "numerical_classification_medium"]
        df_cart = df_cart[(df_cart["model_name"] == "cart_c")]
        
        model_params_cart = [col for col in df_cart.columns if col.startswith('model__') or col == metric_name]
        model_params_cart = [col for col in model_params_cart if not df_cart[col].isna().all()]
        model_params_cart = [col for col in model_params_cart if not col == "model__random_state"]
        df_cart = df_cart[model_params_cart+["mean_time"]+['mean_test_score']].dropna()
        grouped_stats_cart = df_cart.groupby(metric_name)['mean_test_score'].mean().reset_index()

        # Add name and n_simu columns
        grouped_stats_cart["name"] = ["CART"] * len(grouped_stats_cart)
        grouped_stats_cart["n_simu"] = [0] * len(grouped_stats_cart)
        grouped_stats_dpdt["name"] = ["DPDT"] * len(grouped_stats_dpdt)
        grouped_stats_dpdt["n_simu"] = [0] * len(grouped_stats_dpdt)
        grouped_stats["name"] = ["STreeD"] * len(grouped_stats)
        grouped_stats["n_simu"] = [0] * len(grouped_stats)
        
        df_combined = pd.concat([grouped_stats, grouped_stats_cart, grouped_stats_dpdt])
        
        plot_smoothed_curves(
            data=df_combined,
            x=metric_name,
            y="mean_test_score",
            smoothing_bandwidth=2 if metric_name=="mean_nodes_scores" else 0.2,
            level=0.95,
            error_representation="raw_curves",
            show=False,
            savefig_fname=None,
            ax=ax
        )
        
        # Explicitly set font sizes for each axis
        ax.set_xlabel(label, fontsize=24)
        if ax != ax1:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Mean Test Score", fontsize=24)
            
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # Adjust legend
        if ax.get_legend():
            ax.legend(fontsize=22, loc='best')

    # Increase spacing between subplots
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()

# Combine the plots into one
metrics = [
    ("mean_depth_scores", "Depth"),
    ("mean_nodes_scores", "Nodes"),
    ("mean_expected_tests_scores", "Expected Tests")
]

create_combined_plot(metrics, "combined_plot_numerical.pdf")