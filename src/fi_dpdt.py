import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("results_cart_dpdt.csv")
df_dpdt = df[(df["model_name"] == "dpdt_c")]
# Get hyperparameter columns
filter_col = [col for col in df_dpdt if col.startswith('model__') or col == "mean_test_score"]
filter_col = [col for col in filter_col if not df_dpdt[col].isna().all()]
filter_col = [col for col in filter_col if not col == "model__random_state"]

# Remove rows with NaN values in hyperparameter columns or target
df_dpdt = df_dpdt[filter_col].dropna()
target = df_dpdt["mean_test_score"].copy()
df_dpdt = df_dpdt.drop(["mean_test_score"], axis=1)
# print(df_dpdt)

# Convert object columns to float using LabelEncoder
for col in df_dpdt.columns:
        # if df_dpdt[col].dtype == 'object':
    le = LabelEncoder()
    df_dpdt[col] = le.fit_transform(df_dpdt[col])

# Train Random Forest for feature importance
clf = RandomForestRegressor(random_state=42)
clf.fit(df_dpdt, target)
print(df_dpdt.columns)
print(clf.feature_importances_)

# Create importance dataframe
importance_df = pd.DataFrame({
    'Hyperparameter': [col.replace('model__', '') for col in df_dpdt.columns],
    'Importance': clf.feature_importances_
})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Generate LaTeX table
latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lc}\n\\toprule\n"
latex_table += "Hyperparameter & Importance (\%) \\\\\n\\midrule\n"

for _, row in importance_df.iterrows():
    # Convert importance to percentage with 2 decimal places
    importance_pct = f"{row['Importance']*100:.2f}"
    latex_table += f"{row['Hyperparameter']} & {importance_pct} \\\\\n"

latex_table += "\\bottomrule\n\\end{tabular}\n"
latex_table += "\\caption{Hyperparameter importance for DPDT}\n"
latex_table += "\\label{tab:dpdt_importance}\n\\end{table}"

print(latex_table)

# Also save to file
with open('dpdt_importance_table.tex', 'w') as f:
    f.write(latex_table)