import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

df = pd.read_csv("results_cart_dpdt.csv")
df_copy = pd.read_csv("results_cart_dpdt.csv")
df = df[(df["model_name"] == "dpdt_c")]
# Get hyperparameter columns (model parameters)
model_params = [col for col in df.columns if col.startswith('model__')]
model_params = [col for col in model_params if not df[col].isna().all()]
model_params = [col for col in model_params if not col == "model__random_state"]
df = df[model_params+["mean_time"]].dropna()
# Group by model parameters and check if all times are <= 3
always_fast_mask = df.groupby(model_params)['mean_time'].transform(lambda x: x.max() <= 6)

always_fast_combinations = df[always_fast_mask][model_params].drop_duplicates()

print("Parameter combinations that NEVER resulted in mean_time > 10 seconds:")
print(always_fast_combinations)
print(f"\nFound {len(always_fast_combinations)} unique always-fast parameter combinations")

# Optionally, show statistics for these combinations
always_fast_stats = df[always_fast_mask].groupby(model_params)['mean_time'].agg(['mean', 'min', 'max', 'count']).reset_index()
print("\nParameter combinations with their execution time statistics:")
print(always_fast_stats[always_fast_stats['count']>= 5].sort_values('mean', ascending=True))

for index, r in always_fast_stats[always_fast_stats['count']>= 5].iterrows():
    res = {}
    for m in model_params:
        if m == "model__cart_nodes_list":
            res[m.split("model__")[1]] = eval(r[m])
        else:
            res[m.split("model__")[1]] = r[m]
    print(res, ",")

to_proba = []
for index, r in always_fast_stats[always_fast_stats['count']>= 5].iterrows():
    df_temp = df_copy.copy()
    for m in model_params:
        df_temp = df_temp[df_temp[m]==r[m]]
    to_proba.append(df_temp["mean_test_score"].mean())

print(*(np.array(to_proba)/np.array(to_proba).sum()), sep=",")
