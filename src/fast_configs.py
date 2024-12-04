import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

df = pd.read_csv("results_cart_dpdt.csv")
df_copy = pd.read_csv("results_cart_dpdt.csv")
df = df[(df["model_name"] == "cart_c")]
# Get hyperparameter columns (model parameters)
model_params = [col for col in df.columns if col.startswith('model__')]
model_params = [col for col in model_params if not df[col].isna().all()]
model_params = [col for col in model_params if not col == "model__random_state"]
df = df[model_params+["mean_time"]+['mean_test_score']].dropna()
df = df.sort_values('mean_test_score', ascending=False)
bests_fast_configs = 0
to_proba = []

for i, r in df.iterrows():
    df_temp = df.copy()
    for m in model_params:
        df_temp = df_temp[df_temp[m]==r[m]]
    df_good = df_temp.copy()
    df_good = df_good[df_good["mean_time"] <= 60]
    df_temp = df_temp[df_temp["mean_time"] > 60]
    if df_temp.shape[0] > 0 or df_good.shape[0] < 2:
        continue
    else:
        res = {}
        for m in model_params:
            if m == "model__cart_nodes_list":
                res[m.split("model__")[1]] = eval(r[m])
            elif m == "model__max_features" and r[m] == '10000':
                res[m.split("model__")[1]] = int(r[m])
            else:
                res[m.split("model__")[1]] = r[m]
        print(res, ",")
        to_proba.append(df_good["mean_test_score"].mean())
        bests_fast_configs += 1

    if bests_fast_configs >=100:
        break

print(*(np.array(to_proba)/np.array(to_proba).sum()), sep=",")



# # Group by model parameters and check if all times are <= 3
# always_fast_mask = df.groupby(model_params)['mean_time'].transform(lambda x: x.max() <= 6)

# always_fast_combinations = df[always_fast_mask][model_params].drop_duplicates()

# print("Parameter combinations that NEVER resulted in mean_time > 10 seconds:")
# print(always_fast_combinations)
# print(f"\nFound {len(always_fast_combinations)} unique always-fast parameter combinations")

# # Optionally, show statistics for these combinations
# always_fast_stats = df[always_fast_mask].groupby(model_params)['mean_time'].agg(['mean', 'min', 'max', 'count']).reset_index()
# print("\nParameter combinations with their execution time statistics:")
# print(always_fast_stats[always_fast_stats['count']>= 5].sort_values('mean', ascending=True))
# breakpoint()
# for index, r in always_fast_stats[always_fast_stats['count']>= 5].iterrows():
#     res = {}
#     for m in model_params:
#         if m == "model__cart_nodes_list":
#             res[m.split("model__")[1]] = eval(r[m])
#         else:
#             res[m.split("model__")[1]] = r[m]
#     print(res, ",")

# to_proba = []
# for index, r in always_fast_stats[always_fast_stats['count']>= 5].iterrows():
#     df_temp = df_copy.copy()
#     for m in model_params:
#         df_temp = df_temp[df_temp[m]==r[m]]
#     to_proba.append(df_temp["mean_test_score"].mean())

# print(*(np.array(to_proba)/np.array(to_proba).sum()), sep=",")
