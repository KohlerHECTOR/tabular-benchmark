import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv("results.csv")
df_dpdt = df[(df["model_name"] == "cart_c")]
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
clf = RandomForestRegressor()
clf.fit(df_dpdt, target)
print(df_dpdt.columns)
print(clf.feature_importances_)