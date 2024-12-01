import pandas as pd


df1 = pd.read_csv('src/results.csv')
df2 = pd.read_csv('analyses/results/benchmark_total.csv')
# Method 3: If you want all columns (filling with NaN where missing)
combined_df = pd.concat([df1, df2], axis=0, ignore_index=True, join='outer')
print("Shape of df1:", df2.shape)
print("Shape of df2:", df1.shape)
print("Shape of combined:", combined_df.shape)

# Check the columns
print("Columns in combined df:", combined_df.columns.tolist())

# View the result
print(combined_df.head())
print(combined_df.tail())

combined_df.to_csv('analyses/results/combined_results_dpdtcart_depth5.csv', index=False)