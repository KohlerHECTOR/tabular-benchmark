source("analyses/plot_utils.R")

benchmark <- read_csv("analyses/results/combined_results.csv")

######################################################
# Benchmark regression numerical medium
print("Unique model names in raw data:")
print(unique(benchmark$model_name))

df <- benchmark %>% 
  filter(benchmark == "categorical_classification_medium") %>% 
  filter(model_name == "dpdt_c" | model_name == "cart_c")


df <- rename(df)

# checks
checks(df)

# Dataset by dataset

plot_results_per_dataset(df, "accuracy", default_colscale = T)


ggsave("analyses/plots/benchmark_categorical_classif_datasets.pdf", width=15, height=10, bg="white")

# Aggregated
plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="accuracy", quantile=0.1, truncate_scores = F, text_size=8, theme_size=25, max_iter=400)
ggsave("analyses/plots/benchmark_categorical_classif_poster.pdf", width=13.5, height=7, bg="white")

plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="accuracy", max_iter=400)
ggsave("analyses/plots/benchmark_categorical_classif.pdf", width=7, height=6, bg="white")

##################################
# Benchmark regression categorical medium

df <- benchmark %>% 
  filter(benchmark == "categorical_regression_medium") %>%
  filter(model_name == "dpdt_r" | model_name == "cart_r")

df <- rename(df)
checks(df)


plot_results_per_dataset(df, "R2 score", truncate_scores = T, default_colscale = T)
ggsave("analyses/plots/benchmark_categorical_regression_datasets.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, score = "R2 score", quantile=0.5, truncate_scores = T, y_inf=0.5, y_sup=0.95, text_size=8, theme_size=25, max_iter=400)
ggsave("analyses/plots/benchmark_categorical_regression.pdf", width=13.5, height=7, bg="white")


