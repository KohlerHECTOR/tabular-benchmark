source("analyses/plot_utils.R")

benchmark <- read_csv("analyses/results/combined_results.csv")

######################################################
# Benchmark regression numerical medium
print("Unique model names in raw data:")
print(unique(benchmark$model_name))

df <- benchmark %>% 
  filter(benchmark == "numerical_regression_medium") %>% 
  filter(model_name != "HistGradientBoostingTree")

df <- rename(df)

# checks
checks(df)

# Dataset by dataset

plot_results_per_dataset(df, "R2 score", truncate_scores = T)
ggsave("analyses/plots/random_search_regression_numerical_datasets_all.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.4, y_sup=0.95, score="R2 score", quantile=0.5, truncate_scores = T, text_size=8, theme_size=25, max_iter=400)
ggsave("analyses/plots/random_search_regression_numerical_poster_all.pdf", width=13.5, height=7, bg="white")

plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="R2 score", quantile=0.5, truncate_scores = T, default_colscale = T)
ggsave("analyses/plots/random_search_regression_numerical_all.pdf", width=7, height=6, bg="white")


########################################################
# Benchmark classif numerical medium

df <- benchmark %>% 
  filter(benchmark == "numerical_classification_medium") %>% 
  filter(model_name != "HistGradientBoostingTree")



df <- rename(df)
checks(df)

plot_results_per_dataset(df, "accuracy", default_colscale = T)
ggsave("analyses/plots/random_search_classif_numerical_datasets_all.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.55, y_sup=0.95, score="accuracy", quantile=0.1, truncate_scores = F, text_size=8, theme_size=25, max_iter=400)
ggsave("analyses/plots/random_search_classif_numerical_poster_all.pdf",  width=13.5, height=7, bg="white")

plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="accuracy", quantile=0.1, truncate_scores = F)
ggsave("analyses/plots/random_search_classif_numerical_all.pdf", width=7, height=6, bg="white")
