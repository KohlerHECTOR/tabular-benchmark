source("analyses/plot_utils.R")

benchmark <- read_csv("src/results_boosting.csv") %>%
  bind_rows(read_csv("analyses/results/benchmark_total.csv")) 


######################################################
# Benchmark regression numerical medium
print("Unique model names in raw data:")
print(unique(benchmark$model_name))


########################################################
# Benchmark classif numerical medium

df <- benchmark %>% 
  filter(benchmark == "numerical_classification_medium")
  # filter(model_name == "dpdt_c" | model_name == "cart_c" | model_name== "pystreed_c")


df <- rename(df)
checks(df)

plot_results_per_dataset(df, "accuracy", default_colscale = T, equalize_n_iteration = T, max_iter = 100)
ggsave("analyses/plots/random_search_classif_numerical_datasets_boosting_all.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.7, y_sup=1, score="accuracy", quantile=0.1, truncate_scores = F, max_iter = 100, equalize_n_iteration = T)
ggsave("analyses/plots/random_search_classif_numerical_boosting_all.pdf", width=7, height=6, bg="white")
