source("analyses/plot_utils.R")

benchmark <- read_csv("analyses/results/combined_results_dpdtcart_depth5.csv")

######################################################
# Benchmark regression categorical medium
print("Unique model names in raw data:")
print(unique(benchmark$model_name))


########################################################
# Benchmark classif categorical medium

df <- benchmark %>% 
  filter(benchmark == "categorical_classification_medium") %>% 
  filter(model_name == "dpdt_c" | model_name == "cart_c" | model_name == "pystreed_c")


df <- rename(df)
checks(df)

plot_results_per_dataset(df, "accuracy", default_colscale = T, equalize_n_iteration = F, max_iter = 100)
ggsave("analyses/plots/random_search_classif_categorical_datasets_depth5.pdf", width=15, height=10, bg="white")


# Aggregated

plot_aggregated_results(df, y_inf=0.5, y_sup=0.95, score="accuracy", quantile=0.1, truncate_scores = F, max_iter = 100, equalize_n_iteration = F)
ggsave("analyses/plots/random_search_classif_categorical_depth5.pdf", width=7, height=6, bg="white")
