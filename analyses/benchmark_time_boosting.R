source("analyses/plot_utils.R")

# benchmark_numerical <- read_csv("analyses/results/combined_results_dpdtcart_depth5.csv") 

##################
# Numerical classif

df <- read_csv("src/results_boosting_dpdt.csv") %>%
  bind_rows(read_csv("src/results_boosting_cart.csv")) %>%
  filter(benchmark == "numerical_classification_medium") %>%
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()

plot_aggregated_results_time(df, y_inf=0.6)

ggsave("analyses/plots/benchmark_time_numerical_classif_boosting_dpdtcart.pdf", width=7, height=6, bg="white")



df <- read_csv("src/results_boosting_dpdt.csv") %>%
  bind_rows(read_csv("src/results_boosting_cart.csv")) %>%
  filter(benchmark == "categorical_classification_medium") %>%
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  rename()

plot_aggregated_results_time(df, y_inf=0.6)

ggsave("analyses/plots/benchmark_time_categorical_classif_boosting_dpdtcart.pdf", width=7, height=6, bg="white")