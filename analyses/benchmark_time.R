source("analyses/plot_utils.R")

# benchmark_numerical <- read_csv("analyses/results/combined_results_dpdtcart_depth5.csv") 

##################
# Numerical classif

df <- read_csv("src/results_cart_dpdt.csv") %>%
  bind_rows(read_csv("src/results_streed.csv")) %>%
  filter(benchmark == "numerical_classification_medium") %>%
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  filter(model_name == "dpdt_c" | model_name == "cart_c" | model_name== "pystreed_c")%>%
  rename()

plot_aggregated_results_time(df, y_inf=0.8, y_sup=0.95)

ggsave("analyses/plots/benchmark_time_numerical_classif.pdf", width=7, height=6, bg="white")



df <- read_csv("src/results_cart_dpdt.csv") %>%
  bind_rows(read_csv("src/results_streed.csv")) %>%
  filter(benchmark == "categorical_classification_medium") %>%
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  filter(model_name == "dpdt_c" | model_name == "cart_c" | model_name== "pystreed_c")%>%
  rename()

plot_aggregated_results_time(df, y_inf=0.8, y_sup=0.95)

ggsave("analyses/plots/benchmark_time_categorical_classif.pdf", width=7, height=6, bg="white")