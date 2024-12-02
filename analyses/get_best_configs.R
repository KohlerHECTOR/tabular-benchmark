source("analyses/plot_utils.R")

get_top_100_configurations <- function(df) {
  # Get all column names that start with "model__"
  hp_cols <- names(df)[startsWith(names(df), "model__")]
  
  df %>%
    filter(model_name %in% c("pystreed_c", "cart_c", "dpdt_c")) %>%
    filter(benchmark == "numerical_classification_medium") %>%
    group_by(model_name) %>%
    # If there are results under 3 mins, filter for those, otherwise keep all results
    filter(if(any(mean_time <= 180, na.rm = TRUE)) 
           mean_time <= 180 
           else TRUE) %>%
    # Instead of taking just the best, take top 100 by test score
    slice_max(order_by = mean_test_score, n = 10) %>%
    select(model_name, all_of(hp_cols), mean_test_score, mean_val_score, mean_time) %>%
    arrange(model_name, desc(mean_test_score))
}

df <- read_csv("analyses/results/combined_results_dpdtcart_depth5.csv")
top_100_configs <- get_top_100_configurations(df)

write.csv(top_100_configs, file = "analyses/results/configs_best.csv", row.names = FALSE)
