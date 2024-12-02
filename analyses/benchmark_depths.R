source("analyses/plot_utils.R")

df <- read_csv("src/results.csv") %>%
  filter(benchmark == "numerical_classification_medium") %>%
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp, mean_depth_scores) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  filter(model_name == "dpdt_c" | model_name == "cart_c" | model_name== "pystreed_c") %>%
  rename()

# Group by model_name and avg_depth, calculate mean test score
plot_data <- df %>%
  group_by(model_name, mean_depth_scores) %>%
  summarise(
    mean_test_score = mean(mean_test_score, na.rm = TRUE),
    se_test_score = sd(mean_test_score, na.rm = TRUE) / sqrt(n())
  )

# Create the plot
ggplot(plot_data, aes(x = mean_depth_scores, y = mean_test_score, color = model_name)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = mean_test_score - se_test_score, 
                  ymax = mean_test_score + se_test_score,
                  fill = model_name), 
              alpha = 0.2) +
  scale_x_continuous(name = "Average Tree Depth") +
  scale_y_continuous(name = "Mean Test Score") +
  theme_minimal(base_size = 22) +
  theme(legend.position = "bottom", 
        legend.title = element_blank()) +
  colScale

ggsave("analyses/plots/benchmark_depth_numerical_classif.pdf", width=14, height=7.3, bg="white")