source("analyses/plot_utils.R")

df <- read_csv("src/results.csv") %>%
  bind_rows(read_csv("src/results_streed.csv")) %>%
  filter(benchmark == "numerical_classification_medium") %>%
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp, mean_nodes_scores) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  filter(model_name == "dpdt_c" | model_name == "cart_c" | model_name== "pystreed_c") %>%
  rename()

# First group by dataset (data__keyword) and expected tests to get best score per dataset
# Then average these best scores for each rounded expected test value
plot_data <- df %>%
  # Round expected tests first to group similar values
  mutate(mean_nodes_scores = round(mean_nodes_scores, 1)) %>%
  # Filter out values > 5
  filter(mean_nodes_scores <= 63) %>%
  filter(mean_nodes_scores >= 1) %>%
  # Get best score per dataset and expected test value
  group_by(data__keyword, model_name, mean_nodes_scores) %>%
  summarise(
    best_score = max(mean_test_score, na.rm = TRUE),
    .groups = 'keep'
  ) %>%
  # Now average the best scores across datasets
  group_by(model_name, mean_nodes_scores) %>%
  summarise(
    mean_test_score = mean(best_score, na.rm = TRUE),
    se_test_score = sd(best_score, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )

# Create the plot with consistent styling
ggplot(plot_data, aes(x = mean_nodes_scores, y = mean_test_score, color = model_name)) +
  geom_line(size = 2) +
  geom_ribbon(aes(ymin = mean_test_score - se_test_score, 
                  ymax = mean_test_score + se_test_score,
                  fill = model_name), 
              alpha = 0.3) +
  geom_text_repel(aes(label = model_name),
                  data = plot_data %>% 
                    group_by(model_name) %>% 
                    filter(mean_nodes_scores == min(mean_nodes_scores)),
                  bg.color = 'white', 
                  size = 6.5, 
                  bg.r = 0.15,
                  nudge_y = 0, 
                  nudge_x = 0.3, 
                  min.segment.length = 100) +
  scale_x_continuous(name = "Number of Tree Nodes") +
  scale_y_continuous(name = "Mean Test Score") +
  theme_minimal(base_size = 22) +
  theme(legend.position = "none") +
  colScale

ggsave("analyses/plots/benchmark_nodes_numerical_classif.pdf", width=7, height=6, bg="white")



df <- read_csv("src/results.csv") %>%
  bind_rows(read_csv("src/results_streed.csv")) %>%
  filter(benchmark == "categorical_classification_medium") %>%
  select(model_name, data__keyword, mean_val_score, mean_test_score, mean_time, hp, mean_nodes_scores) %>% 
  filter(!is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>% 
  filter(model_name == "dpdt_c" | model_name == "cart_c" | model_name== "pystreed_c") %>%
  rename()

# First group by dataset (data__keyword) and expected tests to get best score per dataset
# Then average these best scores for each rounded expected test value
plot_data <- df %>%
  # Round expected tests first to group similar values
  mutate(mean_nodes_scores = round(mean_nodes_scores, 1)) %>%
  # Filter out values > 5
  filter(mean_nodes_scores <= 63) %>%
  filter(mean_nodes_scores >= 1) %>%
  # Get best score per dataset and expected test value
  group_by(data__keyword, model_name, mean_nodes_scores) %>%
  summarise(
    best_score = max(mean_test_score, na.rm = TRUE),
    .groups = 'keep'
  ) %>%
  # Now average the best scores across datasets
  group_by(model_name, mean_nodes_scores) %>%
  summarise(
    mean_test_score = mean(best_score, na.rm = TRUE),
    se_test_score = sd(best_score, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  )

# Create the plot with consistent styling
ggplot(plot_data, aes(x = mean_nodes_scores, y = mean_test_score, color = model_name)) +
  geom_line(size = 2) +
  geom_ribbon(aes(ymin = mean_test_score - se_test_score, 
                  ymax = mean_test_score + se_test_score,
                  fill = model_name), 
              alpha = 0.3) +
  geom_text_repel(aes(label = model_name),
                  data = plot_data %>% 
                    group_by(model_name) %>% 
                    filter(mean_nodes_scores == min(mean_nodes_scores)),
                  bg.color = 'white', 
                  size = 6.5, 
                  bg.r = 0.15,
                  nudge_y = 0, 
                  nudge_x = 0.3, 
                  min.segment.length = 100) +
  scale_x_continuous(name = "Number of Tree Nodes") +
  scale_y_continuous(name = "Mean Test Score") +
  theme_minimal(base_size = 22) +
  theme(legend.position = "none") +
  colScale

ggsave("analyses/plots/benchmark_nodes_categorical_classif.pdf", width=7, height=6, bg="white")