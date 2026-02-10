
# R代码：绘制算法比较图表
# 需要的包
library(ggplot2)
library(dplyr)
library(readr)
library(scales)

# 设置工作目录（根据需要修改路径）
# setwd("your_path_here")

# 读取数据
data <- read_csv("combined_metrics_data.csv")

# 创建绘图函数
create_comparison_plot <- function(data, metric_name, y_label, title_suffix = "") {
  
  # 筛选特定指标的数据
  plot_data <- data %>% 
    filter(Metric == metric_name) %>%
    mutate(
      Algorithm = factor(Algorithm, levels = c("FDFL-CF", "FDFL-FD", "FoldOpt", "FPTO")),
      Alpha = factor(Alpha, levels = c(0.5, 2.0)),
      Lambda_Label = factor(Lambda_Label, levels = c("No Fairness (λ=0)", "With Fairness (λ=0.5)"))
    )
  
  # 创建图表
  p <- ggplot(plot_data, aes(x = Algorithm, y = Value, fill = Lambda_Label)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.8) +
    geom_errorbar(
      aes(ymin = Value - StdDev, ymax = Value + StdDev),
      position = position_dodge(width = 0.8),
      width = 0.3,
      linewidth = 0.5
    ) +
    facet_wrap(~ paste("α =", Alpha), scales = "free_y", ncol = 2) +
    scale_fill_manual(
      values = c("No Fairness (λ=0)" = "#3498db", "With Fairness (λ=0.5)" = "#e74c3c"),
      name = "Fairness Setting"
    ) +
    labs(
      title = paste("Algorithm Comparison:", gsub("_", " ", metric_name), title_suffix),
      x = "Algorithm",
      y = y_label,
      caption = "Error bars represent ± 1 standard deviation"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.text.y = element_text(size = 10),
      axis.title = element_text(size = 12, face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 10),
      strip.text = element_text(size = 11, face = "bold"),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "grey80", fill = NA, linewidth = 0.5)
    )
  
  return(p)
}

# 为每个指标创建图表
metrics_config <- list(
  "Decision_Regret" = list(y_label = "Decision Regret", title_suffix = ""),
  "Prediction_MSE" = list(y_label = "Prediction MSE", title_suffix = ""),
  "Prediction_Fairness" = list(y_label = "Prediction Fairness", title_suffix = ""),
  "Training_Time" = list(y_label = "Training Time (seconds)", title_suffix = "")
)

# 生成所有图表
plots <- list()
for (metric in names(metrics_config)) {
  config <- metrics_config[[metric]]
  plots[[metric]] <- create_comparison_plot(data, metric, config$y_label, config$title_suffix)
  
  # 保存图表
  ggsave(
    filename = paste0(tolower(metric), "_comparison.png"),
    plot = plots[[metric]],
    width = 12, height = 8, dpi = 300, bg = "white"
  )
  
  # 也保存PDF版本
  ggsave(
    filename = paste0(tolower(metric), "_comparison.pdf"),
    plot = plots[[metric]],
    width = 12, height = 8, device = "pdf"
  )
}

# 显示决策遗憾图表
print(plots$Decision_Regret)

# 创建综合比较图表（所有指标在一个图中）
create_combined_plot <- function(data) {
  
  # 标准化数据用于比较
  normalized_data <- data %>%
    group_by(Metric) %>%
    mutate(
      Normalized_Value = (Value - min(Value, na.rm = TRUE)) / (max(Value, na.rm = TRUE) - min(Value, na.rm = TRUE)),
      Algorithm = factor(Algorithm, levels = c("FDFL-CF", "FDFL-FD", "FoldOpt", "FPTO")),
      Alpha = factor(Alpha, levels = c(0.5, 2.0)),
      Lambda_Label = factor(Lambda_Label, levels = c("No Fairness (λ=0)", "With Fairness (λ=0.5)"))
    )
  
  p <- ggplot(normalized_data, aes(x = Algorithm, y = Normalized_Value, fill = Lambda_Label)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7, alpha = 0.8) +
    facet_grid(Metric ~ paste("α =", Alpha), scales = "free_y") +
    scale_fill_manual(
      values = c("No Fairness (λ=0)" = "#3498db", "With Fairness (λ=0.5)" = "#e74c3c"),
      name = "Fairness Setting"
    ) +
    labs(
      title = "Normalized Algorithm Comparison Across All Metrics",
      x = "Algorithm",
      y = "Normalized Value (0-1 scale)",
      caption = "Values normalized within each metric for comparison"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      axis.text.y = element_text(size = 9),
      axis.title = element_text(size = 12, face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 10),
      strip.text = element_text(size = 10, face = "bold"),
      panel.grid.minor = element_blank()
    )
  
  return(p)
}

# 创建并保存综合图表
combined_plot <- create_combined_plot(data)
ggsave("combined_metrics_comparison.png", combined_plot, width = 16, height = 12, dpi = 300, bg = "white")
ggsave("combined_metrics_comparison.pdf", combined_plot, width = 16, height = 12, device = "pdf")

print("✅ 所有图表已生成并保存！")
