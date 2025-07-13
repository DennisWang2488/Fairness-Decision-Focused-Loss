# ==============================================================================
# 1. SETUP: LOAD LIBRARIES AND DATA
# ==============================================================================

# Install packages if you don't have them
# install.packages(c("data.table", "ggplot2", "scales"))

library(data.table)
library(ggplot2)
library(scales)

# --- Load and Prepare Data ---
# Make sure the 'data.csv' file is in your R working directory,
# or provide the full path.
tryCatch({
  df <- fread('E:\\myREPO\\Fairness-Decision-Focused-Loss\\Organized-FDFL\\src\\data\\data.csv')
}, error = function(e) {
  print("Warning: 'data.csv' not found. Using dummy data for demonstration.")
  # Fallback dummy data
  df <- data.table(
    patient_id = rep(1:2500, each = 2),
    race = sample(0:1, 5000, replace = TRUE),
    cost_avoidable_t = rlnorm(5000, 7, 1.5),
    gagne_sum_t = sample(0:17, 5000, replace = TRUE),
    cost_t = rlnorm(5000, 8, 2) * 100,
    risk_score_t = runif(5000) * 10,
    ghba1c_mean_t = rnorm(5000, 7, 1),
    bps_mean_t = rnorm(5000, 130, 15)
  )
})

# --- Preprocessing and Sampling ---
set.seed(42)
df <- df[sample(.N, 5000)]
df[, race := factor(fifelse(race == 0, "white", "black"))]

# --- Define Plotting Styles ---
style_palette <- c("white" = "#ffa600", "black" = "#764885")
style_linestyles <- c("white" = "solid", "black" = "dashed")

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================

df[, bps_above_139_ind := fifelse(bps_mean_t > 139, 1, 0)]
print(df[, bps_above_139_ind := fifelse(bps_mean_t > 139, 1, 0)])

# ==============================================================================
# 3. CORE PLOTTING FUNCTION
# ==============================================================================

create_r_style_plot <- function(df, y_col, x_col, main_title, y_label, span, use_log_scale = FALSE, y_ticks = NULL) {
  
  # --- Data Aggregation ---
  # Create a copy to avoid modifying the original data.table
  dt_plot <- copy(df)
  
  # Calculate percentile for each observation
  dt_plot[, percentile := as.integer(frank(.SD[[1]]) / .N * 100), by = race, .SDcols = x_col]
  
  # Aggregate by percentile (for 'x' markers and smooth line)
  percentile_data <- dt_plot[, .(agg_y = mean(.SD[[1]])), by = .(race, percentile), .SDcols = y_col]
  setnames(percentile_data, "agg_y", y_col)
  
  # Aggregate by decile (for circles and CI bars)
  dt_plot[, decile_bin := {
    breaks <- unique(quantile(.SD[[1]], probs = 0:10/10, na.rm = TRUE))
    cut(.SD[[1]], breaks = breaks, labels = FALSE, include.lowest = TRUE)
  }, .SDcols = x_col]
  decile_data <- dt_plot[, .(
    mean = mean(.SD[[1]]),
    se = sd(.SD[[1]]) / sqrt(.N)
  ), by = .(race, decile_bin), .SDcols = y_col]
  decile_data <- na.omit(decile_data) # Remove rows where decile_bin might be NA
  decile_data[, `:=`(
    ci_lower = mean - 1.96 * se,
    ci_upper = mean + 1.96 * se,
    decile_centered = decile_bin * 10 - 5
  )]
  
  # --- Manual Log Transformation (to replicate original R code) ---
  vlocation_threshold <- NA
  if (use_log_scale) {
    epsilon <- 0.001
    # Ensure lower CI bound is not negative before log transform
    decile_data[, ci_lower := pmax(epsilon, ci_lower)]
    
    percentile_data[, (y_col) := log(get(y_col) + epsilon)]
    decile_data[, `:=`(
      mean = log(mean + epsilon),
      ci_lower = log(ci_lower + epsilon),
      ci_upper = log(ci_upper + epsilon)
    )]
    y_ticks_log <- log(y_ticks)
    vlocation_threshold <- log(50000)
  } else {
    y_range <- range(percentile_data[[y_col]], na.rm = TRUE)
    vlocation_threshold <- y_range[1] + 0.8 * (y_range[2] - y_range[1])
  }
  
  # --- Plotting Layers (ggplot) ---
  # Start with an empty ggplot object. Aesthetics are now defined in each layer.
  p <- ggplot() +
    # 1. geom_point(shape = 4) for percentiles
    geom_point(data = percentile_data, aes(x = percentile, y = .data[[y_col]], color = race), shape = 4, alpha = 0.8) +
    
    # 2. geom_smooth(se = F, span = ...)
    geom_smooth(data = percentile_data, aes(x = percentile, y = .data[[y_col]], color = race, linetype = race, group = race), se = FALSE, method = "loess", formula = "y ~ x", span = span) +
    
    # 3. geom_pointrange() for deciles
    geom_pointrange(data = decile_data, aes(x = decile_centered, y = mean, ymin = ci_lower, ymax = ci_upper, color = race), size = 0.7) +
    
    # --- Annotations ---
    # These layers do not inherit the race aesthetic, which prevents the error.
    geom_vline(aes(xintercept = 97), color = "black", linetype = "dashed") +
    geom_text(aes(x = 96, y = vlocation_threshold, label = "Defaulted into program"), color = "black", hjust = 1, size = 3) +
    geom_vline(aes(xintercept = 55), color = "darkgray", linetype = "dashed") +
    geom_text(aes(x = 54, y = vlocation_threshold, label = "Referred for screen"), color = "darkgray", hjust = 1, size = 3) +
    
    # --- Scales and Labels ---
    scale_color_manual(values = style_palette, name = "Race") +
    scale_linetype_manual(values = style_linestyles, name = "Race") +
    scale_x_continuous(breaks = seq(0, 100, 10), limits = c(-2, 102)) +
    labs(
      title = main_title,
      x = paste("Percentile of", gsub("_", " ", x_col)),
      y = y_label
    ) +
    
    # --- Theme ---
    theme_bw() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(hjust = 0.5, face = "bold")
    )
  
  # Apply log scale formatting if needed
  if (use_log_scale) {
    p <- p + scale_y_continuous(
      breaks = y_ticks_log,
      labels = label_dollar(scale = 1, prefix = "$", accuracy = 1)(exp(y_ticks_log)),
      limits = c(log(700), log(100000))
    )
  }
  
  # --- Display Plot ---
  # Instead of ggsave, we now just print the plot object to display it.
  print(p)
}

# ==============================================================================
# 4. FIGURE GENERATION (INDIVIDUAL PLOTS)
# ==============================================================================

cost_ticks <- c(1000, 3000, 8000, 20000, 60000)

plots_to_generate <- list(
  list(y_col = 'cost_t', x_col = 'risk_score_t', title = 'Cost vs. Commercial Risk Score', y_label = 'Mean Total Medical Expenditure', use_log_scale = TRUE, y_ticks = cost_ticks, span = 0.45),
  # list(y_col = 'ghba1c_mean_t', x_col = 'risk_score_t', title = 'Diabetes Severity vs. Commercial Risk Score', y_label = 'Mean HbA1c (%)', span = 0.99),
  # list(y_col = 'bps_above_139_ind', x_col = 'risk_score_t', title = 'Hypertension vs. Commercial Risk Score', y_label = 'Fraction with Uncontrolled BP', span = 0.99),
  list(y_col = 'cost_t', x_col = 'benefit', title = 'Cost vs. Benefit', y_label = 'Mean Total Medical Expenditure', use_log_scale = TRUE, y_ticks = cost_ticks, span = 0.45)
)


# Loop through the definitions and create each plot
for (plot_params in plots_to_generate) {
  create_r_style_plot(
    df = df,
    y_col = plot_params$y_col,
    x_col = plot_params$x_col,
    main_title = plot_params$title,
    y_label = plot_params$y_label,
    span = plot_params$span,
    use_log_scale = !is.null(plot_params$use_log_scale) && plot_params$use_log_scale,
    y_ticks = plot_params$y_ticks
  )
}

print("All R-style replica plots have been displayed.")
