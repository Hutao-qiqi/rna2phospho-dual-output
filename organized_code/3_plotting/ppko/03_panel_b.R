# Panel b: P100 overall — clean grouped bar (sci-plot restrained)
# x = site set, fill = metric; mean +- 1.96*SEM; chance line; primary starred.

make_panel_b <- function(d) {
  df <- d$panel_b_bars %>%
    mutate(
      metric_en   = recode(metric, !!!METRIC_EN),
      site_set_en = recode(site_set, !!!SITE_SET_EN),
      site_set_en = factor(site_set_en,
                           levels = c("All sites", "Responsive top 20%")),
      metric_en   = factor(metric_en, levels = c("Cosine", "Direction acc.")),
      is_primary  = site_set == "responsive_top20" & metric == "方向准确率"
    )

  prim <- df %>% filter(is_primary)

  ggplot(df, aes(x = site_set_en, y = mean, fill = metric_en)) +
    geom_hline(yintercept = 0.5, linetype = "dashed",
               color = "grey60", linewidth = 0.35) +
    geom_col(position = position_dodge(width = 0.72), width = 0.62,
             color = NA) +
    # primary endpoint outlined
    geom_col(data = prim, position = position_dodge(width = 0.72), width = 0.62,
             fill = NA, color = "grey20", linewidth = 0.6) +
    geom_errorbar(aes(ymin = ci95_low, ymax = ci95_high),
                  position = position_dodge(width = 0.72),
                  width = 0.16, linewidth = 0.4, color = "grey30") +
    geom_text(aes(y = ci95_high + 0.04, label = sprintf("%.2f", mean)),
              position = position_dodge(width = 0.72),
              size = 2.5, color = "grey20", family = BASE_FONT) +
    annotate("text", x = 0.52, y = 0.53, label = "chance", hjust = 0,
             size = 2.1, color = "grey55", fontface = "italic",
             family = BASE_FONT) +
    scale_fill_manual(values = METRIC_COLORS, name = NULL) +
    scale_y_continuous(limits = c(0, 1.05), breaks = seq(0, 1, 0.25),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(title = "P100 external validation",
         subtitle = "n = 125 comparisons; primary endpoint outlined",
         x = NULL, y = "Score (mean ± 95% CI)") +
    theme_fig4() +
    theme(
      panel.grid.major.x = element_blank(),
      legend.position = c(0.99, 1.02), legend.justification = c(1, 1),
      legend.direction = "horizontal",
      legend.background = element_rect(fill = "white", color = NA),
      axis.text.x = element_text(size = BASE_SIZE - 1)
    )
}
