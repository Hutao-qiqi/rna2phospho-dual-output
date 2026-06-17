# Panel c: per-comparison distribution — RAINCLOUD style
# half-eye cloud (right) + narrow box + drug-class-colored rain (left)
# + median labels at bottom; English labels; n in title.

prepare_panel_c_data <- function(d) {
  all_cos <- d$panel_f_paired %>%
    transmute(drug_class, site_set_en = "All sites",
              metric_en = "Cosine", value = all_cosine_true)
  top_cos <- d$panel_f_paired %>%
    transmute(drug_class, site_set_en = "Responsive top 20%",
              metric_en = "Cosine", value = responsive20_cosine_true)
  dir_long <- d$panel_c_direction_long %>%
    transmute(drug_class,
              site_set_en = ifelse(site_set == "all_sites",
                                   "All sites", "Responsive top 20%"),
              metric_en = "Direction acc.", value = direction_accuracy)
  bind_rows(all_cos, top_cos, dir_long) %>%
    mutate(
      drug_class  = recode(drug_class, !!!DRUG_CLASS_EN),
      drug_class  = factor(drug_class, levels = DRUG_CLASS_ORDER_EN),
      site_set_en = factor(site_set_en,
                           levels = c("All sites", "Responsive top 20%")),
      metric_en   = factor(metric_en, levels = c("Cosine", "Direction acc."))
    )
}

make_panel_c <- function(d) {
  pc <- prepare_panel_c_data(d)
  stats <- pc %>% group_by(site_set_en, metric_en) %>%
    summarise(med = median(value, na.rm = TRUE), .groups = "drop")

  suppressPackageStartupMessages(library(ggdist))

  ggplot(pc, aes(x = site_set_en, y = value)) +
    # cloud: half-eye density pushed to the right
    ggdist::stat_halfeye(
      aes(fill = site_set_en),
      adjust = 0.8, width = 0.6, justification = -0.18,
      .width = 0, point_colour = NA, alpha = 0.5, color = NA
    ) +
    # rain: drug-class-colored jittered points
    geom_jitter(aes(color = drug_class),
                position = position_jitter(width = 0.07, seed = 11),
                size = 0.85, alpha = 0.8, stroke = 0) +
    # box: narrow, nudged slightly left
    geom_boxplot(width = 0.11, outlier.shape = NA, fill = "white",
                 color = "#2C3E50", linewidth = 0.4,
                 position = position_nudge(x = -0.12)) +
    # median labels at the bottom
    geom_text(data = stats, aes(y = 0.03, label = sprintf("med = %.2f", med)),
              size = 2.3, color = "grey25", family = BASE_FONT) +
    facet_wrap(~ metric_en, ncol = 2) +
    scale_fill_manual(values = SITE_SET_COLORS, guide = "none") +
    scale_color_manual(values = DRUG_CLASS_COLORS, breaks = DRUG_CLASS_ORDER_EN,
                       name = "Drug class") +
    scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.25),
                       expand = expansion(mult = c(0.01, 0.02))) +
    labs(title = "Per-comparison distribution (n = 125)",
         x = NULL, y = "Score") +
    theme_fig4() +
    theme(
      legend.position = "right",
      panel.grid.major.x = element_blank(),
      panel.spacing.x = unit(0.9, "lines"),
      axis.text.x = element_text(size = BASE_SIZE - 1)
    ) +
    guides(color = guide_legend(override.aes = list(size = 2.4, alpha = 1)))
}
