# Panel f: PPKO vs zero baseline — clean scatter + marginal violins
# single accent color; on-message (points above diagonal); sci-plot palette.

make_panel_f <- function(d) {
  df <- d$panel_f_paired %>%
    transmute(zero_value = responsive20_cosine_zero,
              true_value = responsive20_cosine_true,
              n_shared_sites)
  ws <- d$panel_f_wilcox %>%
    filter(site_set == "responsive_top20", metric == "余弦") %>% slice(1)

  p_core <- ggplot(df, aes(x = zero_value, y = true_value)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                color = "grey60", linewidth = 0.4) +
    geom_density_2d(color = "grey88", linewidth = 0.25, bins = 6) +
    geom_point(aes(size = n_shared_sites), color = "#5A89B3",
               alpha = 0.7, stroke = 0) +
    scale_size_continuous(range = c(1.0, 3.0), name = "Shared\nsites",
                          breaks = c(30, 40, 50)) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                       expand = c(0.01, 0.01)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25),
                       expand = c(0.01, 0.01)) +
    annotate("text", x = 0.04, y = 0.97, hjust = 0, vjust = 1,
             family = BASE_FONT, size = 2.4, color = "grey15", lineheight = 1.15,
             label = sprintf("ΔCos = %.2f\nWilcoxon P = %.1e\n%d/%d (%.0f%%) above diagonal",
                             ws$mean_delta_true_minus_zero, ws$wilcoxon_two_sided_p,
                             ws$n_above_diagonal_true_gt_zero, ws$n_pairs,
                             100 * ws$n_above_diagonal_true_gt_zero / ws$n_pairs)) +
    labs(title = "PPKO vs zero baseline",
         subtitle = "Responsive top 20% cosine, n = 125 paired",
         x = "Zero baseline", y = "PPKO") +
    theme_fig4() +
    theme(legend.position = "right",
          panel.grid.major = element_line(linewidth = 0.2, color = "grey94"))

  suppressPackageStartupMessages(library(ggExtra))
  ggExtra::ggMarginal(p_core, type = "violin", margins = "both",
                      fill = "#C1D8E9", color = "#5A89B3", alpha = 0.7, size = 5)
}
