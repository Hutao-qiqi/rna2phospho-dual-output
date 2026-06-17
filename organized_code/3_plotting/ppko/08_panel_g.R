# Panel g: TCGA-TCPA patient validation — ROC + boxplot (clean sci-plot)

make_panel_g <- function(d, stack = FALSE) {
  roc_df <- d$panel_g_roc %>%
    filter(score == "ppko_target_prior_abs_mean") %>%
    arrange(fpr, tpr)
  roc_df <- bind_rows(
    tibble(fpr = 0, tpr = 0), roc_df %>% select(fpr, tpr), tibble(fpr = 1, tpr = 1)
  ) %>% distinct()

  auc_row <- d$panel_g_auc %>%
    filter(score == "ppko_target_prior_abs_mean") %>% slice(1)

  p_roc <- ggplot(roc_df, aes(fpr, tpr)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                color = "grey60", linewidth = 0.35) +
    geom_line(color = "#5A89B3", linewidth = 0.9) +
    annotate("text", x = 0.34, y = 0.26, hjust = 0, vjust = 1, family = BASE_FONT,
             size = 2.5, color = "grey15", lineheight = 1.25,
             label = sprintf("n = %d (%d R / %d NR)\nAUC = %.2f\n95%% CI %.2f-%.2f\nperm. P = %.3f",
                             auc_row$n, auc_row$n_responder, auc_row$n_non_responder,
                             auc_row$auc, auc_row$bootstrap_ci_low,
                             auc_row$bootstrap_ci_high, auc_row$permutation_p_right)) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = c(0.005, 0.005)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25), expand = c(0.005, 0.005)) +
    labs(x = "False positive rate", y = "True positive rate") +
    coord_fixed() + theme_fig4()

  box_df <- d$panel_g_box %>%
    mutate(response_label = recode(response_label, !!!RESPONSE_EN),
           response_label = factor(response_label,
                                   levels = c("Non-responder", "Responder")))

  suppressPackageStartupMessages(library(ggbeeswarm))
  p_box <- ggplot(box_df, aes(response_label, tcga_v10b_primary_score,
                              fill = response_label)) +
    geom_boxplot(width = 0.5, outlier.shape = NA, color = "grey25", linewidth = 0.4) +
    ggbeeswarm::geom_quasirandom(width = 0.13, size = 0.8, color = "grey30", alpha = 0.6) +
    scale_fill_manual(values = RESPONSE_COLORS, guide = "none") +
    labs(x = NULL, y = "PPKO target-prior |Δ|") +
    theme_fig4() +
    theme(panel.grid.major.x = element_blank(),
          axis.text.x = element_text(size = BASE_SIZE - 1))

  if (stack) p_roc / p_box + plot_layout(heights = c(1.85, 1))
  else       p_roc + p_box + plot_layout(widths = c(1.35, 1))
}
