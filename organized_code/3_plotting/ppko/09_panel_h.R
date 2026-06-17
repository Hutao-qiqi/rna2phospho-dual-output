# Panel h (formerly i): control benchmarks — clean lollipop + random-k curve
# Hand-curated ceiling and controls weaker than PPKO removed.

make_panel_h <- function(d) {
  ppko_primary <- d$panel_i_general %>%
    filter(score == "ppko_target_prior_abs_mean") %>% pull(auc) %>% .[1]

  gen <- d$panel_i_general %>%
    mutate(score_label_en = recode(score_label, !!!SCORE_LABEL_EN,
                                   .default = score_label),
           group_label = ifelse(score_group == "SCP682-PPKO V10B",
                                 "PPKO", "Generic control"),
           is_primary = score == "ppko_target_prior_abs_mean") %>%
    filter(score != "control_hand_pathway_score",
           !(group_label == "Generic control" & auc < ppko_primary)) %>%
    arrange(auc) %>%
    mutate(score_label_en = factor(score_label_en, levels = score_label_en))

  p_left <- ggplot(gen, aes(auc, score_label_en, color = group_label)) +
    geom_vline(xintercept = 0.5, linetype = "dashed",
               color = "grey60", linewidth = 0.35) +
    geom_segment(aes(x = 0.5, xend = auc, yend = score_label_en),
                 linewidth = 0.5) +
    geom_errorbarh(aes(xmin = bootstrap_ci_low, xmax = bootstrap_ci_high),
                   height = 0, linewidth = 0.45, alpha = 0.6) +
    geom_point(aes(fill = group_label, size = is_primary),
               shape = 21, color = "white", stroke = 0.5) +
    geom_text(aes(label = sprintf("%.2f", auc), x = bootstrap_ci_high + 0.03),
              hjust = 0, size = 2.3, color = "grey25", family = BASE_FONT) +
    scale_color_manual(values = SCORE_GROUP_COLORS, name = NULL) +
    scale_fill_manual(values = SCORE_GROUP_COLORS, name = NULL) +
    scale_size_manual(values = c(`TRUE` = 4.5, `FALSE` = 3), guide = "none") +
    scale_x_continuous(limits = c(0.3, 1.0), breaks = seq(0.4, 1.0, 0.2)) +
    labs(title = "Control benchmarks",
         subtitle = sprintf("TCGA-TCPA n = %d, identical patient set", gen$n[1]),
         x = "AUC (bootstrap 95% CI)", y = NULL) +
    theme_fig4() +
    theme(legend.position = "top", legend.justification = "right",
          panel.grid.major.y = element_blank(),
          axis.text.y = element_text(size = BASE_SIZE - 1))

  rnd <- d$panel_i_random %>%
    mutate(k = as.integer(sub("random_(\\d+)_phospho_markers_mean", "\\1", score)))

  p_right <- ggplot(rnd, aes(k, auc_mean)) +
    geom_ribbon(aes(ymin = auc_ci_low, ymax = auc_ci_high),
                fill = "#D4D4D4", alpha = 0.4) +
    geom_hline(yintercept = ppko_primary, color = "#5A89B3",
               linetype = "dashed", linewidth = 0.6) +
    annotate("text", x = 20, y = ppko_primary + 0.02, hjust = 1, vjust = 0,
             size = 2.3, color = "#5A89B3", family = BASE_FONT,
             label = sprintf("PPKO = %.2f", ppko_primary)) +
    geom_hline(yintercept = 0.5, color = "grey60", linetype = "dashed",
               linewidth = 0.35) +
    geom_line(color = "#7A7A7A", linewidth = 0.6) +
    geom_point(color = "#7A7A7A", fill = "#D4D4D4", size = 2.4, shape = 21,
               stroke = 0.5) +
    geom_text(aes(label = sprintf("%.2f", auc_mean), y = auc_mean - 0.045),
              size = 2.2, color = "grey30", family = BASE_FONT) +
    scale_x_continuous(breaks = c(3, 5, 10, 20)) +
    scale_y_continuous(limits = c(0.3, 1.0), breaks = seq(0.4, 1.0, 0.2)) +
    labs(title = "Random k-marker control",
         subtitle = "1,000 draws per k; mean ± 95% CI",
         x = "k phospho markers", y = "AUC") +
    theme_fig4()

  p_left + p_right + plot_layout(widths = c(1.5, 1))
}
