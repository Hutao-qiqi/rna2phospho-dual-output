# Panel d: by drug class — clean dumbbell (cosine point -> direction point)
# sorted by direction accuracy; CI whiskers; sci-plot palette.

make_panel_d <- function(d) {
  wide <- d$panel_d_class %>%
    filter(site_set == "responsive_top20",
           metric %in% c("余弦", "方向准确率")) %>%
    mutate(metric_en  = recode(metric, !!!METRIC_EN),
           drug_class = recode(drug_class, !!!DRUG_CLASS_EN)) %>%
    select(drug_class, metric_en, mean, ci95_low, ci95_high, n)

  ord <- wide %>% filter(metric_en == "Direction acc.") %>%
    arrange(mean) %>% pull(drug_class)
  n_lab <- wide %>% filter(metric_en == "Direction acc.") %>%
    select(drug_class, n)

  wide <- wide %>%
    mutate(drug_class = factor(drug_class, levels = ord),
           metric_en  = factor(metric_en, levels = c("Cosine", "Direction acc.")))

  seg <- wide %>% select(drug_class, metric_en, mean) %>%
    pivot_wider(names_from = metric_en, values_from = mean)

  kw_p <- d$panel_d_class_kw %>%
    filter(site_set == "responsive_top20") %>% pull(kruskal_wallis_p) %>% .[1]

  ggplot() +
    geom_hline(yintercept = 0.5, linetype = "dashed",
               color = "grey75", linewidth = 0.3) +
    # connector
    geom_segment(data = seg,
                 aes(x = drug_class, xend = drug_class,
                     y = Cosine, yend = `Direction acc.`),
                 color = "grey78", linewidth = 1.0) +
    # CI whiskers
    geom_linerange(data = wide,
                   aes(x = drug_class, ymin = ci95_low, ymax = ci95_high,
                       color = metric_en),
                   linewidth = 0.4, alpha = 0.55,
                   position = position_dodge(width = 0)) +
    # points
    geom_point(data = wide,
               aes(x = drug_class, y = mean, color = metric_en),
               size = 2.6) +
    # n labels
    geom_text(data = n_lab, aes(x = drug_class, y = 0.02,
                                label = paste0("n=", n)),
              hjust = 0, size = 2.0, color = "grey50", family = BASE_FONT) +
    scale_color_manual(values = METRIC_COLORS, name = NULL) +
    scale_y_continuous(limits = c(0, 1.0), breaks = seq(0, 1, 0.25),
                       expand = expansion(mult = c(0.01, 0.03))) +
    coord_flip() +
    labs(title = "By drug class",
         subtitle = sprintf("Responsive top 20%%; Kruskal-Wallis P = %.3f", kw_p),
         x = NULL, y = "Score (mean ± 95% CI)") +
    theme_fig4() +
    theme(
      panel.grid.major.y = element_blank(),
      legend.position = "top", legend.justification = "right",
      axis.text.y = element_text(size = BASE_SIZE - 1)
    )
}
