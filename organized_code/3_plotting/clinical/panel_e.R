source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_common.R")

make_panel_e <- function() {
  d <- read.delim(file.path(.FIG5_DATA, "panel_e_tcga_kirc_site_over_parent_waterfall.tsv"),
                  sep = "\t", stringsAsFactors = FALSE)
  d <- d[order(d$site_over_parent_residual), ]
  d$rank <- seq_len(nrow(d))
  d$status <- ifelse(d$survival_event > 0, "Deceased", "Alive/censored")
  n <- nrow(d)
  q <- floor(n / 4)
  low <- d[seq_len(q), ]
  high <- d[(n - q + 1):n, ]
  tab <- matrix(c(sum(high$survival_event), nrow(high) - sum(high$survival_event),
                  sum(low$survival_event), nrow(low) - sum(low$survival_event)), nrow = 2, byrow = TRUE)
  ft <- stats::fisher.test(tab)
  txt <- sprintf("Top quartile events: %d/%d\nBottom quartile events: %d/%d\nOR=%.2f; %s",
                 sum(high$survival_event), nrow(high),
                 sum(low$survival_event), nrow(low),
                 unname(ft$estimate), p_text(ft$p.value))

  main <- ggplot(d, aes(rank, site_over_parent_residual, fill = status)) +
    geom_col(width = 0.92, linewidth = 0) +
    geom_hline(yintercept = 0, color = COL_TEXT, linewidth = 0.35) +
    scale_fill_manual(values = c("Alive/censored" = COL_BLUE, "Deceased" = COL_WARM), name = NULL) +
    annotate("text", x = Inf, y = Inf, label = txt, hjust = 1.02, vjust = 1.15,
             family = "Arial", size = 2.0, color = COL_TEXT) +
    labs(title = "Patient-level site-over-parent residual",
         subtitle = "Predicted pS6 residualized against RPS6 mRNA",
         x = NULL, y = "pS6 beyond RPS6 mRNA") +
    theme_fig5(7) +
    theme(legend.position = c(0.12, 0.86), legend.background = element_blank(),
          axis.text.x = element_blank(), axis.ticks.x = element_blank())

  anno <- ggplot(d, aes(rank, 1, color = status)) +
    geom_point(shape = 15, size = 0.55, show.legend = FALSE) +
    scale_color_manual(values = c("Alive/censored" = COL_BLUE, "Deceased" = COL_WARM)) +
    labs(x = "Patients sorted by site-over-parent residual", y = NULL) +
    theme_void(base_family = "Arial", base_size = 6) +
    theme(axis.title.x = element_text(color = COL_TEXT, margin = margin(2, 0, 0, 0)))

  cowplot::plot_grid(main, anno, ncol = 1, rel_heights = c(1, 0.11), align = "v")
}
