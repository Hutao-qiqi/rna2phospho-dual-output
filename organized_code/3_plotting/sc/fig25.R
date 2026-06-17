# fig25.R — 校准：每读数把 predicted / observed 的 10 分位 bin 均值各自 z 化后叠加可靠性曲线
#   z 化使不同抗体尺度可比；对角 y=x = 完美单调校准。每队列注标中位 bin-Spearman。
#   PDO/CAF（失败队列）与 Vivo 校准较差，如实呈现。
# 数据：fig3/fig25_calibration_curves_z.tsv + fig25_calibration_cohort_summary.tsv（prep 生成）

make_fig25 <- function() {
  z <- utils::read.delim(file.path(FIG3_DIR, "fig25_calibration_curves_z.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  s <- utils::read.delim(file.path(FIG3_DIR, "fig25_calibration_cohort_summary.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  z$pred_z <- as.numeric(z$pred_z); z$obs_z <- as.numeric(z$obs_z)

  nice <- c(signal_seq_gse256403_hela_2024 = "SIGNAL-seq HeLa",
            gse300551_iccite_plex_kinase_2025 = "GSE300551",
            signal_seq_gse256404_pdo_caf_2024 = "SIGNAL-seq PDO/CAF",
            vivo_seq_th17_2025 = "Vivo-seq Th17")
  keep <- names(nice)
  z <- z[z$cohort_id %in% keep, ]
  z$coh <- factor(nice[z$cohort_id], levels = unname(nice))
  z$grp <- interaction(z$cohort_id, z$target_id, drop = TRUE)

  s <- s[s$cohort_id %in% keep, ]
  s$coh <- factor(nice[s$cohort_id], levels = unname(nice))
  s$lab <- sprintf("median bin ρ = %.2f\n(%d readouts)",
                   as.numeric(s$median_bin_spearman), as.integer(s$n_targets))

  lim <- c(-3.05, 3.05)
  ggplot2::ggplot(z, ggplot2::aes(x = pred_z, y = obs_z)) +
    ggplot2::annotate("segment", x = lim[1], y = lim[1], xend = lim[2], yend = lim[2],
                      color = "#B0B0B0", linewidth = 0.35, linetype = "dashed") +
    ggplot2::geom_line(ggplot2::aes(group = grp), color = "#51999F",
                       linewidth = 0.3, alpha = 0.45) +
    ggplot2::geom_point(color = "#347", fill = "#7BC0CD", shape = 21,
                        size = 0.55, stroke = 0.12, alpha = 0.5) +
    ggplot2::geom_text(data = s, ggplot2::aes(x = lim[1] + 0.08, y = lim[2] - 0.05, label = lab),
                       hjust = 0, vjust = 1, family = "Arial", size = 4.7 / PT,
                       color = COL_TEXT, lineheight = 1.0, inherit.aes = FALSE) +
    ggplot2::facet_wrap(~ coh, nrow = 2) +
    ggplot2::scale_x_continuous(limits = lim, breaks = c(-2, 0, 2)) +
    ggplot2::scale_y_continuous(limits = lim, breaks = c(-2, 0, 2)) +
    ggplot2::coord_equal() +
    ggplot2::labs(
      x = "Predicted bin mean (z within readout)",
      y = "Observed bin mean (z within readout)",
      title = "Calibration: binned predicted vs observed (z-scored)",
      subtitle = "Each line = one readout's 10 prediction-bin means; dashed = perfect monotonic calibration (y = x).") +
    theme_fig3() +
    ggplot2::theme(panel.spacing = ggplot2::unit(3, "mm"))
}
