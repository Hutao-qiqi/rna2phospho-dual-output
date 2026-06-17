# fig02.R — 抗体/readout 敏感性哑铃图
#   同基因/同 parent readout 的低 vs 高克隆 Spearman 连线 + fold 比注释。
# 数据：fig3/fig2_clone_sensitivity_dumbbell.tsv

make_fig02 <- function() {
  d <- utils::read.delim(file.path(FIG3_DIR, "fig2_clone_sensitivity_dumbbell.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  d <- d[order(d$absolute_difference), ]
  d$ord  <- seq_len(nrow(d))
  d$ylab <- paste0(d$gene, " · ", d$cohort_or_scope)

  pts <- rbind(
    data.frame(x = d$low_spearman,  ord = d$ord, kind = "lower readout"),
    data.frame(x = d$high_spearman, ord = d$ord, kind = "higher readout"))
  pts$kind <- factor(pts$kind, levels = c("lower readout", "higher readout"))

  ggplot2::ggplot() +
    ggplot2::geom_segment(data = d,
      ggplot2::aes(x = low_spearman, xend = high_spearman, y = ord, yend = ord),
      color = "#BDBDBD", linewidth = 1.4, lineend = "round") +
    ggplot2::geom_point(data = pts, ggplot2::aes(x = x, y = ord, fill = kind),
                        shape = 21, color = "white", size = 2.6, stroke = 0.4) +
    ggplot2::geom_text(data = d,
      ggplot2::aes(x = pmax(low_spearman, high_spearman) + 0.03, y = ord,
                   label = sprintf("%.1f×", fold_ratio_abs)),
      hjust = 0, size = 5 / PT, family = "Arial", color = "#333333") +
    ggplot2::scale_fill_manual(values = c("lower readout" = "#92B1D9",
                                          "higher readout" = "#D98973"), name = NULL) +
    ggplot2::scale_y_continuous(breaks = d$ord, labels = d$ylab, expand = c(0, 0.6)) +
    ggplot2::scale_x_continuous(limits = c(-0.02, 0.80), breaks = seq(0, 0.8, 0.2)) +
    ggplot2::labs(x = "Spearman ρ", y = NULL,
                  title = "Antibody / readout clone sensitivity",
                  subtitle = "Same gene; lower vs higher antibody-clone Spearman. Label = |high|/|low|.") +
    theme_fig3() +
    ggplot2::theme(panel.grid.major.x = ggplot2::element_line(color = "#ECECEC", linewidth = 0.3),
                   axis.text.y = ggplot2::element_text(size = 6.2))
}
