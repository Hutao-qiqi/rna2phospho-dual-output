# panel_c.R — Fig 3 panel c：锚点磷酸位点
#   左：CTNND1 Thr310 在 HeLa 的 predicted vs observed 密度散点（hexbin）
#   右：STAT3 Tyr705 跨平台 Spearman 柱（GSE300551 / Vivo-Th17）
#
# source 后调用 make_panel_c() 返回 cowplot 横排合成对象。
#
# 数据：
#   左 fig3/panel_a_hela_ctnnd1_t310_scatter.tsv  (predicted, observed, ...)
#   右 fig3/panel_b_stat3_cross_platform.tsv       (test_dataset, n, spearman)

.PANEL_C_DIR <- file.path(
  "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11",
  "04_figure_source_data", "fig3")

.make_c_scatter <- function() {
  d <- utils::read.delim(file.path(.PANEL_C_DIR,
       "panel_a_hela_ctnnd1_t310_scatter.tsv"), sep = "\t",
       stringsAsFactors = FALSE)
  d <- d[is.finite(d$observed) & is.finite(d$predicted), ]
  rho <- stats::cor(d$observed, d$predicted, method = "spearman")

  ggplot2::ggplot(d, ggplot2::aes(x = observed, y = predicted)) +
    ggplot2::geom_hex(bins = 34) +
    ggplot2::geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
                         color = "#ED8D5A", linewidth = 0.5) +
    ggplot2::scale_fill_gradientn(colors = FIG3_SEQ_TEAL, trans = "log10",
                                  name = "cells") +
    ggplot2::annotate("text", x = -Inf, y = Inf, hjust = -0.10, vjust = 1.25,
                      label = sprintf("ρ = %.3f\nn = %s", rho,
                                      format(nrow(d), big.mark = ",")),
                      family = "Arial", size = 6 / PT, color = COL_TEXT) +
    ggplot2::labs(x = "Observed (z)", y = "Predicted (z)",
                  title = "CTNND1 Thr310 · HeLa") +
    theme_fig3() +
    ggplot2::theme(
      legend.position  = "right",
      legend.key.width = ggplot2::unit(2.4, "mm"),
      legend.key.height= ggplot2::unit(3.4, "mm"),
      aspect.ratio     = 1)
}

.make_c_stat3 <- function() {
  d <- utils::read.delim(file.path(.PANEL_C_DIR,
       "panel_b_stat3_cross_platform.tsv"), sep = "\t",
       stringsAsFactors = FALSE)
  d <- d[d$test_dataset != "all", ]
  nm <- c("gse300551_iccite_plex_kinase_2025" = "GSE300551",
          "vivo_seq_th17_2025" = "Vivo-Th17")
  d$grp <- nm[d$test_dataset]
  d <- d[match(c("GSE300551", "Vivo-Th17"), d$grp), ]
  d$xlab <- paste0(d$grp, "\n(n = ", format(d$n, big.mark = ","), ")")
  d$xlab <- factor(d$xlab, levels = d$xlab)

  cols <- stats::setNames(unname(FIG3_PAL[c("deepteal", "ours")]), levels(d$xlab))

  ggplot2::ggplot(d, ggplot2::aes(x = xlab, y = spearman, fill = xlab)) +
    ggplot2::geom_hline(yintercept = 0, color = COL_ZERO,
                        linewidth = 0.3, linetype = "dashed") +
    ggplot2::geom_col(width = 0.58, color = "black", linewidth = 0.25) +
    ggplot2::geom_text(ggplot2::aes(y = spearman + 0.013,
                                    label = sprintf("%.3f", spearman)),
                       size = 5.4 / PT, family = "Arial", color = COL_TEXT,
                       vjust = 0) +
    ggplot2::scale_fill_manual(values = cols, guide = "none") +
    ggplot2::scale_y_continuous(limits = c(0, 0.40), breaks = seq(0, 0.4, 0.1),
                                expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = "Spearman ρ",
                  title = "STAT3 Tyr705 · cross-platform") +
    theme_fig3() +
    ggplot2::theme(
      axis.text.x  = ggplot2::element_text(size = 6.0, lineheight = 0.9),
      axis.ticks.x = ggplot2::element_blank())
}

make_panel_c <- function() {
  cowplot::plot_grid(
    .make_c_scatter(), .make_c_stat3(),
    nrow = 1, rel_widths = c(1.32, 1))
}
