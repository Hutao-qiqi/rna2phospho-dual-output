# panel_b.R — Fig 3 panel b：内部训练集重建 + 4 个外部单细胞队列的 per-site median Spearman
#
# source 后调用 make_panel_b() 返回 ggplot 对象（不带 panel letter）。
#
# 数据：fig3/fig3_panel_b_data.tsv
#   display_cohort / site_count / median_spearman / mean / min / max / sites_kept
# 柱 = median ρ；外部队列须 = per-site min–max（n > 1 才画）；柱上标 median。
# 重要：Internal = 分布内训练重建（headline train_median_spearman_all=0.469, 38 位点）；
#       其余 4 个 = 分布外外部验证。两者口径不同，用板岩蓝 + 点线分隔区分。
# 风格：Fig 2 bar idiom（geom_col 0.66 黑边 0.25 + 顶值 + 零线）。

.PANEL_B_TSV <- file.path(
  "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11",
  "04_figure_source_data", "fig3", "fig3_panel_b_data.tsv")

make_panel_b <- function() {
  d <- utils::read.delim(.PANEL_B_TSV, sep = "\t", stringsAsFactors = FALSE)
  for (cc in c("median_spearman", "min_spearman", "max_spearman"))
    d[[cc]] <- suppressWarnings(as.numeric(d[[cc]]))

  ord <- c("Internal (train recon)", "HeLa", "Blair", "GSE300551", "Vivo-seq Th17")
  short <- c("Internal (train recon)" = "Internal", "HeLa" = "HeLa", "Blair" = "Blair",
             "GSE300551" = "GSE300551", "Vivo-seq Th17" = "Vivo-Th17")
  d <- d[match(ord, d$display_cohort), ]
  d$cohort <- factor(d$display_cohort, levels = ord)
  d$internal <- d$display_cohort == "Internal (train recon)"
  d$xlab <- ifelse(d$internal,
                   "Internal\n38 · recon",
                   paste0(short[as.character(d$cohort)], "\n", d$site_count,
                          ifelse(d$site_count == 1, " site", " sites")))
  d$xlab <- factor(d$xlab, levels = d$xlab)

  # 须仅外部 n > 1
  d$wmin <- ifelse(!d$internal & d$site_count > 1, d$min_spearman, NA)
  d$wmax <- ifelse(!d$internal & d$site_count > 1, d$max_spearman, NA)
  d$top  <- pmax(d$median_spearman, ifelse(is.na(d$wmax), d$median_spearman, d$wmax))

  cols <- stats::setNames(
    c("#8491B4", unname(FIG3_PAL[c("deepteal", "teal", "lightorg", "ours")])), ord)

  ggplot2::ggplot(d, ggplot2::aes(x = xlab, y = median_spearman, fill = cohort)) +
    ggplot2::geom_hline(yintercept = 0, color = COL_ZERO,
                        linewidth = 0.3, linetype = "dashed") +
    # 分布内 vs 外部验证 分隔线
    ggplot2::geom_vline(xintercept = 1.5, linetype = "dotted",
                        color = "#9A9A9A", linewidth = 0.35) +
    ggplot2::geom_col(width = 0.66, color = "black", linewidth = 0.25) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = wmin, ymax = wmax),
                           width = 0.18, linewidth = 0.3, color = "#444444",
                           na.rm = TRUE) +
    ggplot2::geom_text(ggplot2::aes(y = top + 0.022,
                                    label = sprintf("%.3f", median_spearman)),
                       size = 5.4 / PT, family = "Arial", color = COL_TEXT, vjust = 0) +
    # 组别标注
    ggplot2::annotate("text", x = 1, y = 0.72, label = "in-dist.",
                      family = "Arial", size = 5 / PT, fontface = "italic", color = "#555555") +
    ggplot2::annotate("text", x = 3.5, y = 0.72, label = "external validation (OOD)",
                      family = "Arial", size = 5 / PT, fontface = "italic", color = "#555555") +
    ggplot2::scale_fill_manual(values = cols, guide = "none") +
    ggplot2::scale_y_continuous(limits = c(0, 0.76),
                                breaks = seq(0, 0.7, 0.1), expand = c(0, 0)) +
    ggplot2::labs(
      x = NULL, y = "Per-site Spearman ρ",
      title = "Internal recon. + external SC validation",
      subtitle = "Internal = in-dist. training recon.; others = OOD external validation.") +
    theme_fig3() +
    ggplot2::theme(
      axis.text.x  = ggplot2::element_text(size = 5.6, lineheight = 0.9),
      axis.ticks.x = ggplot2::element_blank())
}
