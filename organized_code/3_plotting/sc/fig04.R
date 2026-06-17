# fig04.R — 跨队列 Spearman 矩阵（56 readout × 内部CV + 5 外部队列）
#   行已是聚类顺序；y 标签按通路上色；发散填充（蓝-白-橙，居中 0）。
# 数据：fig3/fig4_cross_cohort_spearman_matrix.tsv

make_fig04 <- function() {
  m <- utils::read.delim(file.path(FIG3_DIR, "fig4_cross_cohort_spearman_matrix.tsv"),
                         sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
  coln <- c(internal_cv = "Internal CV", gse300551 = "GSE300551", he_la = "HeLa",
            blair = "Blair", vivo_th17 = "Vivo-Th17", pdo_caf = "PDO-CAF")
  cohorts <- intersect(names(coln), colnames(m))
  targets <- m$target_id
  tlab    <- fig3_short(targets, 22)

  long <- do.call(rbind, lapply(cohorts, function(cc)
    data.frame(ti = seq_along(targets), cohort = unname(coln[cc]),
               rho = suppressWarnings(as.numeric(m[[cc]])), stringsAsFactors = FALSE)))
  long$target <- factor(long$ti, levels = rev(seq_along(targets)), labels = rev(tlab))
  long$cohort <- factor(long$cohort, levels = unname(coln[cohorts]))

  ycol <- unname(FIG3_GROUP_TEXT[fig3_target_group(rev(targets))])

  ggplot2::ggplot(long, ggplot2::aes(cohort, target, fill = rho)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.3) +
    ggplot2::scale_fill_gradient2(low = "#3C7E8C", mid = "white", high = "#ED8D5A",
                                  midpoint = 0, limits = c(-0.4, 1.0),
                                  oob = scales::squish, na.value = "#EFEFEF",
                                  name = "Spearman ρ") +
    ggplot2::scale_x_discrete(position = "top", expand = c(0, 0)) +
    ggplot2::scale_y_discrete(expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = NULL, title = "Cross-cohort per-readout Spearman") +
    theme_fig3() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 40, hjust = 0, vjust = 0, size = 6),
      axis.text.y = ggplot2::element_text(size = 4.6, color = ycol),
      axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
      legend.position = "right",
      legend.key.width = ggplot2::unit(2.6, "mm"),
      legend.key.height = ggplot2::unit(5, "mm"),
      plot.margin = ggplot2::margin(6, 8, 4, 4, "pt"))
}
