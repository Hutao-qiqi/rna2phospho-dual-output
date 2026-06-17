# fig15.R — 跨队列高误差 RNA-NMF 程序的 Hallmark 一致性热图
#   每队列取与逐细胞误差最相关（最大正 ρ）的程序，画其 Hallmark −log10(q)。
# 数据：fig3/fig15_cross_cohort_hallmark_matrix.tsv + fig15_high_error_programs_per_cohort.tsv

make_fig15 <- function() {
  m <- utils::read.delim(file.path(FIG3_DIR, "fig15_cross_cohort_hallmark_matrix.tsv"),
                         sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
  picks <- utils::read.delim(file.path(FIG3_DIR, "fig15_high_error_programs_per_cohort.tsv"),
                             sep = "\t", stringsAsFactors = FALSE)
  gene_sets <- gsub("HALLMARK_", "", m$gene_set)
  cohort_cols <- setdiff(names(m), "gene_set")
  ord <- c("SIGNAL-seq HeLa", "GSE300551", "Vivo-seq Th17", "Blair", "SIGNAL-seq PDO/CAF")
  cohort_cols <- c(intersect(ord, cohort_cols), setdiff(cohort_cols, ord))

  long <- do.call(rbind, lapply(cohort_cols, function(cc)
    data.frame(gene_set = gene_sets, cohort = cc,
               val = suppressWarnings(as.numeric(m[[cc]])), stringsAsFactors = FALSE)))
  long$gene_set <- factor(long$gene_set, levels = rev(gene_sets))
  # x 标签：cohort + 程序 + ρ
  pk <- picks[match(cohort_cols, picks$display_cohort), ]
  xlab <- paste0(cohort_cols, "\n", pk$component,
                 sprintf("  ρ=%+.2f", pk$rho_with_error))
  long$cohort <- factor(long$cohort, levels = cohort_cols,
                        labels = xlab)
  vmax <- as.numeric(stats::quantile(long$val[long$val > 0], 0.95, na.rm = TRUE))
  long$lab <- ifelse(!is.na(long$val) & long$val > 0.5, sprintf("%.1f", long$val), "")
  long$dark <- !is.na(long$val) & long$val > vmax * 0.5
  warm <- c("#FFF5EB", "#FDD49E", "#FDAE6B", "#F16913", "#A63603")

  ggplot2::ggplot(long, ggplot2::aes(cohort, gene_set, fill = val)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.3) +
    ggplot2::geom_text(ggplot2::aes(label = lab, color = dark),
                       size = 4.6 / PT, show.legend = FALSE) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#222222"), guide = "none") +
    ggplot2::scale_fill_gradientn(colors = warm, limits = c(0, vmax),
                                  oob = scales::squish, na.value = "#F3F3F3",
                                  name = "−log10(q)") +
    ggplot2::scale_x_discrete(position = "top", expand = c(0, 0)) +
    ggplot2::scale_y_discrete(expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = NULL,
                  title = "Hallmark enrichment of high-error RNA-NMF programs across cohorts") +
    theme_fig3() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(size = 5.6, lineheight = 0.9),
      axis.text.y = ggplot2::element_text(size = 6),
      axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
      legend.position = "right",
      legend.key.width = ggplot2::unit(2.6, "mm"),
      legend.key.height = ggplot2::unit(5, "mm"),
      plot.margin = ggplot2::margin(6, 8, 4, 4, "pt"))
}
