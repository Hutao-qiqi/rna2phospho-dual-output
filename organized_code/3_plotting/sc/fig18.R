# fig18.R — 各队列细胞状态 landscape（RNA-NMF 程序得分 PCA-2D，按主导程序着色）
#   注：每队列各自 PCA，坐标系不跨队列共享。
# 数据：fig3/fig18_landscape_<slug>.tsv.gz + fig18_landscape_summary.tsv

make_fig18 <- function() {
  tab20 <- c("#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
             "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
             "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
             "#17becf", "#9edae5")
  summ <- utils::read.delim(file.path(FIG3_DIR, "fig18_landscape_summary.tsv"),
                            sep = "\t", stringsAsFactors = FALSE)
  cohorts <- list(c("HeLa", "hela"), c("GSE300551", "gse300551"),
                  c("Vivo-seq Th17", "vivo_seq_th17"), c("Blair", "blair"),
                  c("PDO-CAF", "pdo_caf"))
  mk <- function(disp, slug) {
    f <- file.path(FIG3_DIR, paste0("fig18_landscape_", slug, ".tsv.gz"))
    if (!file.exists(f)) return(NULL)
    d <- utils::read.delim(gzfile(f), sep = "\t", stringsAsFactors = FALSE)
    progs <- sort(unique(d$dominant_prog))
    d$col <- tab20[(match(d$dominant_prog, progs) - 1) %% 20 + 1]
    sr <- summ[summ$cohort == disp, ]
    sub <- if (nrow(d) > 8000) d[sample(nrow(d), 8000), ] else d
    ttl <- if (nrow(sr)) sprintf("%s\nn=%s · %d prog · PC1 %.0f%% / PC2 %.0f%%",
                                 disp, format(sr$n_cells, big.mark = ","), sr$n_programs,
                                 sr$pc1_var * 100, sr$pc2_var * 100) else disp
    ggplot2::ggplot(sub, ggplot2::aes(pc1, pc2)) +
      ggplot2::geom_point(ggplot2::aes(color = col), size = 0.35, stroke = 0, alpha = 0.6) +
      ggplot2::scale_color_identity() +
      ggplot2::labs(title = ttl, x = NULL, y = NULL) +
      theme_fig3() +
      ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                     plot.title = ggplot2::element_text(size = 6.2, lineheight = 0.95),
                     plot.margin = ggplot2::margin(3, 5, 3, 5, "pt"))
  }
  ps <- Filter(Negate(is.null), lapply(cohorts, function(x) mk(x[1], x[2])))
  body <- cowplot::plot_grid(plotlist = ps, ncol = 3)
  cowplot::ggdraw() +
    cowplot::draw_label("Per-cohort cell-state landscape (PCA on RNA-NMF scores; coordinates not shared across cohorts)",
                        x = 0.012, y = 0.99, hjust = 0, vjust = 1,
                        fontfamily = "Arial", size = 7, color = "#222222") +
    cowplot::draw_plot(body, x = 0, y = 0, width = 1, height = 0.96)
}
