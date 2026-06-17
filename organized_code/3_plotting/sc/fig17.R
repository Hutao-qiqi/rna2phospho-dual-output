# fig17.R — 各队列 phospho-NMF × RNA-NMF Spearman 矩阵（2×2）
# 数据：fig3/fig17_phospho_rna_spearman_<slug>.tsv（行 = phospho NMF，列 = rna_nmf）

make_fig17 <- function() {
  cohorts <- list(c("HeLa", "hela"), c("GSE300551", "gse300551"),
                  c("Vivo-seq Th17", "vivo_seq_th17"), c("PDO-CAF", "pdo_caf"))
  mk <- function(disp, slug) {
    f <- file.path(FIG3_DIR, paste0("fig17_phospho_rna_spearman_", slug, ".tsv"))
    if (!file.exists(f)) return(NULL)
    m <- utils::read.delim(f, sep = "\t", row.names = 1, check.names = FALSE)
    ph <- rownames(m); rn <- colnames(m)
    long <- data.frame(
      ph = rep(factor(ph, levels = rev(ph)), times = length(rn)),
      rn = rep(factor(rn, levels = rn), each = length(ph)),
      val = as.vector(as.matrix(m)))
    long$lab <- ifelse(abs(long$val) >= 0.3, sprintf("%+.2f", long$val), "")
    long$dark <- abs(long$val) > 0.4
    ggplot2::ggplot(long, ggplot2::aes(rn, ph, fill = val)) +
      ggplot2::geom_tile(color = "white", linewidth = 0.3) +
      ggplot2::geom_text(ggplot2::aes(label = lab, color = dark),
                         size = 4 / PT, show.legend = FALSE) +
      ggplot2::scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#222222"), guide = "none") +
      ggplot2::scale_fill_gradient2(low = "#3C7E8C", mid = "white", high = "#ED8D5A",
                                    midpoint = 0, limits = c(-0.5, 0.5),
                                    oob = scales::squish, name = "ρ") +
      ggplot2::scale_x_discrete(expand = c(0, 0)) +
      ggplot2::scale_y_discrete(expand = c(0, 0)) +
      ggplot2::labs(x = NULL, y = NULL, title = paste0(disp, " · ", ncol(m), " RNA programs")) +
      theme_fig3() +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, vjust = 1, size = 4.6),
        axis.text.y = ggplot2::element_text(size = 6),
        axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
        legend.position = "right", legend.key.width = ggplot2::unit(2.2, "mm"),
        legend.key.height = ggplot2::unit(3.6, "mm"),
        plot.margin = ggplot2::margin(4, 6, 2, 4, "pt"))
  }
  ps <- Filter(Negate(is.null), lapply(cohorts, function(x) mk(x[1], x[2])))
  body <- cowplot::plot_grid(plotlist = ps, ncol = 2, align = "hv")
  cowplot::ggdraw() +
    cowplot::draw_label("Per-cohort phospho-NMF × RNA-NMF Spearman matrices",
                        x = 0.012, y = 0.99, hjust = 0, vjust = 1,
                        fontfamily = "Arial", size = 7, color = "#222222") +
    cowplot::draw_plot(body, x = 0, y = 0, width = 1, height = 0.97)
}
