# fig13.R — 各外部队列 NMF：H 矩阵（成分 × readout，行归一化后按主导模块排）
#   4 队列纵向堆叠（每队列 readout 数不同，各自一行）。
# 数据：fig3/fig13_nmf_<slug>_H.tsv

make_fig13 <- function() {
  cohorts <- list(c("HeLa", "hela"), c("GSE300551", "gse300551"),
                  c("Vivo-seq Th17", "vivo_seq_th17"), c("PDO-CAF", "pdo_caf"))
  mk <- function(disp, slug) {
    f <- file.path(FIG3_DIR, paste0("fig13_nmf_", slug, "_H.tsv"))
    if (!file.exists(f)) return(NULL)
    H <- utils::read.delim(f, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
    comps <- H$component
    M <- as.matrix(H[, -1]); rownames(M) <- comps
    # 行归一化（每成分除以其最大），按主导模块 + 载荷排 readout
    Mn <- M / apply(M, 1, function(r) ifelse(max(r) > 0, max(r), 1))
    dom <- apply(Mn, 2, which.max); mx <- apply(Mn, 2, max)
    o <- order(dom, -mx)
    M <- M[, o, drop = FALSE]
    rd <- fig3_short(colnames(M), 16)
    long <- data.frame(
      comp = rep(factor(comps, levels = rev(comps)), times = ncol(M)),
      rd   = rep(factor(rd, levels = rd), each = nrow(M)),
      val  = as.vector(M))
    ggplot2::ggplot(long, ggplot2::aes(rd, comp, fill = val)) +
      ggplot2::geom_tile(color = "white", linewidth = 0.3) +
      ggplot2::scale_fill_gradientn(colors = FIG3_SEQ_TEAL, name = "H loading") +
      ggplot2::scale_x_discrete(expand = c(0, 0)) +
      ggplot2::scale_y_discrete(expand = c(0, 0)) +
      ggplot2::labs(x = NULL, y = NULL, title = paste0(disp, " · k=", nrow(M))) +
      theme_fig3() +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, vjust = 1, size = 5),
        axis.text.y = ggplot2::element_text(size = 6.5),
        axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
        legend.position = "right", legend.key.width = ggplot2::unit(2.2, "mm"),
        legend.key.height = ggplot2::unit(3.4, "mm"),
        plot.margin = ggplot2::margin(4, 6, 8, 6, "pt"))
  }
  ps <- Filter(Negate(is.null), lapply(cohorts, function(x) mk(x[1], x[2])))
  body <- cowplot::plot_grid(plotlist = ps, ncol = 1, align = "v")
  cowplot::ggdraw() +
    cowplot::draw_label("Per-cohort phospho-NMF H matrices (component × readout, row-normalised order)",
                        x = 0.012, y = 0.992, hjust = 0, vjust = 1,
                        fontfamily = "Arial", size = 7, color = "#222222") +
    cowplot::draw_plot(body, x = 0, y = 0, width = 1, height = 0.975)
}
