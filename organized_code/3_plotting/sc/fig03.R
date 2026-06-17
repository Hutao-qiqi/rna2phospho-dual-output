# fig03.R — HeLa UMAP 上的逐细胞预测误差（主图 CTNND1_T310 + 4 个小图）
# 数据：fig3/fig3_hela_umap_error_long.tsv（5 readout × 1,143 细胞）

make_fig03 <- function() {
  d <- utils::read.delim(file.path(FIG3_DIR, "fig3_hela_umap_error_long.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  d$abs_error <- suppressWarnings(as.numeric(d$abs_error))
  err_cmap <- c("#383C73", "#6CBFB5", "#F2D06B", "#C97064")
  main_t <- "CTNND1_T310"
  others <- setdiff(unique(d$target_id), main_t)[1:4]

  mk <- function(tt, big) {
    s <- d[d$target_id == tt, ]
    vmax <- as.numeric(stats::quantile(s$abs_error, 0.98, na.rm = TRUE))
    s$err_cl <- pmin(s$abs_error, vmax)
    ggplot2::ggplot(s, ggplot2::aes(umap1, umap2, color = err_cl)) +
      ggplot2::geom_point(size = if (big) 0.55 else 0.35, stroke = 0) +
      ggplot2::scale_color_gradientn(colors = err_cmap, name = "|pred − obs|") +
      ggplot2::labs(title = fig3_short(tt, 18),
                    x = if (big) "UMAP 1" else NULL,
                    y = if (big) "UMAP 2" else NULL) +
      theme_fig3() +
      ggplot2::theme(
        axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
        legend.position = if (big) "right" else "none",
        legend.key.width = ggplot2::unit(2.2, "mm"),
        legend.key.height = ggplot2::unit(3.6, "mm"),
        plot.margin = ggplot2::margin(3, 4, 3, 4, "pt"))
  }
  p_main <- mk(main_t, TRUE)
  minis  <- lapply(others, mk, big = FALSE)
  right  <- cowplot::plot_grid(plotlist = minis, ncol = 2)
  cowplot::plot_grid(p_main, right, nrow = 1, rel_widths = c(1.35, 1))
}
