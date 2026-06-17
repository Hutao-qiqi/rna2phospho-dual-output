# fig09.R — HeLa 逐细胞 NMF (k=3)：上排 UMAP 按 NMF 得分着色，下排各成分 loading 条
# 数据：fig3/fig9_hela_nmf_W_cell_scores.tsv + fig9_hela_nmf_H_readout_loadings.tsv
#       + SRC_CACHE/scp682_sc11_hela_scfoundation_umap.tsv

make_fig09 <- function() {
  W <- utils::read.delim(file.path(FIG3_DIR, "fig9_hela_nmf_W_cell_scores.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  H <- utils::read.delim(file.path(FIG3_DIR, "fig9_hela_nmf_H_readout_loadings.tsv"),
                         sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
  um <- utils::read.delim(file.path(SRC_CACHE, "scp682_sc11_hela_scfoundation_umap.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  d <- merge(W, um[, c("cell_id", "row_index", "umap1", "umap2")],
             by = c("cell_id", "row_index"))
  teal <- c("#F0F0F0", "#A6CEE3", "#1F78B4", "#08306B")

  mk_um <- function(cc) {
    v <- d[[cc]]; lo <- stats::quantile(v, 0.02); hi <- stats::quantile(v, 0.98)
    d$cl <- pmin(pmax(v, lo), hi)
    ggplot2::ggplot(d, ggplot2::aes(umap1, umap2, color = cl)) +
      ggplot2::geom_point(size = 0.5, stroke = 0) +
      ggplot2::scale_color_gradientn(colors = teal, name = NULL) +
      ggplot2::labs(title = toupper(cc), x = NULL, y = NULL) +
      theme_fig3() +
      ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                     legend.position = "right", legend.key.width = ggplot2::unit(2, "mm"),
                     legend.key.height = ggplot2::unit(3.2, "mm"),
                     plot.margin = ggplot2::margin(3, 4, 3, 4, "pt"))
  }

  rcols <- setdiff(names(H), "component")
  rlab  <- gsub("_", " ", gsub("_p_site_pending", "", rcols))
  mk_load <- function(i) {
    comp <- paste0("NMF", i)
    vals <- as.numeric(H[H$component == comp, rcols])
    df <- data.frame(readout = rlab, val = vals, stringsAsFactors = FALSE)
    df <- df[order(df$val), ]
    df$readout <- factor(df$readout, levels = df$readout)
    df$col <- ifelse(df$val >= stats::median(df$val), "#1F78B4", "#A6CEE3")
    ggplot2::ggplot(df, ggplot2::aes(val, readout, fill = col)) +
      ggplot2::geom_col(width = 0.7) +
      ggplot2::scale_fill_identity() +
      ggplot2::labs(x = paste0(comp, " loading"), y = NULL) +
      theme_fig3() +
      ggplot2::theme(axis.text.y = ggplot2::element_text(size = 5),
                     axis.ticks.y = ggplot2::element_blank())
  }

  top <- cowplot::plot_grid(plotlist = lapply(c("nmf1", "nmf2", "nmf3"), mk_um), ncol = 3)
  bot <- cowplot::plot_grid(plotlist = lapply(1:3, mk_load), ncol = 3)
  body <- cowplot::plot_grid(top, bot, ncol = 1, rel_heights = c(1.5, 1))
  cowplot::ggdraw() +
    cowplot::draw_label("HeLa per-cell NMF (k=3) on predicted phospho — 1,143 cells × 13 readouts",
                        x = 0.012, y = 0.987, hjust = 0, vjust = 1,
                        fontfamily = "Arial", size = 7, color = "#222222") +
    cowplot::draw_plot(body, x = 0, y = 0, width = 1, height = 0.965)
}
