#!/usr/bin/env Rscript
# make_main_panels.R — 渲染终稿 FIG3 a–h 对应的干净单 panel 源文件到 main_figure/
#
# 终稿 FIG3（FIG3.ai）由以下源拼成；本脚本出 e/f/g/h 的裁出版（b/c/d 直接拷源图，a 为 AI 自绘）：
#   a 架构（Illustrator 自绘）          e NMF1/2/3 UMAP（fig9 UMAP 部分）
#   b cohort Spearman（fig3_panel_b）   f phospho-NMF3×RNA-NMF01 hexbin（panel_e）
#   c 通路注意力（fig1）                g RNA-NMF01 hallmark（panel_e）
#   d HeLa UMAP 误差（fig3_hela_umap）  h NMF 分块热图（fig11 热图部分）

suppressMessages({
  library(ggplot2); library(cowplot); library(grid); library(scales); library(ragg); library(svglite)
})
SC  <- "E:/data/gongke/TCGA-TCPA/paper_final/fig3/scripts"
OUT <- "E:/data/gongke/TCGA-TCPA/paper_final/fig3/main_figure"
source(file.path(SC, "panels", "theme_fig3.R"))
source(file.path(SC, "panels", "panel_e.R"))   # .make_e_hex / .make_e_hallmark

save_fig <- function(p, stem, w, h) {
  grDevices::cairo_pdf(file.path(OUT, paste0(stem, ".pdf")), width = w/25.4, height = h/25.4, family = "Arial")
  print(p); invisible(grDevices::dev.off())
  svglite::svglite(file.path(OUT, paste0(stem, ".svg")), width = w/25.4, height = h/25.4)
  print(p); invisible(grDevices::dev.off())
  ragg::agg_png(file.path(OUT, paste0(stem, ".png")), width = w, height = h, units = "mm", res = 400)
  print(p); invisible(grDevices::dev.off())
  message("  ", stem)
}

## ---- panel e: HeLa NMF1/2/3 UMAP ----
panel_e <- function() {
  W  <- utils::read.delim(file.path(FIG3_DIR, "fig9_hela_nmf_W_cell_scores.tsv"), sep = "\t")
  um <- utils::read.delim(file.path(SRC_CACHE, "scp682_sc11_hela_scfoundation_umap.tsv"), sep = "\t")
  d  <- merge(W, um[, c("cell_id", "row_index", "umap1", "umap2")], by = c("cell_id", "row_index"))
  teal <- c("#F0F0F0", "#A6CEE3", "#1F78B4", "#08306B")
  mk <- function(cc) {
    v <- d[[cc]]; lo <- stats::quantile(v, .02); hi <- stats::quantile(v, .98); d$cl <- pmin(pmax(v, lo), hi)
    ggplot2::ggplot(d, ggplot2::aes(umap1, umap2, color = cl)) +
      ggplot2::geom_point(size = .5, stroke = 0) +
      ggplot2::scale_color_gradientn(colors = teal, name = NULL) +
      ggplot2::labs(title = toupper(cc), x = NULL, y = NULL) + theme_fig3() +
      ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                     legend.position = "right", legend.key.width = ggplot2::unit(2, "mm"),
                     legend.key.height = ggplot2::unit(3.2, "mm"))
  }
  cowplot::plot_grid(plotlist = lapply(c("nmf1", "nmf2", "nmf3"), mk), ncol = 3)
}

## ---- panel h: NMF 分块热图（成分条 + readout×cell z 热图，无底部 hexbin）----
panel_h <- function() {
  co <- utils::read.delim(file.path(FIG3_DIR, "fig11_nmf_classic_heatmap_cell_order.tsv"), sep = "\t")
  ro <- utils::read.delim(file.path(FIG3_DIR, "fig11_nmf_classic_heatmap_readout_order.tsv"), sep = "\t")
  po <- utils::read.delim(file.path(DT_EXT, "scp682_sc11_predicted_observed_signal_seq_gse256403_hela_2024.tsv"), sep = "\t")
  modcol <- c(NMF1 = "#E64B35", NMF2 = "#4DBBD5", NMF3 = "#00A087")
  nC <- nrow(co); nR <- nrow(ro)
  m <- tapply(po$predicted, list(po$cell_id, po$target_id), function(x) x[1])
  m <- m[co$cell_id, ro$target_id, drop = FALSE]; z <- scale(m)
  zlim <- max(2.5, as.numeric(stats::quantile(abs(z), .98, na.rm = TRUE))); zt <- t(z)
  long <- data.frame(cell = rep(seq_len(nC), each = nR),
                     rd = factor(rep(ro$target_id, times = nC), levels = rev(ro$target_id)),
                     z = pmin(pmax(as.vector(zt), -zlim), zlim))
  cell_b <- cumsum(rle(co$dominant_component)$lengths); cell_b <- cell_b[-length(cell_b)] + 0.5
  mod_run <- rle(ro$module)$lengths; mod_b <- (nR - cumsum(mod_run)[-length(mod_run)]) + 0.5
  ycol <- unname(modcol[rev(ro$module)]); div5 <- c("#08306B", "#2171B5", "#FFFFFF", "#CB181D", "#67000D")
  p_main <- ggplot2::ggplot(long, ggplot2::aes(cell, rd, fill = z)) + ggplot2::geom_tile() +
    ggplot2::geom_vline(xintercept = cell_b, color = "white", linewidth = 0.6) +
    ggplot2::geom_hline(yintercept = mod_b, color = "white", linewidth = 0.6) +
    ggplot2::scale_fill_gradientn(colors = div5, limits = c(-zlim, zlim), oob = scales::squish, name = "z (pred)") +
    ggplot2::scale_x_continuous(expand = c(0, 0)) +
    ggplot2::scale_y_discrete(expand = c(0, 0), labels = function(x) fig3_short(x, 16)) +
    ggplot2::labs(x = NULL, y = NULL) + theme_fig3() +
    ggplot2::theme(axis.text.x = ggplot2::element_blank(), axis.ticks.x = ggplot2::element_blank(),
                   axis.text.y = ggplot2::element_text(size = 6, color = ycol),
                   axis.line = ggplot2::element_blank(), axis.ticks.y = ggplot2::element_blank(),
                   legend.position = "right", legend.key.width = ggplot2::unit(2.4, "mm"),
                   legend.key.height = ggplot2::unit(4, "mm"), plot.margin = ggplot2::margin(2, 8, 2, 4, "pt"))
  strip <- data.frame(cell = seq_len(nC), comp = factor(co$dominant_component, levels = names(modcol)))
  p_strip <- ggplot2::ggplot(strip, ggplot2::aes(cell, 1, fill = comp)) + ggplot2::geom_tile() +
    ggplot2::scale_fill_manual(values = modcol, name = "dominant") +
    ggplot2::scale_x_continuous(expand = c(0, 0)) + ggplot2::scale_y_continuous(expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = NULL) + theme_fig3() +
    ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                   axis.line = ggplot2::element_blank(), legend.position = "right",
                   legend.key.size = ggplot2::unit(2.6, "mm"), plot.margin = ggplot2::margin(2, 8, 0, 4, "pt"))
  cowplot::plot_grid(p_strip, p_main, ncol = 1, rel_heights = c(0.05, 1), align = "v", axis = "lr")
}

save_fig(panel_e(),          "panel_e_nmf_umap",            150, 52)
save_fig(.make_e_hex(),      "panel_f_phospho_rna_hexbin",   74, 62)
save_fig(.make_e_hallmark(), "panel_g_hallmark_enrichment",  80, 58)
save_fig(panel_h(),          "panel_h_nmf_classic_heatmap", 180, 74)
message("Done.")
