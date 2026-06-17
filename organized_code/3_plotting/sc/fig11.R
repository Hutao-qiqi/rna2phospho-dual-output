# fig11.R — Cell 风格 NMF 分块热图（HeLa）
#   主图：readout × cell（z 化预测），细胞按主导 NMF 成分分块、readout 按模块分块。
#   顶条：每细胞主导成分。底部：3 个 NMF 得分 vs 误差 hexbin。
# 数据：fig3/fig11_nmf_classic_heatmap_cell_order.tsv + _readout_order.tsv
#       + DT_EXT HeLa predicted_observed + fig10_nmf_vs_error_per_cell.tsv

make_fig11 <- function() {
  co <- utils::read.delim(file.path(FIG3_DIR, "fig11_nmf_classic_heatmap_cell_order.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  ro <- utils::read.delim(file.path(FIG3_DIR, "fig11_nmf_classic_heatmap_readout_order.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  po <- utils::read.delim(file.path(DT_EXT,
        "scp682_sc11_predicted_observed_signal_seq_gse256403_hela_2024.tsv"),
        sep = "\t", stringsAsFactors = FALSE)
  per <- utils::read.delim(file.path(FIG3_DIR, "fig10_nmf_vs_error_per_cell.tsv"),
                           sep = "\t", stringsAsFactors = FALSE)

  modcol <- c(NMF1 = "#E64B35", NMF2 = "#4DBBD5", NMF3 = "#00A087")
  nC <- nrow(co); nR <- nrow(ro)

  # z 矩阵：cells × readouts（显示顺序），按 readout 列 z 化
  m <- tapply(po$predicted, list(po$cell_id, po$target_id), function(x) x[1])
  m <- m[co$cell_id, ro$target_id, drop = FALSE]
  z <- scale(m)                                  # 每 readout 列 z 化
  zlim <- max(2.5, as.numeric(stats::quantile(abs(z), 0.98, na.rm = TRUE)))
  zt <- t(z)                                     # readouts × cells

  rd_levels <- rev(ro$target_id)
  long <- data.frame(
    cell = rep(seq_len(nC), each = nR),
    rd   = factor(rep(ro$target_id, times = nC), levels = rd_levels),
    z    = pmin(pmax(as.vector(zt), -zlim), zlim))

  # 边界
  cell_b <- cumsum(rle(co$dominant_component)$lengths)
  cell_b <- cell_b[-length(cell_b)] + 0.5
  mod_run <- rle(ro$module)$lengths
  mod_b   <- (nR - cumsum(mod_run)[-length(mod_run)]) + 0.5   # 反转 y 后的边界
  ycol <- unname(modcol[rev(ro$module)])
  div5 <- c("#08306B", "#2171B5", "#FFFFFF", "#CB181D", "#67000D")

  p_main <- ggplot2::ggplot(long, ggplot2::aes(cell, rd, fill = z)) +
    ggplot2::geom_tile() +
    ggplot2::geom_vline(xintercept = cell_b, color = "white", linewidth = 0.6) +
    ggplot2::geom_hline(yintercept = mod_b, color = "white", linewidth = 0.6) +
    ggplot2::scale_fill_gradientn(colors = div5, limits = c(-zlim, zlim),
                                  oob = scales::squish, name = "z (pred)") +
    ggplot2::scale_x_continuous(expand = c(0, 0)) +
    ggplot2::scale_y_discrete(expand = c(0, 0), labels = function(x) fig3_short(x, 16)) +
    ggplot2::labs(x = NULL, y = NULL) +
    theme_fig3() +
    ggplot2::theme(axis.text.x = ggplot2::element_blank(), axis.ticks.x = ggplot2::element_blank(),
                   axis.text.y = ggplot2::element_text(size = 6, color = ycol),
                   axis.line = ggplot2::element_blank(), axis.ticks.y = ggplot2::element_blank(),
                   legend.position = "right", legend.key.width = ggplot2::unit(2.4, "mm"),
                   legend.key.height = ggplot2::unit(4, "mm"),
                   plot.margin = ggplot2::margin(2, 8, 2, 4, "pt"))

  # 顶条：主导成分
  strip <- data.frame(cell = seq_len(nC), comp = factor(co$dominant_component, levels = names(modcol)))
  p_strip <- ggplot2::ggplot(strip, ggplot2::aes(cell, 1, fill = comp)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_manual(values = modcol, name = "dominant") +
    ggplot2::scale_x_continuous(expand = c(0, 0)) + ggplot2::scale_y_continuous(expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = NULL) +
    theme_fig3() +
    ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                   axis.line = ggplot2::element_blank(), legend.position = "right",
                   legend.key.size = ggplot2::unit(2.6, "mm"),
                   plot.margin = ggplot2::margin(2, 8, 0, 4, "pt"))

  # 底部 3 个 hexbin
  mk_hex <- function(i) {
    comp <- paste0("nmf", i)
    rho <- stats::cor(per[[comp]], per$error_high_var, method = "spearman")
    ggplot2::ggplot(per, ggplot2::aes(.data[[comp]], error_high_var)) +
      ggplot2::geom_hex(bins = 26) +
      ggplot2::scale_fill_gradientn(colors = FIG3_SEQ_TEAL, trans = "log10", name = "cells") +
      ggplot2::annotate("text", x = -Inf, y = Inf, hjust = -0.08, vjust = 1.2,
                        label = sprintf("ρ = %+.3f", rho), family = "Arial",
                        size = 5.4 / PT, color = paste0(modcol[paste0("NMF", i)])) +
      ggplot2::labs(x = paste0("NMF", i, " score"),
                    y = if (i == 1) "|pred − obs|" else NULL, title = NULL) +
      theme_fig3() + ggplot2::theme(legend.position = "none",
                                    plot.margin = ggplot2::margin(4, 6, 2, 4, "pt"))
  }
  hexrow <- cowplot::plot_grid(plotlist = lapply(1:3, mk_hex), ncol = 3, align = "hv")

  top_block <- cowplot::plot_grid(p_strip, p_main, ncol = 1,
                                  rel_heights = c(0.05, 1), align = "v", axis = "lr")
  body <- cowplot::plot_grid(top_block, hexrow, ncol = 1, rel_heights = c(1, 0.5))
  cowplot::ggdraw() +
    cowplot::draw_label("HeLa per-cell NMF — readouts × cells block-sorted, with prediction-error correlation",
                        x = 0.012, y = 0.992, hjust = 0, vjust = 1,
                        fontfamily = "Arial", size = 7, color = "#222222") +
    cowplot::draw_plot(body, x = 0, y = 0, width = 1, height = 0.975)
}
