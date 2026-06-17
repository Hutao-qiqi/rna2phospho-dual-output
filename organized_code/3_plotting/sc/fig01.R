# fig01.R — 通路注意力路由热图（8 pathway token × 56 readout）
#   number-in-cell + 顶部 per-readout max 柱 + 右侧 per-pathway mean 柱 + 列标签按通路上色。
# 数据：fig3/fig1_attention_heatmap_matrix.tsv (+ col_annotation)，列顺序已是聚类顺序。

make_fig01 <- function() {
  mat <- utils::read.delim(file.path(FIG3_DIR, "fig1_attention_heatmap_matrix.tsv"),
                           sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
  ann <- utils::read.delim(file.path(FIG3_DIR, "fig1_attention_heatmap_col_annotation.tsv"),
                           sep = "\t", stringsAsFactors = FALSE)
  paths <- mat$pathway
  M <- as.matrix(mat[, -1]); np <- nrow(M); nr <- ncol(M)
  path_lab <- gsub("_", " ", gsub("_axis", "", paths))
  rlab <- fig3_short(ann$target_id, 18)
  grp  <- ann$biological_group

  # 经典 R 红蓝发散（RdBu）：蓝=低 / 白=中位数 / 红=高
  vmin   <- as.numeric(stats::quantile(M, 0.02)); vmax <- as.numeric(stats::quantile(M, 0.98))
  center <- as.numeric(stats::median(M))
  hi_thr <- center + 0.55 * (vmax - center); lo_thr <- center - 0.55 * (center - vmin)

  long <- data.frame(
    pi   = rep(seq_len(np), times = nr),
    rj   = rep(seq_len(nr), each = np),
    attn = as.vector(M))
  long$path    <- factor(path_lab[long$pi], levels = rev(path_lab))
  long$readout <- factor(long$rj, levels = seq_len(nr), labels = rlab)
  long$attn_cl <- pmin(pmax(long$attn, vmin), vmax)
  long$dark    <- long$attn_cl >= hi_thr | long$attn_cl <= lo_thr   # 两端深色→白字

  xcol <- unname(FIG3_GROUP_TEXT[grp])

  p_heat <- ggplot2::ggplot(long, ggplot2::aes(readout, path, fill = attn_cl)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.25) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.2f", attn), color = dark),
                       size = 3.5 / PT, show.legend = FALSE) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#222222"), guide = "none") +
    ggplot2::scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                                  midpoint = center, limits = c(vmin, vmax),
                                  oob = scales::squish, name = "attention") +
    ggplot2::scale_x_discrete(expand = c(0, 0)) +
    ggplot2::scale_y_discrete(expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = NULL) +
    theme_fig3() +
    ggplot2::theme(
      axis.text.x  = ggplot2::element_text(angle = 45, hjust = 1, vjust = 1,
                                           size = 4.6, color = xcol),
      axis.text.y  = ggplot2::element_text(size = 7),
      axis.line    = ggplot2::element_blank(),
      axis.ticks   = ggplot2::element_blank(),
      legend.position = "right",
      legend.key.width = ggplot2::unit(2.4, "mm"),
      legend.key.height = ggplot2::unit(4.5, "mm"))

  # 右侧 per-pathway mean
  pm <- data.frame(path = factor(path_lab, levels = rev(path_lab)),
                   mean = rowMeans(M))
  p_right <- ggplot2::ggplot(pm, ggplot2::aes(mean, path)) +
    ggplot2::geom_col(fill = "#B2182B", width = 0.68) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", mean)),
                       hjust = -0.12, size = 4.6 / PT, color = "#333333") +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = c(0, 0.30))) +
    ggplot2::labs(x = "mean", y = NULL) +
    theme_fig3() +
    ggplot2::theme(axis.text.y = ggplot2::element_blank(),
                   axis.ticks.y = ggplot2::element_blank(),
                   axis.line.y = ggplot2::element_blank(),
                   plot.margin = ggplot2::margin(6, 8, 2, 2, "pt"))

  # 顶部 per-readout max
  tm <- data.frame(readout = factor(seq_len(nr), levels = seq_len(nr), labels = rlab),
                   maxv = apply(M, 2, max))
  p_top <- ggplot2::ggplot(tm, ggplot2::aes(readout, maxv)) +
    ggplot2::geom_col(fill = "#9A9A9A", width = 0.82) +
    ggplot2::scale_x_discrete(expand = c(0, 0)) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.12))) +
    ggplot2::labs(x = NULL, y = "max") +
    theme_fig3() +
    ggplot2::theme(axis.text.x = ggplot2::element_blank(),
                   axis.ticks.x = ggplot2::element_blank(),
                   axis.line.x = ggplot2::element_blank(),
                   plot.margin = ggplot2::margin(4, 8, 0, 8, "pt"))

  main_row <- cowplot::plot_grid(p_heat, p_right, nrow = 1,
                                 rel_widths = c(1, 0.14), align = "h", axis = "tb")
  top_row  <- cowplot::plot_grid(p_top, NULL, nrow = 1, rel_widths = c(1, 0.14))
  body <- cowplot::plot_grid(top_row, main_row, ncol = 1,
                             rel_heights = c(0.17, 1), align = "v", axis = "lr")
  cowplot::ggdraw() +
    cowplot::draw_label(
      "Pathway attention routing (per-pathway × per-readout, value shown in cells)",
      x = 0.012, y = 0.985, hjust = 0, vjust = 1,
      fontfamily = "Arial", size = 7, color = "#222222") +
    cowplot::draw_plot(body, x = 0, y = 0, width = 1, height = 0.965)
}
