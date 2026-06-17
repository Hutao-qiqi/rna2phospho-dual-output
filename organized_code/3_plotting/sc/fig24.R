# fig24.R — 5 折交叉验证：逐读数稳定性（pooled "all" 折）
#   水平 pointrange：点 = 跨 5 折中位 ρ，须 = min–max，颜色 = 折间 SD（越小越稳）。
#   38 个监督读数，按中位 ρ 排序。
# 数据：reviewer 表 fivefold_stability_by_readout.tsv（filter test_dataset=="all"）

make_fig24 <- function() {
  d <- utils::read.delim(file.path(RV_DIR, "fivefold_stability_by_readout.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  d <- d[d$test_dataset == "all", ]
  for (c in c("median_spearman", "sd_spearman", "min_spearman", "max_spearman"))
    d[[c]] <- as.numeric(d[[c]])
  d <- d[is.finite(d$median_spearman), ]
  d <- d[order(d$median_spearman), ]
  d$tl  <- fig3_short(d$target_id, 20)
  d$ord <- factor(seq_len(nrow(d)))
  d$lab <- d$tl

  xlo <- min(d$min_spearman, 0) - 0.02
  xhi <- max(d$max_spearman) + 0.02

  ggplot2::ggplot(d, ggplot2::aes(y = ord)) +
    ggplot2::geom_vline(xintercept = 0, color = COL_ZERO, linewidth = 0.3,
                        linetype = "dashed") +
    ggplot2::geom_linerange(ggplot2::aes(xmin = min_spearman, xmax = max_spearman),
                            color = "#9FB6BC", linewidth = 0.5) +
    ggplot2::geom_point(ggplot2::aes(x = median_spearman, fill = sd_spearman),
                        shape = 21, color = "black", size = 1.7, stroke = 0.25) +
    ggplot2::scale_fill_gradientn(
      colours = c("#1F5A66", "#4198AC", "#BFDFD2", "#ECB66C", "#ED8D5A"),
      name = "fold SD",
      breaks = c(0.02, 0.06, 0.10),
      guide = ggplot2::guide_colorbar(barwidth = ggplot2::unit(16, "mm"),
                                      barheight = ggplot2::unit(1.8, "mm"),
                                      title.vjust = 1)) +
    ggplot2::scale_y_discrete(labels = d$lab, expand = ggplot2::expansion(add = 0.7)) +
    ggplot2::scale_x_continuous(limits = c(xlo, xhi),
                                breaks = seq(0, 0.5, 0.1),
                                expand = c(0, 0)) +
    ggplot2::labs(
      x = "Per-readout Spearman ρ (5-fold CV)", y = NULL,
      title = "5-fold cross-validation stability per readout",
      subtitle = "Point = median across 5 folds; bar = min–max; colour = fold-to-fold SD.") +
    theme_fig3() +
    ggplot2::theme(
      axis.text.y  = ggplot2::element_text(size = 5.0),
      axis.ticks.y = ggplot2::element_blank(),
      legend.position = "bottom")
}
