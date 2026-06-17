# fig08.R — Phospho readout 概览（按通路分组的 内部CV vs 最佳外部 双行热图）
#   取代旧"周期表"卡片版：每列一个 readout，两行 = internal CV / best external，
#   按通路分组、组内按最高 Spearman 降序，组间竖线分隔，列标签按通路上色。
# 数据：fig3/fig8_phospho_periodic_table.tsv

make_fig08 <- function() {
  d <- utils::read.delim(file.path(FIG3_DIR, "fig8_phospho_periodic_table.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  d$internal_cv_spearman  <- suppressWarnings(as.numeric(d$internal_cv_spearman))
  d$external_max_spearman <- suppressWarnings(as.numeric(d$external_max_spearman))
  d$grank <- match(d$group, FIG3_GROUP_ORDER)
  d$sortm <- pmax(d$internal_cv_spearman, d$external_max_spearman, na.rm = TRUE)
  d <- d[order(d$grank, -d$sortm), ]
  d$ord <- seq_len(nrow(d))
  d$tl  <- fig3_short(d$target_id, 16)

  long <- rbind(
    data.frame(ord = d$ord, metric = "internal CV",   val = d$internal_cv_spearman),
    data.frame(ord = d$ord, metric = "best external", val = d$external_max_spearman))
  long$metric <- factor(long$metric, levels = c("best external", "internal CV"))
  long$dark   <- !is.na(long$val) & long$val > 0.40

  bnd  <- which(diff(d$grank) != 0) + 0.5
  xcol <- unname(FIG3_GROUP_TEXT[d$group])
  # 组名标注（每组中点）
  gmid <- tapply(d$ord, d$group, mean)
  glab <- data.frame(x = as.numeric(gmid), g = names(gmid))
  glab <- glab[order(glab$x), ]

  ggplot2::ggplot(long, ggplot2::aes(x = ord, y = metric, fill = val)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.4) +
    ggplot2::geom_text(ggplot2::aes(label = ifelse(is.na(val), "·", sprintf("%.2f", val)),
                                    color = dark), size = 4 / PT, show.legend = FALSE) +
    ggplot2::geom_vline(xintercept = bnd, color = "#9A9A9A", linewidth = 0.4) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#222222"), guide = "none") +
    ggplot2::scale_fill_gradientn(colors = FIG3_SEQ_TEAL, limits = c(0, 0.65),
                                  oob = scales::squish, na.value = "#EEEEEE",
                                  name = "Spearman ρ") +
    ggplot2::scale_x_continuous(breaks = d$ord, labels = d$tl, expand = c(0, 0)) +
    ggplot2::scale_y_discrete(expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = NULL,
                  title = "Phospho readout overview — internal CV vs best external Spearman") +
    theme_fig3() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, vjust = 1,
                                          size = 4.6, color = xcol),
      axis.text.y = ggplot2::element_text(size = 6.5),
      axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
      legend.position = "right",
      legend.key.width = ggplot2::unit(2.6, "mm"),
      legend.key.height = ggplot2::unit(5, "mm"),
      plot.margin = ggplot2::margin(14, 8, 2, 4, "pt"))
}
