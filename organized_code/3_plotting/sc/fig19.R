# fig19.R — Per-readout 难度地图
#   a: 56 readout 按外部均值 ρ 排序的水平柱（按通路上色）+ 内部 CV 中位标记。
#   b: 内部 CV vs 外部均值 散点（对角线 + Spearman + 极端点标注）。
# 数据：fig3/fig19_per_readout_difficulty.tsv

make_fig19 <- function() {
  d <- utils::read.delim(file.path(FIG3_DIR, "fig19_per_readout_difficulty.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  for (cc in c("internal_cv_median", "external_mean", "external_max", "external_min"))
    d[[cc]] <- suppressWarnings(as.numeric(d[[cc]]))
  d$group <- factor(d$group, levels = FIG3_GROUP_ORDER)

  # ---- panel a ----
  a <- d
  a$sort_key <- ifelse(is.na(a$external_mean), -1, a$external_mean)
  a <- a[order(a$sort_key), ]            # 升序 → 高分在上（y 轴）
  a$ord <- seq_len(nrow(a))
  a$tl  <- fig3_short(a$target_id, 22)
  p_a <- ggplot2::ggplot(a) +
    ggplot2::geom_vline(xintercept = 0, color = "#9A9A9A", linewidth = 0.3) +
    ggplot2::geom_col(ggplot2::aes(x = ifelse(is.na(external_mean), 0, external_mean),
                                   y = ord, fill = group),
                      width = 0.72, color = "white", linewidth = 0.2,
                      orientation = "y") +
    ggplot2::geom_point(data = a[!is.na(a$internal_cv_median), ],
                        ggplot2::aes(x = internal_cv_median, y = ord),
                        shape = 124, size = 1.7, color = "#222222") +
    ggplot2::scale_fill_manual(values = FIG3_GROUP_FILL, name = "pathway", drop = TRUE) +
    ggplot2::scale_y_continuous(breaks = a$ord, labels = a$tl, expand = c(0, 0.6)) +
    ggplot2::scale_x_continuous(limits = c(-0.1, max(0.6, max(a$external_mean, na.rm = TRUE) * 1.05))) +
    ggplot2::labs(x = "Spearman ρ", y = NULL,
                  title = "Per-readout difficulty (bar = external mean; tick = internal CV)") +
    theme_fig3() +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 5),
                   axis.ticks.y = ggplot2::element_blank(),
                   legend.position = "none")

  # ---- panel b ----
  b <- d[is.finite(d$internal_cv_median) & is.finite(d$external_mean), ]
  rho <- stats::cor(b$internal_cv_median, b$external_mean, method = "spearman")
  lo <- min(b$internal_cv_median, b$external_mean) - 0.05
  hi <- max(b$internal_cv_median, b$external_mean) + 0.05
  b$gap <- b$external_mean - b$internal_cv_median
  ext <- rbind(utils::head(b[order(-b$gap), ], 3), utils::head(b[order(b$gap), ], 3))
  ext$tl <- fig3_short(ext$target_id, 15)
  p_b <- ggplot2::ggplot(b, ggplot2::aes(internal_cv_median, external_mean)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                         color = "#9A9A9A", linewidth = 0.4) +
    ggplot2::geom_point(ggplot2::aes(fill = group), shape = 21, color = "white",
                        size = 2.2, stroke = 0.3) +
    ggrepel::geom_text_repel(data = ext, ggplot2::aes(label = tl),
                             size = 5 / PT, family = "Arial", color = "#222222",
                             min.segment.length = 0, segment.size = 0.2, max.overlaps = Inf) +
    ggplot2::annotate("text", x = -Inf, y = Inf, hjust = -0.1, vjust = 1.2,
                      label = sprintf("ρ = %+.3f\nn = %d", rho, nrow(b)),
                      family = "Arial", size = 5.4 / PT, color = "#222222") +
    ggplot2::scale_fill_manual(values = FIG3_GROUP_FILL, name = "pathway", drop = TRUE) +
    ggplot2::scale_x_continuous(limits = c(lo, hi)) +
    ggplot2::scale_y_continuous(limits = c(lo, hi)) +
    ggplot2::labs(x = "Internal CV median ρ", y = "External mean ρ",
                  title = "Internal vs external per readout") +
    theme_fig3() +
    ggplot2::theme(legend.position = "bottom", legend.key.size = ggplot2::unit(2.6, "mm")) +
    ggplot2::guides(fill = ggplot2::guide_legend(nrow = 3, byrow = TRUE,
                                                 override.aes = list(size = 2.2)))

  # a 占满左侧全高；b 放右上角（散点天然方形，不强行拉伸）
  cowplot::ggdraw() +
    cowplot::draw_plot(p_a, x = 0.00, y = 0.00, width = 0.52, height = 1.00) +
    cowplot::draw_plot(p_b, x = 0.54, y = 0.26, width = 0.46, height = 0.72)
}
