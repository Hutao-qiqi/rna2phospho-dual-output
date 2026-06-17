# fig16.R — 细胞周期 marker 在各队列高误差 RNA-NMF 程序中的富集
#   a: cohort × {S, G2/M} 命中计数热图（v/总数）。
#   b: 每个命中 marker 在程序 top-200 列表中的排名散点（top-50 区高亮）。
# 数据：fig3/fig16_marker_counts.tsv + fig16_cell_cycle_marker_hits.tsv

make_fig16 <- function() {
  cnt  <- utils::read.delim(file.path(FIG3_DIR, "fig16_marker_counts.tsv"),
                            sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
  hits <- utils::read.delim(file.path(FIG3_DIR, "fig16_cell_cycle_marker_hits.tsv"),
                            sep = "\t", stringsAsFactors = FALSE)
  names(cnt)[1] <- "cohort"
  ord <- c("SIGNAL-seq HeLa", "GSE300551", "Vivo-seq Th17", "Blair", "SIGNAL-seq PDO/CAF")
  ord <- intersect(ord, intersect(cnt$cohort, unique(hits$display_cohort)))

  # ---- panel a ----
  totS <- cnt$total_S[1]; totG <- cnt$total_G2M[1]
  la <- rbind(
    data.frame(cohort = cnt$cohort, set = "S",    n = cnt$S,      denom = totS),
    data.frame(cohort = cnt$cohort, set = "G2/M", n = cnt[["G2/M"]], denom = totG))
  la$cohort <- factor(la$cohort, levels = rev(ord))
  la$set <- factor(la$set, levels = c("S", "G2/M"))
  la$dark <- la$n > 4
  p_a <- ggplot2::ggplot(la, ggplot2::aes(set, cohort, fill = n)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.5) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%d/%d", n, denom), color = dark),
                       size = 6 / PT, fontface = "bold", show.legend = FALSE) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#222222"), guide = "none") +
    ggplot2::scale_fill_gradientn(colors = c("#FFF5EB", "#FDD49E", "#F16913", "#A63603"),
                                  limits = c(0, max(8, max(la$n))), oob = scales::squish,
                                  name = "hits") +
    ggplot2::scale_x_discrete(position = "top", expand = c(0, 0)) +
    ggplot2::scale_y_discrete(expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = NULL, title = "Marker hits per cohort") +
    theme_fig3() +
    ggplot2::theme(axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                   axis.text.y = ggplot2::element_text(size = 6),
                   legend.position = "right", legend.key.width = ggplot2::unit(2.6, "mm"),
                   legend.key.height = ggplot2::unit(5, "mm"))

  # ---- panel b ----
  hits$cohort <- factor(hits$display_cohort, levels = rev(ord))
  hits$marker_set <- factor(hits$marker_set, levels = c("S", "G2/M"))
  p_b <- ggplot2::ggplot(hits, ggplot2::aes(rank_in_program, cohort)) +
    ggplot2::annotate("rect", xmin = 0, xmax = 50, ymin = -Inf, ymax = Inf,
                      fill = "#FCE5DC", alpha = 0.5) +
    ggplot2::geom_jitter(ggplot2::aes(fill = cohort, shape = marker_set),
                         height = 0.22, width = 0, size = 1.7, color = "white", stroke = 0.3) +
    ggplot2::scale_shape_manual(values = c("S" = 21, "G2/M" = 24), name = "marker set") +
    ggplot2::scale_fill_manual(values = c("SIGNAL-seq HeLa" = "#6CBFB5", "GSE300551" = "#1F3A5F",
                                          "Vivo-seq Th17" = "#9C8FC4", "Blair" = "#D4A56B",
                                          "SIGNAL-seq PDO/CAF" = "#C0392B"), guide = "none") +
    ggplot2::scale_x_continuous(limits = c(0, 205), expand = c(0, 0)) +
    ggplot2::labs(x = "Rank in program top-200 (1 = strongest); shaded = top-50",
                  y = NULL, title = "Rank of each cell-cycle marker hit") +
    theme_fig3() +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 6),
                   legend.position = "right") +
    ggplot2::guides(shape = ggplot2::guide_legend(override.aes = list(fill = "#888888")))

  cowplot::plot_grid(p_a, p_b, nrow = 1, rel_widths = c(1, 1.5))
}
