# fig26.R — 注意力负对照：每读数「最强生物通路 token」vs「random_control token」配对对比
#   slopegraph：每读数一条线连接两端 + 两端箱线；生物 token 注意力系统性高于打乱的对照 token。
#   注标中位 Δ 与配对 Wilcoxon p（来自 random_control_attention_summary.tsv，dataset=="all"）。
# 数据：reviewer 表 random_control_attention_contrast.tsv + random_control_attention_summary.tsv

make_fig26 <- function() {
  d <- utils::read.delim(file.path(RV_DIR, "random_control_attention_contrast.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  s <- utils::read.delim(file.path(RV_DIR, "random_control_attention_summary.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  d <- d[d$dataset == "all", ]
  d$max_biological_attention <- as.numeric(d$max_biological_attention)
  d$random_control_attention <- as.numeric(d$random_control_attention)
  d$delta <- d$max_biological_attention - d$random_control_attention

  long <- rbind(
    data.frame(target = d$target_id, kind = "max biological",
               x = 1, value = d$max_biological_attention, up = d$delta > 0),
    data.frame(target = d$target_id, kind = "random control",
               x = 2, value = d$random_control_attention, up = d$delta > 0))
  long$kind <- factor(long$kind, levels = c("max biological", "random control"))

  sa  <- s[s$dataset == "all", ]
  med <- sprintf("%+.3f", as.numeric(sa$median_delta))
  pv  <- as.numeric(sa$paired_wilcoxon_p); e <- floor(log10(pv)); m <- sprintf("%.1f", pv / 10^e)
  npos <- sum(d$delta > 0); ntot <- nrow(d)
  # plotmath：三行，保证上标负指数正确
  lab <- paste0("atop(Delta[median]=='", med, "', atop(italic(P)=='", m, "'%*%10^", e,
                "~'(paired)', 'biological > control: ", npos, "/", ntot, "'))")

  ggplot2::ggplot() +
    ggplot2::geom_line(data = long,
                       ggplot2::aes(x = x, y = value, group = target, color = up),
                       linewidth = 0.3, alpha = 0.55) +
    ggplot2::geom_boxplot(data = long,
                          ggplot2::aes(x = x, y = value, group = kind),
                          width = 0.24, outlier.shape = NA, fill = NA,
                          color = "#333333", linewidth = 0.4) +
    ggplot2::geom_point(data = long,
                        ggplot2::aes(x = x, y = value, fill = kind),
                        shape = 21, color = "white", size = 1.5, stroke = 0.25,
                        position = ggplot2::position_nudge(
                          x = ifelse(long$x == 1, -0.16, 0.16))) +
    ggplot2::annotate("text", x = 1.5, y = max(long$value) + 0.004, label = lab,
                      parse = TRUE, hjust = 0.5, vjust = 1, family = "Arial", size = 4.9 / PT,
                      color = COL_TEXT) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "#51999F", `FALSE` = "#C9A24B"),
                                guide = "none") +
    ggplot2::scale_fill_manual(values = c("max biological" = "#ED8D5A",
                                          "random control" = "#BFBFBF"),
                               guide = "none") +
    ggplot2::scale_x_continuous(breaks = c(1, 2),
                                labels = c("max\nbiological", "random\ncontrol"),
                                limits = c(0.6, 2.4)) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.03, 0.12))) +
    ggplot2::labs(
      x = NULL, y = "Mean attention weight",
      title = "Attention negative control: biological vs random-control token",
      subtitle = "Each line = one of 56 readouts (pooled across cohorts); paired Wilcoxon signed-rank.") +
    theme_fig3() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(size = 6.2, lineheight = 0.9),
                   axis.ticks.x = ggplot2::element_blank())
}
