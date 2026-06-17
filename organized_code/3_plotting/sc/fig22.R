# fig22.R — 组件消融：配对 Δ(variant − full)，只在「共享读数」上配对（诚实处理子集混淆）
#   三个 variant 各一列：− pathway attn / − expanded graph / − scFoundation。
#   每队列水平柱 = 中位 Δ；叠加逐读数配对点；柱尾标 n（共享读数数）。
#   负 Δ = 去掉该组件后 ρ 下降（组件有贡献）。
#   注：scFoundation 变体在部分队列仅 n=1 共享读数（HeLa/Vivo/Blair），点即柱，勿过度解读。
# 数据：fig3/fig22_component_ablation_paired_{per_target,summary}.tsv（由 prep/reviewer_ed_prep.py 生成）

make_fig22 <- function() {
  per  <- utils::read.delim(file.path(FIG3_DIR, "fig22_component_ablation_paired_per_target.tsv"),
                            sep = "\t", stringsAsFactors = FALSE)
  summ <- utils::read.delim(file.path(FIG3_DIR, "fig22_component_ablation_paired_summary.tsv"),
                            sep = "\t", stringsAsFactors = FALSE)
  per$delta  <- as.numeric(per$delta)
  summ$median_delta <- as.numeric(summ$median_delta)
  summ$n_shared     <- as.integer(summ$n_shared)

  coh_short <- c("SIGNAL-seq HeLa" = "HeLa", "GSE300551" = "GSE300551",
                 "Blair" = "Blair", "Vivo-seq Th17" = "Vivo-Th17",
                 "SIGNAL-seq PDO/CAF" = "PDO/CAF")
  coh_ord   <- c("HeLa", "GSE300551", "Blair", "Vivo-Th17", "PDO/CAF")
  var_ord   <- c("- pathway attn", "- expanded graph", "- scFoundation")
  var_lab   <- c("- pathway attn" = "− pathway attn",
                 "- expanded graph" = "− expanded graph",
                 "- scFoundation" = "− scFoundation")

  fx <- function(d) {
    d$coh <- factor(coh_short[d$cohort_name], levels = rev(coh_ord))
    d$var <- factor(d$variant, levels = var_ord)
    d
  }
  per  <- fx(per); summ <- fx(summ)
  summ$dir <- ifelse(summ$median_delta < 0, "down", "up")
  # n 标签：柱尾外侧 + 上移到行间隙，避开点云
  summ$nx <- summ$median_delta + ifelse(summ$median_delta < 0, -0.006, 0.006)
  summ$nhj <- ifelse(summ$median_delta < 0, 1, 0)

  ggplot2::ggplot() +
    ggplot2::geom_vline(xintercept = 0, color = COL_ZERO, linewidth = 0.3) +
    ggplot2::geom_col(data = summ,
                      ggplot2::aes(x = median_delta, y = coh, fill = dir),
                      width = 0.62, color = "black", linewidth = 0.22) +
    ggplot2::geom_point(data = per,
                        ggplot2::aes(x = delta, y = coh),
                        position = ggplot2::position_jitter(height = 0.10, width = 0, seed = 7),
                        shape = 21, fill = "#FFFFFF", color = "#3A3A3A",
                        size = 0.85, stroke = 0.3, alpha = 0.85) +
    ggplot2::geom_text(data = summ,
                       ggplot2::aes(x = nx, y = coh, label = paste0("n=", n_shared),
                                    hjust = nhj),
                       position = ggplot2::position_nudge(y = 0.34),
                       size = 4.4 / PT, family = "Arial", color = COL_SUB) +
    ggplot2::facet_wrap(~ var, nrow = 1,
                        labeller = ggplot2::as_labeller(var_lab)) +
    ggplot2::scale_fill_manual(
      values = c(down = "#4198AC", up = "#DBCB92"),
      breaks = c("down", "up"),
      labels = c("removal lowers ρ (component helps)", "removal raises ρ"),
      name = NULL) +
    ggplot2::scale_x_continuous(limits = c(-0.345, 0.165),
                                breaks = c(-0.3, -0.2, -0.1, 0, 0.1),
                                expand = ggplot2::expansion(mult = c(0.02, 0.02))) +
    ggplot2::labs(
      x = "Paired Δ median Spearman (variant − full), shared readouts",
      y = NULL,
      title = "Component ablation: paired Δ on shared readouts",
      subtitle = "Points = per-readout Δ (variant − full); negative = removing the component lowers ρ. n = shared readouts.") +
    theme_fig3() +
    ggplot2::theme(
      panel.spacing = ggplot2::unit(3.5, "mm"),
      axis.text.y = ggplot2::element_text(size = 6.0),
      legend.position = "bottom")
}
