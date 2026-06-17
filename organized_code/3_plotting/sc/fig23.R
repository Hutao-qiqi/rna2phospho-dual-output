# fig23.R — GNN 校正头 vs site-aware MLP（同特征、同折、配对）
#   每 readout 一点（跨 5 折取中位）：y = SCP682-SC，x = site-aware MLP。
#   对角线 y=x；点在线上方 = GNN 头更优。按数据集上色，注标中位 Δ + 配对 Wilcoxon p。
# 数据：fig3/fig23_gnn_vs_mlp_per_target.tsv（prep 生成）+ reviewer 表 gnn_vs_site_aware_mlp_summary.tsv

make_fig23 <- function() {
  d <- utils::read.delim(file.path(FIG3_DIR, "fig23_gnn_vs_mlp_per_target.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  s <- utils::read.delim(file.path(RV_DIR, "gnn_vs_site_aware_mlp_summary.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  d$scp682_sc_median <- as.numeric(d$scp682_sc_median)
  d$mlp_median       <- as.numeric(d$mlp_median)

  ds_nice <- c(iccite_seq_tcell_2025 = "iCCITE T-cell",
               qurie_seq_bjab_2021   = "QuRIE BJAB")
  ds_ord  <- c("iCCITE T-cell", "QuRIE BJAB")
  d$ds <- factor(ds_nice[d$test_dataset], levels = ds_ord)
  cols <- stats::setNames(unname(FIG3_PAL[c("deepteal", "ours")]), ds_ord)

  # 注释：每数据集中位 Δ + Wilcoxon p（plotmath，保证上标负指数正确排版）
  s$ds <- ds_nice[s$test_dataset]
  mk <- function(z) {
    r <- s[s$ds == z, ]
    p <- as.numeric(r$paired_wilcoxon_p); e <- floor(log10(p)); m <- sprintf("%.1f", p / 10^e)
    dm <- sprintf("%+.3f", as.numeric(r$median_delta)); n <- as.integer(r$n_targets)
    paste0("'", z, ":'~~Delta[med]=='", dm, "'*','~~italic(P)=='", m,
           "'%*%10^", e, "~~'(n=", n, ")'")
  }

  lim <- c(0, 0.58)
  ggplot2::ggplot(d, ggplot2::aes(x = mlp_median, y = scp682_sc_median, fill = ds)) +
    ggplot2::annotate("segment", x = lim[1], y = lim[1], xend = lim[2], yend = lim[2],
                      color = "#9A9A9A", linewidth = 0.35, linetype = "dashed") +
    ggplot2::geom_point(shape = 21, color = "white", size = 1.9, stroke = 0.3, alpha = 0.92) +
    ggplot2::annotate("text", x = lim[1] + 0.012, y = lim[2], label = mk("iCCITE T-cell"),
                      parse = TRUE, hjust = 0, vjust = 1, family = "Arial", size = 5 / PT,
                      color = COL_TEXT) +
    ggplot2::annotate("text", x = lim[1] + 0.012, y = lim[2] - 0.042, label = mk("QuRIE BJAB"),
                      parse = TRUE, hjust = 0, vjust = 1, family = "Arial", size = 5 / PT,
                      color = COL_TEXT) +
    ggplot2::annotate("text", x = lim[2], y = lim[1] + 0.045, label = "above line:\nGNN head wins",
                      hjust = 1, vjust = 0, family = "Arial", size = 4.8 / PT,
                      fontface = "italic", color = COL_SUB, lineheight = 0.95) +
    ggplot2::scale_fill_manual(values = cols, name = NULL) +
    ggplot2::scale_x_continuous(limits = lim, breaks = seq(0, 0.5, 0.1), expand = c(0, 0)) +
    ggplot2::scale_y_continuous(limits = lim, breaks = seq(0, 0.5, 0.1), expand = c(0, 0)) +
    ggplot2::coord_equal() +
    ggplot2::labs(
      x = "site-aware MLP — per-readout median ρ",
      y = "SCP682-SC (GNN head) — median ρ",
      title = "GNN correction head vs site-aware MLP",
      subtitle = "Matched features, folds and readouts; each point = one readout (5-fold CV median).") +
    theme_fig3() +
    ggplot2::theme(legend.position = "bottom")
}
