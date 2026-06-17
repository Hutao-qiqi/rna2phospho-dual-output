# fig12.R — phospho-NMF3 主导（应激）细胞 vs 其他细胞的表型差异
#   a: UMAP 高亮 NMF3-dominant 细胞。
#   b: 逐 readout 预测均值差（NMF3 − other）发散柱 + Mann-Whitney 显著性。
#   c: 5 个高方差 readout 上预测 vs 实测均值差（方向一致性）。
# 数据：fig3/fig9_hela_nmf_W_cell_scores.tsv + SRC_CACHE umap
#       + fig3/fig12_per_readout_predicted_shift.tsv + fig12_per_readout_observed_shift_high_variance.tsv

make_fig12 <- function() {
  W <- utils::read.delim(file.path(FIG3_DIR, "fig9_hela_nmf_W_cell_scores.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  um <- utils::read.delim(file.path(SRC_CACHE, "scp682_sc11_hela_scfoundation_umap.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  comp <- c("nmf1", "nmf2", "nmf3")
  W$dom <- comp[max.col(W[, comp], ties.method = "first")]
  d <- merge(W[, c("cell_id", "row_index", "dom")],
             um[, c("cell_id", "row_index", "umap1", "umap2")], by = c("cell_id", "row_index"))
  d$grp <- ifelse(d$dom == "nmf3", "NMF3-dominant", "other")
  n3 <- sum(d$grp == "NMF3-dominant"); no <- sum(d$grp == "other")

  p_a <- ggplot2::ggplot() +
    ggplot2::geom_point(data = d[d$grp == "other", ], ggplot2::aes(umap1, umap2),
                        color = "#D4D4D4", size = 0.5, stroke = 0) +
    ggplot2::geom_point(data = d[d$grp == "NMF3-dominant", ], ggplot2::aes(umap1, umap2),
                        color = "#C0392B", size = 0.9, stroke = 0.2) +
    ggplot2::annotate("text", x = -Inf, y = Inf, hjust = -0.08, vjust = 1.25,
                      label = sprintf("NMF3-dominant n=%d\nother n=%d", n3, no),
                      family = "Arial", size = 5.4 / PT, color = "#222222") +
    ggplot2::labs(x = "UMAP 1", y = "UMAP 2", title = "NMF3-dominant cells on HeLa UMAP") +
    theme_fig3() +
    ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank())

  # panel b
  b <- utils::read.delim(file.path(FIG3_DIR, "fig12_per_readout_predicted_shift.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  b <- b[order(b$mean_pred_shift), ]
  b$ord <- seq_len(nrow(b)); b$tl <- fig3_short(b$readout, 18)
  b$sig <- fig3_sig(b$mannwhitney_p)
  b$fillc <- ifelse(b$mean_pred_shift >= 0, "#C0392B", "#2874A6")
  p_b <- ggplot2::ggplot(b, ggplot2::aes(mean_pred_shift, ord)) +
    ggplot2::geom_vline(xintercept = 0, color = "#9A9A9A", linewidth = 0.3) +
    ggplot2::geom_col(ggplot2::aes(fill = fillc), width = 0.72, orientation = "y") +
    ggplot2::geom_text(ggplot2::aes(label = sig,
                                    x = mean_pred_shift + sign(mean_pred_shift) * 0.01),
                       size = 5 / PT, hjust = ifelse(b$mean_pred_shift >= 0, 0, 1)) +
    ggplot2::scale_fill_identity() +
    ggplot2::scale_y_continuous(breaks = b$ord, labels = b$tl, expand = c(0, 0.6)) +
    ggplot2::labs(x = "mean predicted shift (NMF3 − other)", y = NULL,
                  title = "Per-readout predicted shift") +
    theme_fig3() + ggplot2::theme(axis.text.y = ggplot2::element_text(size = 5.4),
                                  axis.ticks.y = ggplot2::element_blank())

  # panel c
  cc <- utils::read.delim(file.path(FIG3_DIR, "fig12_per_readout_observed_shift_high_variance.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  cc$tl <- fig3_short(cc$readout, 16)
  m <- rbind(data.frame(tl = cc$tl, kind = "observed", shift = cc$obs_shift, match = cc$pred_obs_sign_match),
             data.frame(tl = cc$tl, kind = "predicted", shift = cc$pred_shift, match = cc$pred_obs_sign_match))
  m$kind <- factor(m$kind, levels = c("observed", "predicted"))
  p_c <- ggplot2::ggplot(m, ggplot2::aes(shift, tl, fill = kind)) +
    ggplot2::geom_vline(xintercept = 0, color = "#9A9A9A", linewidth = 0.3) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.7), width = 0.62,
                      orientation = "y", color = "white", linewidth = 0.2) +
    ggplot2::scale_fill_manual(values = c("observed" = "#1F3A5F", "predicted" = "#ED8D5A"), name = NULL) +
    ggplot2::labs(x = "mean shift (NMF3 − other)", y = NULL,
                  title = "Observed vs predicted shift (high-variance)") +
    theme_fig3() + ggplot2::theme(axis.text.y = ggplot2::element_text(size = 6))

  cowplot::plot_grid(p_a, p_b, p_c, nrow = 1, rel_widths = c(1, 1.25, 1.1))
}
