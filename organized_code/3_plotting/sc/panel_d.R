# panel_d.R — 扩展 ScNET 图残差消融（matched），GSE300551-primary。
#
# 主结论只在 GSE300551（唯一多读数 powered 外部队列）上算：
#   逐读数 no-graph − full 的 Δ（20 读数，几乎全为负 = 图有贡献）。
#   headline：full 0.307 → no-graph 0.258，paired 中位 Δ −0.029，单侧 Wilcoxon P = 9.54×10⁻⁷。
# HeLa/Vivo/Blair 仅作 anchor（侧栏文字），不进入队列级结论；不报 macro（避免与 Fig 3b 的 0.271 撞车）。
#
# 数据（权威 20260522 模型）：
#   paper_final/fig5/sc_kirc_rps6_validation/tables/
#     scp682_sc_graph_ablation_gse300551_per_target.tsv   GSE300551 逐读数 full/variant/delta
#     scp682_sc_graph_ablation_reporting_split.tsv          各队列 full/no-graph 中位 + 用途

.GABL_DIR <- file.path("E:/data/gongke/TCGA-TCPA/paper_final",
                       "fig5", "sc_kirc_rps6_validation", "tables")

make_panel_d <- function() {
  pt <- utils::read.delim(file.path(.GABL_DIR, "scp682_sc_graph_ablation_gse300551_per_target.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  sp <- utils::read.delim(file.path(.GABL_DIR, "scp682_sc_graph_ablation_reporting_split.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  pt$delta <- as.numeric(pt$delta)
  pt <- pt[is.finite(pt$delta), ]
  pt <- pt[order(pt$delta), ]
  pt$tl  <- fig3_short(pt$target_id, 20)
  pt$ord <- factor(seq_len(nrow(pt)))

  gse <- sp[sp$reporting_scope == "multi_readout_primary", ][1, ]
  full_med <- as.numeric(gse$full_model_median_spearman)
  ng_med   <- as.numeric(gse$graph_removed_median_spearman)
  pmed     <- as.numeric(gse$median_per_target_delta_graph_removed_minus_full)

  # anchor 侧栏文字
  anc <- sp[sp$reporting_scope == "anchor_or_low_n", ]
  anc_nice <- c("SIGNAL-seq HeLa" = "HeLa", "Vivo-seq Th17" = "Vivo", "Blair" = "Blair")
  anc$lab <- sprintf("%s (n=%d): %.3f→%.3f",
                     anc_nice[anc$cohort_name], as.integer(anc$n_readouts),
                     as.numeric(anc$full_model_median_spearman),
                     as.numeric(anc$graph_removed_median_spearman))
  anc_txt <- paste0("anchors (supporting only):\n",
                    paste(anc$lab[match(c("SIGNAL-seq HeLa","Vivo-seq Th17","Blair"),
                                        anc$cohort_name)], collapse = "\n"))

  head_l1 <- sprintf("GSE300551 (20 readouts): %.3f → %.3f", full_med, ng_med)
  # 第二行用 plotmath，保证上标负指数正确（Arial 下 Unicode 上标减号是豆腐块）
  head_l2 <- paste0("'paired median '*Delta*' ", sprintf("%+.3f", pmed),
                    ", '*italic(P)*' = 9.54'%*%10^-7")

  xlo <- min(pt$delta) - 0.03

  ggplot2::ggplot(pt, ggplot2::aes(y = ord)) +
    ggplot2::geom_vline(xintercept = 0, color = COL_ZERO, linewidth = 0.3) +
    ggplot2::geom_vline(xintercept = pmed, color = "#4198AC", linewidth = 0.35,
                        linetype = "dashed") +
    ggplot2::geom_segment(ggplot2::aes(x = 0, xend = delta, yend = ord),
                          color = "#9FB6BC", linewidth = 0.45) +
    ggplot2::geom_point(ggplot2::aes(x = delta), shape = 21, fill = "#4198AC",
                        color = "black", size = 1.5, stroke = 0.25) +
    ggplot2::annotate("text", x = xlo, y = nrow(pt) + 1.0, label = head_l1,
                      hjust = 0, vjust = 1, family = "Arial", size = 5.0 / PT,
                      color = COL_TEXT) +
    ggplot2::annotate("text", x = xlo, y = nrow(pt) - 0.1, label = head_l2,
                      parse = TRUE, hjust = 0, vjust = 1, family = "Arial",
                      size = 5.0 / PT, color = COL_TEXT) +
    ggplot2::annotate("text", x = -0.001, y = 2.4, label = anc_txt,
                      hjust = 1, vjust = 1, family = "Arial", size = 4.4 / PT,
                      color = COL_SUB, lineheight = 1.1) +
    ggplot2::scale_y_discrete(labels = pt$tl, expand = ggplot2::expansion(add = c(0.6, 1.8))) +
    ggplot2::scale_x_continuous(limits = c(xlo, 0.02),
                                breaks = c(-0.2, -0.15, -0.1, -0.05, 0),
                                expand = c(0, 0)) +
    ggplot2::labs(
      x = "Per-readout Δ Spearman (no-graph − full)", y = NULL,
      title = "Site-graph ablation — GSE300551 (matched)",
      subtitle = "Dashed = paired median Δ; negative = removing the graph lowers ρ.") +
    theme_fig3() +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 5.0),
                   axis.ticks.y = ggplot2::element_blank())
}
