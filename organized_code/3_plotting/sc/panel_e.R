# panel_e.R — Fig 3 panel e：预测应激程序 × 独立 RNA 程序交叉验证（HeLa）
#   左：phospho-NMF3（模型预测的应激/高误差程序）vs RNA-NMF01 密度散点（hexbin）
#   右：RNA-NMF01 的 MSigDB Hallmark 富集 lollipop（−log10 FDR q）
#
# source 后调用 make_panel_e() 返回 cowplot 横排合成对象。
#
# 数据：
#   左 phospho: fig3/fig9_hela_nmf_W_cell_scores.tsv (cell_id, nmf1..nmf3)
#      RNA:     sc11_validation_rna_nmf_v4/source_data/
#               signal_seq_gse256403_hela_2024_rna_nmf_program_scores.tsv.gz (rna_nmf01)
#   右 hallmark: sc11_validation_rna_nmf_v4/source_data/rna_nmf_hallmark_enrichment.tsv
#
# 统计：phospho-NMF3 vs RNA-NMF01 Spearman ρ=+0.564, P=4.5×10⁻⁹⁷, n=1,143。

.PANEL_E_FIG3 <- file.path(
  "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11",
  "04_figure_source_data", "fig3")
.PANEL_E_RNA <- file.path(
  "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11",
  "04_figure_source_data", "sc11_validation_rna_nmf_v4", "source_data")

.make_e_hex <- function() {
  ph <- utils::read.delim(file.path(.PANEL_E_FIG3,
        "fig9_hela_nmf_W_cell_scores.tsv"), sep = "\t",
        stringsAsFactors = FALSE)
  rna <- utils::read.delim(gzfile(file.path(.PANEL_E_RNA,
        "signal_seq_gse256403_hela_2024_rna_nmf_program_scores.tsv.gz")),
        sep = "\t", stringsAsFactors = FALSE)
  m <- merge(ph[, c("cell_id", "nmf3")],
             rna[, c("cell_id", "rna_nmf01")], by = "cell_id")
  rho <- stats::cor(m$nmf3, m$rna_nmf01, method = "spearman")

  ggplot2::ggplot(m, ggplot2::aes(x = nmf3, y = rna_nmf01)) +
    ggplot2::geom_hex(bins = 32) +
    ggplot2::geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
                         color = "#ED8D5A", linewidth = 0.5) +
    ggplot2::scale_fill_gradientn(colors = FIG3_SEQ_TEAL, trans = "log10",
                                  name = "cells") +
    ggplot2::annotate("text", x = -Inf, y = Inf, hjust = -0.10, vjust = 1.22,
                      label = sprintf("ρ = %.3f\nP = 4.5×10⁻⁹⁷\nn = %s", rho,
                                      format(nrow(m), big.mark = ",")),
                      family = "Arial", size = 6 / PT, color = COL_TEXT) +
    ggplot2::labs(x = "phospho-NMF3 (stress) score",
                  y = "RNA-NMF01 score",
                  title = "phospho-NMF3 × RNA-NMF01 (HeLa)") +
    theme_fig3() +
    ggplot2::theme(
      legend.position   = "right",
      legend.key.width  = ggplot2::unit(2.4, "mm"),
      legend.key.height = ggplot2::unit(3.4, "mm"))
}

.make_e_hallmark <- function() {
  h <- utils::read.delim(file.path(.PANEL_E_RNA,
       "rna_nmf_hallmark_enrichment.tsv"), sep = "\t",
       stringsAsFactors = FALSE)
  h <- h[h$cohort_id == "signal_seq_gse256403_hela_2024" &
         h$component == "rna_nmf01", ]
  h <- h[order(-h$neg_log10_q), ][1:7, ]
  h$short <- gsub("HALLMARK_", "", h$gene_set)
  h$short <- factor(h$short, levels = rev(h$short))   # 最高排最上

  q05 <- -log10(0.05)

  ggplot2::ggplot(h, ggplot2::aes(x = neg_log10_q, y = short)) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, xend = neg_log10_q,
                                       y = short, yend = short),
                          color = "#888888", linewidth = 0.35) +
    ggplot2::geom_vline(xintercept = q05, linetype = "dashed",
                        color = COL_ZERO, linewidth = 0.3) +
    ggplot2::geom_point(ggplot2::aes(fill = neg_log10_q, size = overlap_count),
                        shape = 21, color = "black", stroke = 0.25) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%d", overlap_count)),
                       hjust = -0.9, size = 4.6 / PT, family = "Arial",
                       color = "#555555") +
    ggplot2::scale_fill_gradientn(colors = FIG3_RHO_COLORS, guide = "none") +
    ggplot2::scale_size_continuous(range = c(1.8, 3.8), name = "overlap genes",
                                   breaks = c(4, 6, 8)) +
    ggplot2::scale_x_continuous(limits = c(0, 2.2),
                                expand = ggplot2::expansion(mult = c(0.01, 0.14))) +
    ggplot2::labs(x = "−log10(FDR q)", y = NULL,
                  title = "RNA-NMF01 hallmark enrichment",
                  subtitle = "Size = overlap genes; dashed = q 0.05.") +
    theme_fig3() +
    ggplot2::theme(legend.position = "right")
}

make_panel_e <- function() {
  cowplot::plot_grid(
    .make_e_hex(), .make_e_hallmark(),
    nrow = 1, rel_widths = c(1, 1.18))
}
