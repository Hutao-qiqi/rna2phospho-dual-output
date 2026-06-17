# fig14.R — HeLa RNA-NMF 独立验证 phospho-NMF3"应激"亚群
#   a: phospho-NMF × RNA-NMF per-cell Spearman 热图（3×10，显著标星）。
#   b: phospho-NMF3 vs RNA-NMF01 hexbin（ρ/P）。
#   c: top-50 细胞集合在 UMAP 上的重叠（Fisher）。
#   d: RNA-NMF01 Hallmark 富集 top-10。
# 数据：fig3/fig14_phospho_vs_rna_nmf_spearman.tsv (+ _pvalue, _top50_overlap, merge.gz) + RNA_SD hallmark

make_fig14 <- function() {
  M <- utils::read.delim(file.path(FIG3_DIR, "fig14_phospho_vs_rna_nmf_spearman.tsv"),
                         sep = "\t", row.names = 1, check.names = FALSE)
  P <- utils::read.delim(file.path(FIG3_DIR, "fig14_phospho_vs_rna_nmf_pvalue.tsv"),
                         sep = "\t", row.names = 1, check.names = FALSE)
  ph <- rownames(M); rn <- colnames(M)
  ov <- utils::read.delim(file.path(FIG3_DIR, "fig14_top50_overlap.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  mg <- utils::read.delim(gzfile(file.path(FIG3_DIR, "fig14_hela_per_cell_phospho_rna_merge.tsv.gz")),
                          sep = "\t", stringsAsFactors = FALSE)

  # ---- a heatmap ----
  longa <- data.frame(
    ph  = rep(factor(ph, levels = rev(ph)), times = length(rn)),
    rn  = rep(factor(rn, levels = rn), each = length(ph)),
    val = as.vector(as.matrix(M)),
    p   = as.vector(as.matrix(P)))
  longa$star <- fig3_sig(longa$p)
  longa$dark <- abs(longa$val) > 0.25
  p_a <- ggplot2::ggplot(longa, ggplot2::aes(rn, ph, fill = val)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.3) +
    ggplot2::geom_text(ggplot2::aes(label = star, color = dark), size = 5 / PT, show.legend = FALSE) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "#222222"), guide = "none") +
    ggplot2::scale_fill_gradient2(low = "#3C7E8C", mid = "white", high = "#ED8D5A",
                                  midpoint = 0, limits = c(-0.4, 0.4), oob = scales::squish, name = "ρ") +
    ggplot2::scale_x_discrete(expand = c(0, 0)) + ggplot2::scale_y_discrete(expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = NULL, title = "phospho-NMF × RNA-NMF (Spearman)") +
    theme_fig3() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, vjust = 1, size = 5),
                   axis.line = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                   legend.position = "right", legend.key.width = ggplot2::unit(2.2, "mm"),
                   legend.key.height = ggplot2::unit(3.4, "mm"))

  # ---- b hexbin ----
  rho <- stats::cor(mg$NMF3, mg$rna_nmf01, method = "spearman")
  p_b <- ggplot2::ggplot(mg, ggplot2::aes(NMF3, rna_nmf01)) +
    ggplot2::geom_hex(bins = 30) +
    ggplot2::geom_smooth(method = "lm", formula = y ~ x, se = FALSE, color = "#ED8D5A", linewidth = 0.5) +
    ggplot2::scale_fill_gradientn(colors = FIG3_SEQ_TEAL, trans = "log10", name = "cells") +
    ggplot2::annotate("text", x = -Inf, y = Inf, hjust = -0.08, vjust = 1.2,
                      label = sprintf("ρ = %+.3f\nn = %s", rho, format(nrow(mg), big.mark = ",")),
                      family = "Arial", size = 5.4 / PT, color = "#222222") +
    ggplot2::labs(x = "phospho-NMF3 score", y = "RNA-NMF01 score",
                  title = "phospho-NMF3 vs RNA-NMF01") +
    theme_fig3() + ggplot2::theme(legend.position = "right",
                                  legend.key.width = ggplot2::unit(2.2, "mm"),
                                  legend.key.height = ggplot2::unit(3.4, "mm"))

  # ---- c UMAP top-50 overlap ----
  N <- 50
  ph_top <- mg$cell_id[order(-mg$NMF3)][1:N]
  rna_top <- mg$cell_id[order(-mg$rna_nmf01)][1:N]
  both <- intersect(ph_top, rna_top)
  mg$status <- "other"
  mg$status[mg$cell_id %in% setdiff(ph_top, rna_top)] <- "phospho-only"
  mg$status[mg$cell_id %in% setdiff(rna_top, ph_top)] <- "rna-only"
  mg$status[mg$cell_id %in% both] <- "both"
  mg$status <- factor(mg$status, levels = c("other", "phospho-only", "rna-only", "both"))
  fp <- ov$fisher_p_one_sided[1]
  p_c <- ggplot2::ggplot(mg[order(mg$status), ], ggplot2::aes(umap1, umap2, color = status, size = status)) +
    ggplot2::geom_point(stroke = 0) +
    ggplot2::scale_color_manual(values = c(other = "#D8D8D8", `phospho-only` = "#1F78B4",
                                           `rna-only` = "#33A02C", both = "#E31A1C"), name = NULL) +
    ggplot2::scale_size_manual(values = c(other = 0.45, `phospho-only` = 1.0,
                                          `rna-only` = 1.0, both = 1.6), guide = "none") +
    ggplot2::labs(x = "UMAP 1", y = "UMAP 2",
                  title = sprintf("top-50 sets (overlap=%d, Fisher P=%.1e)", length(both), fp)) +
    theme_fig3() +
    ggplot2::theme(axis.text = ggplot2::element_blank(), axis.ticks = ggplot2::element_blank(),
                   legend.position = "right") +
    ggplot2::guides(color = ggplot2::guide_legend(override.aes = list(size = 1.8)))

  # ---- d hallmark ----
  h <- utils::read.delim(file.path(RNA_SD, "rna_nmf_hallmark_enrichment.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  h <- h[h$cohort_id == "signal_seq_gse256403_hela_2024" & h$component == "rna_nmf01", ]
  h <- h[order(-h$neg_log10_q), ][1:10, ]
  h$short <- factor(gsub("HALLMARK_", "", h$gene_set), levels = rev(gsub("HALLMARK_", "", h$gene_set)))
  p_d <- ggplot2::ggplot(h, ggplot2::aes(neg_log10_q, short)) +
    ggplot2::geom_col(fill = "#C0392B", width = 0.72) +
    ggplot2::geom_vline(xintercept = -log10(0.05), linetype = "dashed", color = "#9A9A9A", linewidth = 0.3) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("k=%d", overlap_count)), hjust = -0.2,
                       size = 4.6 / PT, color = "#555555") +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = c(0, 0.12))) +
    ggplot2::labs(x = "−log10(FDR q)", y = NULL,
                  title = "RNA-NMF01 hallmark enrichment (top 10) — validates G2M/mitotic 'stress'") +
    theme_fig3()

  top <- cowplot::plot_grid(p_a, p_b, p_c, nrow = 1, rel_widths = c(1.05, 0.95, 1.25))
  cowplot::plot_grid(top, p_d, ncol = 1, rel_heights = c(1, 0.7))
}
