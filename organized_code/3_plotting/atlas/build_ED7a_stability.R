#!/usr/bin/env Rscript
# ED Fig 7a — module-number stability: k=20 and k=40 module x cancer heatmaps
# (same diagonal layout + RdBu palette as the main-figure atlas)
suppressPackageStartupMessages({library(ComplexHeatmap); library(circlize); library(grid)})
ROOT <- "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
col_fun <- colorRamp2(c(-1.5,-0.5,0,1.5,3), c("#053061","#4393C3","#F7F7F7","#D6604D","#B2182B"))

for (K in c(20, 40)) {
  mbc  <- as.matrix(read.delim(sprintf("%s/results/module_by_cancer_median_k%d.tsv",ROOT,K), row.names=1, check.names=FALSE))
  summ <- read.delim(sprintf("%s/results/module_summary_k%d.tsv",ROOT,K), stringsAsFactors=FALSE)
  mat <- mbc[summ$module, , drop=FALSE]
  pref <- summ$pref_cancer
  cancer_order <- c(unique(pref), setdiff(colnames(mat), unique(pref)))
  mat <- mat[, cancer_order, drop=FALSE]
  rownames(mat) <- paste0(summ$module, "  ", summ$pref_cancer)

  ht <- Heatmap(mat, name="activity (z)", col=col_fun,
    cluster_rows=FALSE, cluster_columns=FALSE,
    row_names_side="right", row_names_gp=gpar(fontsize=5, fontfamily="Arial"),
    column_names_gp=gpar(fontsize=6, fontfamily="Arial"), column_names_rot=45, border=TRUE,
    column_title=sprintf("k = %d  (%d modules; %d of 32 types with a private module)",
                         K, nrow(mat), length(unique(pref))),
    column_title_gp=gpar(fontsize=7, fontfamily="Arial", fontface="bold"),
    heatmap_legend_param=list(title_gp=gpar(fontsize=5.5,fontfamily="Arial"),
      labels_gp=gpar(fontsize=5,fontfamily="Arial"), at=c(-1,0,1,2,3),
      legend_height=unit(10,"mm"), grid_width=unit(2,"mm")))
  h <- if (K == 20) 105 else 180
  cairo_pdf(sprintf("%s/figures/ED7a_stability_k%d.pdf",ROOT,K), width=150/25.4, height=h/25.4, family="Arial")
  draw(ht); invisible(dev.off())
  cat(sprintf("wrote ED7a_stability_k%d.pdf  (%d modules)\n", K, nrow(mat)))
}
