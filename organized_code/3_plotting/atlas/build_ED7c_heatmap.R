#!/usr/bin/env Rscript
# ED Fig 7c — 33-project (incl. LAML) module x cancer heatmap
suppressPackageStartupMessages({library(ComplexHeatmap); library(circlize); library(grid)})
ROOT <- "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
mbc  <- as.matrix(read.delim(paste0(ROOT,"/results/module_by_cancer_median_k30_33proj.tsv"), row.names=1, check.names=FALSE))
summ <- read.delim(paste0(ROOT,"/results/module_summary_k30_33proj.tsv"), stringsAsFactors=FALSE)
mat <- mbc[summ$module, , drop=FALSE]
pref <- summ$pref_cancer
cancer_order <- c(unique(pref), setdiff(colnames(mat), unique(pref)))
mat <- mat[, cancer_order, drop=FALSE]
rownames(mat) <- paste0(summ$module, "  ", summ$pref_cancer)
laml_modules <- summ$module[summ$pref_cancer == "LAML"]

col_fun <- colorRamp2(c(-1.5,-0.5,0,1.5,3), c("#053061","#4393C3","#F7F7F7","#D6604D","#B2182B"))
ht <- Heatmap(mat, name="activity (z)", col=col_fun, cluster_rows=FALSE, cluster_columns=FALSE,
  row_names_side="right", row_names_gp=gpar(fontsize=5, fontfamily="Arial"),
  column_names_gp=gpar(fontsize=6, fontfamily="Arial"), column_names_rot=45, border=TRUE,
  column_title=sprintf("33-project NMF including LAML (k = 30; %d modules) — LAML-preferred: %s",
                       nrow(mat), if (length(laml_modules)) paste(laml_modules, collapse=", ") else "none"),
  column_title_gp=gpar(fontsize=7, fontfamily="Arial", fontface="bold"),
  heatmap_legend_param=list(title_gp=gpar(fontsize=5.5,fontfamily="Arial"),
    labels_gp=gpar(fontsize=5,fontfamily="Arial"), at=c(-1,0,1,2,3),
    legend_height=unit(10,"mm"), grid_width=unit(2,"mm")))
cairo_pdf(paste0(ROOT,"/figures/ED7c_33project_k30.pdf"), width=152/25.4, height=145/25.4, family="Arial")
draw(ht); invisible(dev.off())
cat(sprintf("wrote ED7c_33project_k30.pdf  (LAML modules: %s)\n",
            if (length(laml_modules)) paste(laml_modules, collapse=", ") else "none"))
