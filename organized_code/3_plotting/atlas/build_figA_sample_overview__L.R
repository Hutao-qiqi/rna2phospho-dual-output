#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(ComplexHeatmap); library(circlize); library(grid); library(RColorBrewer)})
ROOT <- "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
K <- 30
sma  <- read.delim(sprintf("%s/results/sample_module_activity_k%d.tsv",ROOT,K), check.names=FALSE)
summ <- read.delim(sprintf("%s/results/module_summary_k%d.tsv",ROOT,K), stringsAsFactors=FALSE)
mods <- paste0("M", 1:K)

M <- t(as.matrix(sma[, mods]))            # K x n_samples
cancer <- sma$cancer
pref <- summ$pref_cancer
cancer_order <- c(unique(pref), setdiff(sort(unique(cancer)), unique(pref)))

# order samples by cancer, then within cancer by its leading module activity
lead <- summ$module[match(cancer_order, summ$pref_cancer)]   # leading module per cancer (NA if none)
ord_key <- sapply(seq_along(cancer), function(j){
  cz <- match(cancer[j], cancer_order)
  lm <- lead[cz]; within <- if (is.na(lm)) 0 else -M[lm, j]
  cz*1e6 + within
})
ord <- order(ord_key)
M <- M[summ$module, ord]; cancer <- factor(cancer[ord], levels=cancer_order)
rownames(M) <- paste0(summ$module, " ", summ$pref_cancer)

cancer_cols <- setNames(colorRampPalette(brewer.pal(12,"Set3"))(length(cancer_order)), cancer_order)
ha <- HeatmapAnnotation(cancer=cancer, col=list(cancer=cancer_cols),
        simple_anno_size=unit(2.5,"mm"), show_legend=FALSE,
        annotation_name_gp=gpar(fontsize=5.5, fontfamily="Arial"))

col_fun <- colorRamp2(c(-2,0,2,4), c("#2171B5","#FFFFFF","#F6C8B6","#CB181D"))
ht <- Heatmap(M, name="module\nactivity (z)", col=col_fun,
  cluster_rows=FALSE, cluster_columns=FALSE,
  top_annotation=ha,
  column_split=cancer, column_gap=unit(0.4,"mm"), column_title_rot=90,
  column_title_gp=gpar(fontsize=5, fontfamily="Arial"),
  show_column_names=FALSE,
  row_names_side="right", row_names_gp=gpar(fontsize=5.8, fontfamily="Arial"),
  use_raster=TRUE, raster_quality=3,
  heatmap_legend_param=list(title_gp=gpar(fontsize=6,fontfamily="Arial"),
    labels_gp=gpar(fontsize=5.5,fontfamily="Arial"), at=c(-2,0,2,4)))
cairo_pdf(sprintf("%s/figures/figA_sample_overview_k%d.pdf",ROOT,K), width=200/25.4, height=110/25.4, family="Arial")
draw(ht, column_title="Pan-cancer phospho-module activity across 10,023 TCGA tumours",
     column_title_gp=gpar(fontsize=8, fontfamily="Arial", fontface="bold"))
invisible(dev.off()); cat(sprintf("wrote figA_sample_overview_k%d.pdf\n", K))
