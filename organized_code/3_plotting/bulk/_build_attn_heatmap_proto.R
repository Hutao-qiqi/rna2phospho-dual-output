#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(ComplexHeatmap); library(circlize); library(RColorBrewer); library(grid)
})
setwd("E:/data/gongke/TCGA-TCPA/SCP682_MAIN/attention_export")
M <- as.matrix(read.delim("attn_submatrix.tsv", row.names = 1, check.names = FALSE))
anno <- read.delim("attn_submatrix_anno.tsv", stringsAsFactors = FALSE)
M[is.na(M)] <- 0

func_cols <- c("DNA/cell-cycle"="#CB181D","Cytoskeleton/adhesion"="#2171B5",
               "RNA splicing"="#41AB5D","Signalling"="#F6C8B6")
ha <- rowAnnotation(`function` = anno$func, col = list(`function` = func_cols),
                    simple_anno_size = unit(3,"mm"),
                    annotation_name_gp = gpar(fontsize=6, fontfamily="Arial"))
col_fun <- colorRamp2(c(0, 0.03, 0.13), c("#FFFFFF", "#FDD0A2", "#CB181D"))

ht <- Heatmap(M, name="attention", col=col_fun,
  clustering_method_rows="ward.D2", clustering_method_columns="ward.D2",
  show_row_names=TRUE, show_column_names=FALSE,
  row_names_gp=gpar(fontsize=4.2, fontfamily="Arial"),
  left_annotation=ha, border=TRUE,
  row_split=4, column_split=4,
  row_title=NULL, column_title="Learned site-site attention among 18 hub proteins (76 sites)",
  column_title_gp=gpar(fontsize=8, fontfamily="Arial", fontface="bold"),
  heatmap_legend_param=list(title="attention", title_gp=gpar(fontsize=6,fontfamily="Arial"),
                            labels_gp=gpar(fontsize=5.5,fontfamily="Arial")))
cairo_pdf("attn_heatmap_proto.pdf", width=150/25.4, height=140/25.4, family="Arial")
draw(ht, heatmap_legend_side="right", annotation_legend_side="right")
invisible(dev.off())
cat("wrote attn_heatmap_proto.pdf\n")
