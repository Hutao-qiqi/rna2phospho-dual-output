suppressPackageStartupMessages({library(ComplexHeatmap);library(circlize);library(grid)})
setwd("E:/data/gongke/TCGA-TCPA/SCP682_MAIN/attention_export")
fc <- as.matrix(read.delim("nmf_factor_by_cancer.tsv", row.names=1, check.names=FALSE))
# z-score per factor (row) across cancers
z <- t(scale(t(fc)))
col_fun <- colorRamp2(c(-1.5,0,1.5,3), c("#2171B5","#FFFFFF","#F6C8B6","#CB181D"))
ht <- Heatmap(z, name="cancer\nspecificity\n(z)", col=col_fun,
  clustering_method_rows="ward.D2", clustering_method_columns="ward.D2",
  row_names_gp=gpar(fontsize=6.5,fontfamily="Arial"),
  column_names_gp=gpar(fontsize=6.5,fontfamily="Arial"),
  column_title="NMF phospho-modules are cancer-type specific",
  column_title_gp=gpar(fontsize=8,fontfamily="Arial",fontface="bold"),
  cell_fun=function(j,i,x,y,w,h,fill){
    if(z[i,j]>1.5) grid.text("*", x, y, gp=gpar(fontsize=7,col="black"))},
  heatmap_legend_param=list(title_gp=gpar(fontsize=6,fontfamily="Arial"),
                            labels_gp=gpar(fontsize=5.5,fontfamily="Arial")))
cairo_pdf("nmf_module_by_cancer_heatmap.pdf", width=110/25.4, height=95/25.4, family="Arial")
draw(ht); invisible(dev.off()); cat("wrote nmf_module_by_cancer_heatmap.pdf\n")
