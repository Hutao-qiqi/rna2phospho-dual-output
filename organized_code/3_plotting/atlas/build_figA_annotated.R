#!/usr/bin/env Rscript
# figA upgraded: sample x module heatmap with multi-row clinical annotation bars
# (style after the TIME-subtype reference figure); columns grouped by cancer.
suppressPackageStartupMessages({library(ComplexHeatmap); library(circlize); library(grid); library(RColorBrewer)})
ROOT <- "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
K <- 30
sma  <- read.delim(sprintf("%s/results/sample_module_activity_k%d.tsv",ROOT,K), check.names=FALSE)
summ <- read.delim(sprintf("%s/results/module_summary_k%d.tsv",ROOT,K), stringsAsFactors=FALSE)
clin <- read.delim("E:/data/gongke/TCGA-TCPA/02_results/model_prediction/20260529_tcga_full_scp682_main_reprediction_v1/clinical_covariates/tcga_scp682_sample_clinical_covariates_all33.tsv", stringsAsFactors=FALSE)
mods <- paste0("M", 1:K)

M <- t(as.matrix(sma[, mods]))            # K x n
cancer <- sma$cancer
pref <- summ$pref_cancer
cancer_order <- c(unique(pref), setdiff(sort(unique(cancer)), unique(pref)))
lead <- summ$module[match(cancer_order, summ$pref_cancer)]
ord_key <- sapply(seq_along(cancer), function(j){
  cz <- match(cancer[j], cancer_order); lm <- lead[cz]
  cz*1e6 + (if (is.na(lm)) 0 else -M[lm, j])
})
ord <- order(ord_key)
M <- M[summ$module, ord]; cancer <- factor(cancer[ord], levels=cancer_order)
sids <- sma$sample_id[ord]
rownames(M) <- paste0(summ$module, " ", summ$pref_cancer)

cl <- clin[match(sids, clin$sample_id), ]   # align clinical to column order
cancer_cols <- setNames(colorRampPalette(brewer.pal(12,"Set3"))(length(cancer_order)), cancer_order)

ha <- HeatmapAnnotation(
  Age        = cl$age,
  Sex        = cl$sex,
  Purity     = cl$tumor_purity,
  Grade      = cl$grade_num,
  Stage      = cl$stage_num,
  `OS event` = cl$os_event,
  Cancer     = cancer,
  col = list(
    Age        = colorRamp2(c(30,80), c("#FFF7BC","#CC4C02")),
    Sex        = c(Female="#F6C8B6", Male="#6BAED6"),
    Purity     = colorRamp2(c(0.1,1), c("#F7F7F7","#54278F")),
    Grade      = colorRamp2(c(1,4), c("#FEE0D2","#A50F15")),
    Stage      = colorRamp2(c(1,4), c("#DEEBF7","#08519C")),
    `OS event` = colorRamp2(c(0,1), c("#E5E5E5","#252525")),
    Cancer     = cancer_cols),
  na_col = "grey92", simple_anno_size = unit(2.2,"mm"), gap = unit(0.3,"mm"),
  annotation_name_gp = gpar(fontsize=5.5, fontfamily="Arial"),
  annotation_legend_param = list(
    Age    = list(title="Age",    legend_height=unit(8,"mm"), grid_width=unit(1.8,"mm")),
    Sex    = list(title="Sex"),
    Purity = list(title="Purity", legend_height=unit(8,"mm"), grid_width=unit(1.8,"mm")),
    Grade  = list(title="Grade",  legend_height=unit(8,"mm"), grid_width=unit(1.8,"mm")),
    Stage  = list(title="Stage",  legend_height=unit(8,"mm"), grid_width=unit(1.8,"mm")),
    `OS event` = list(title="OS event", at=c(0,1), labels=c("alive","dead"))),
  show_legend = c(TRUE,TRUE,TRUE,TRUE,TRUE,TRUE,FALSE))

ht_opt(legend_title_gp  = gpar(fontsize=5.5, fontfamily="Arial", fontface="bold"),
       legend_labels_gp = gpar(fontsize=4.8, fontfamily="Arial"),
       legend_grid_height = unit(2.0,"mm"), legend_grid_width = unit(2.0,"mm"))
col_fun <- colorRamp2(c(-1.5,-0.5,0,1.5,3), c("#053061","#4393C3","#F7F7F7","#D6604D","#B2182B"))
ht <- Heatmap(M, name="module\nactivity (z)", col=col_fun,
  cluster_rows=FALSE, cluster_columns=FALSE, top_annotation=ha,
  column_split=cancer, column_gap=unit(0.4,"mm"), column_title_rot=90,
  column_title_gp=gpar(fontsize=5, fontfamily="Arial"), show_column_names=FALSE,
  row_names_side="right", row_names_gp=gpar(fontsize=5.8, fontfamily="Arial"),
  use_raster=TRUE, raster_quality=3,
  heatmap_legend_param=list(title_gp=gpar(fontsize=5.5,fontfamily="Arial"),
    labels_gp=gpar(fontsize=4.8,fontfamily="Arial"), at=c(-1,0,1,2,3),
    legend_height=unit(10,"mm"), grid_width=unit(1.8,"mm")))
cairo_pdf(sprintf("%s/figures/figA_annotated_k%d.pdf",ROOT,K), width=210/25.4, height=140/25.4, family="Arial")
draw(ht, column_title="Pan-cancer phospho-module activity with clinical covariates (10,023 TCGA tumours)",
     column_title_gp=gpar(fontsize=8, fontfamily="Arial", fontface="bold"),
     annotation_legend_side="right", heatmap_legend_side="right", merge_legends=TRUE)
invisible(dev.off()); cat(sprintf("wrote figA_annotated_k%d.pdf\n", K))
