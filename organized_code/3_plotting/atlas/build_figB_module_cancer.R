#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(ComplexHeatmap); library(circlize); library(grid)})
ROOT <- "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
K <- 30
mbc  <- as.matrix(read.delim(sprintf("%s/results/module_by_cancer_median_k%d.tsv",ROOT,K), row.names=1, check.names=FALSE))
summ <- read.delim(sprintf("%s/results/module_summary_k%d.tsv",ROOT,K), stringsAsFactors=FALSE)

abbr <- c(OXIDATIVE_PHOSPHORYLATION="OXPHOS", ESTROGEN_RESPONSE_EARLY="Estrogen",
  ESTROGEN_RESPONSE_LATE="Estrogen", ANDROGEN_RESPONSE="Androgen",
  INTERFERON_GAMMA_RESPONSE="IFN-g", INTERFERON_ALPHA_RESPONSE="IFN-a",
  EPITHELIAL_MESENCHYMAL_TRANSITION="EMT", G2M_CHECKPOINT="G2M", E2F_TARGETS="E2F",
  MYC_TARGETS_V1="MYC", MYC_TARGETS_V2="MYC", XENOBIOTIC_METABOLISM="Xenobiotic",
  HYPOXIA="Hypoxia", P53_PATHWAY="p53", KRAS_SIGNALING_DN="KRAS-dn",
  KRAS_SIGNALING_UP="KRAS-up", ALLOGRAFT_REJECTION="Immune", INFLAMMATORY_RESPONSE="Inflammation",
  APICAL_JUNCTION="ApicalJunction", FATTY_ACID_METABOLISM="FattyAcid", MYOGENESIS="Myogenesis",
  MITOTIC_SPINDLE="MitSpindle", PROTEIN_SECRETION="Secretion", COAGULATION="Coagulation",
  HEDGEHOG_SIGNALING="Hedgehog", UV_RESPONSE_DN="UV-dn", APOPTOSIS="Apoptosis",
  ANGIOGENESIS="Angiogenesis", NOTCH_SIGNALING="Notch", TGF_BETA_SIGNALING="TGFb",
  MTORC1_SIGNALING="mTORC1", IL2_STAT5_SIGNALING="IL2-STAT5", DNA_REPAIR="DNArepair",
  SPERMATOGENESIS="Spermatogen", PANCREAS_BETA_CELLS="PancBeta", REACTIVE_OXYGEN_SPECIES_PATHWAY="ROS",
  CHOLESTEROL_HOMEOSTASIS="Cholesterol", ADIPOGENESIS="Adipogenesis", COMPLEMENT="Complement")

# row order: summ already sorted by pref_cancer, pref_z desc
mat <- mbc[summ$module, , drop=FALSE]
# column order: cancers in module-preference order -> diagonal; shared cancers to the right
pref <- summ$pref_cancer
cancer_order <- c(unique(pref), setdiff(colnames(mat), unique(pref)))
mat <- mat[, cancer_order, drop=FALSE]

# row labels: M# CANCER (+ hallmark abbr if FDR<0.05)
hl <- ifelse(summ$top_fdr < 0.05, paste0("  ", ifelse(summ$top_hallmark %in% names(abbr), abbr[summ$top_hallmark], "")), "")
rownames(mat) <- paste0(summ$module, "  ", summ$pref_cancer, hl)

col_fun <- colorRamp2(c(-1.5,-0.6,0,1.5,3), c("#08519C","#6BAED6","#FFFFFF","#F6C8B6","#CB181D"))
ht <- Heatmap(mat, name="module\nactivity (z)", col=col_fun,
  cluster_rows=FALSE, cluster_columns=FALSE,
  row_names_side="right", row_names_gp=gpar(fontsize=6.2, fontfamily="Arial"),
  column_names_gp=gpar(fontsize=6.5, fontfamily="Arial"), column_names_rot=45,
  border=TRUE,
  column_title="Pan-cancer phospho-modules across 32 TCGA solid tumours (k = 30)",
  column_title_gp=gpar(fontsize=7.5, fontfamily="Arial", fontface="bold"),
  heatmap_legend_param=list(title_gp=gpar(fontsize=6,fontfamily="Arial"),
    labels_gp=gpar(fontsize=5.5,fontfamily="Arial"), at=c(-1,0,1,2,3)))
cairo_pdf(sprintf("%s/figures/figB_module_cancer_k%d.pdf",ROOT,K), width=165/25.4, height=140/25.4, family="Arial")
draw(ht); invisible(dev.off())
cat(sprintf("wrote figB_module_cancer_k%d.pdf\n", K))
