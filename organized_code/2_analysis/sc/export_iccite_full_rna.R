suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

root <- "E:/data/gongke/TCGA-TCPA"
rds_path <- file.path(root, "_downloads", "iccite_seq_tcell_2025", "Perturb_icCITE_seq_FOXP3_regulators.rds")
out_dir <- file.path(root, "_downloads", "iccite_seq_tcell_2025", "export", "rna_full_counts")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(rds_path)

rna <- tryCatch(
  GetAssayData(obj, assay = "RNA", layer = "counts"),
  error = function(e) GetAssayData(obj, assay = "RNA", slot = "counts")
)

Matrix::writeMM(rna, file.path(out_dir, "rna_full_counts.mtx"))
write.table(rownames(rna), file.path(out_dir, "rna_full_counts_features.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(colnames(rna), file.path(out_dir, "rna_full_counts_barcodes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)

summary <- data.frame(
  item = c("n_genes", "n_cells", "nnzero"),
  value = c(nrow(rna), ncol(rna), length(rna@x))
)
write.table(summary, file.path(out_dir, "rna_full_counts_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
print(summary)
