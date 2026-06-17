suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

root <- "E:/data/gongke/TCGA-TCPA"
rds_path <- file.path(root, "_downloads", "iccite_seq_tcell_2025", "Perturb_icCITE_seq_FOXP3_regulators.rds")
out_dir <- file.path(root, "_downloads", "iccite_seq_tcell_2025", "export")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

get_counts <- function(obj, assay) {
  DefaultAssay(obj) <- assay
  tryCatch(
    GetAssayData(obj, assay = assay, layer = "counts"),
    error = function(e) GetAssayData(obj, assay = assay, slot = "counts")
  )
}

write_sparse_bundle <- function(mat, dir_path, prefix) {
  dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  Matrix::writeMM(mat, file.path(dir_path, paste0(prefix, ".mtx")))
  write.table(rownames(mat), file.path(dir_path, paste0(prefix, "_features.tsv")), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
  write.table(colnames(mat), file.path(dir_path, paste0(prefix, "_barcodes.tsv")), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
}

obj <- readRDS(rds_path)

metadata <- obj@meta.data
metadata$cell_id <- rownames(metadata)
metadata <- metadata[, c("cell_id", setdiff(colnames(metadata), "cell_id")), drop = FALSE]
write.table(metadata, file.path(out_dir, "cell_metadata.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

intra <- get_counts(obj, "intra")
write_sparse_bundle(intra, file.path(out_dir, "intra_counts"), "intra_counts")

phospho_features <- grep("phospho|Pho|pSTAT|RPS6Pho", rownames(intra), ignore.case = TRUE, value = TRUE)
phospho <- intra[phospho_features, , drop = FALSE]
write_sparse_bundle(phospho, file.path(out_dir, "phospho_counts"), "phospho_counts")

rna <- get_counts(obj, "RNA")
hvgs <- VariableFeatures(obj)
hvgs <- intersect(hvgs, rownames(rna))
rna_hvg <- rna[hvgs, , drop = FALSE]
write_sparse_bundle(rna_hvg, file.path(out_dir, "rna_hvg_counts"), "rna_hvg_counts")

features <- data.frame(
  assay = rep(c("RNA_HVG", "intra", "phospho"), c(length(hvgs), nrow(intra), length(phospho_features))),
  feature = c(hvgs, rownames(intra), phospho_features)
)
write.table(features, file.path(out_dir, "exported_features.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

summary <- data.frame(
  item = c("n_cells", "n_rna_hvg", "n_intra_features", "n_phospho_features"),
  value = c(ncol(rna), length(hvgs), nrow(intra), length(phospho_features))
)
write.table(summary, file.path(out_dir, "export_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

cat("exported to", out_dir, "\n")
print(summary)
print(phospho_features)
