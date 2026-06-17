suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})

root <- "E:/data/gongke/TCGA-TCPA"
rds_path <- file.path(root, "_downloads", "iccite_seq_tcell_2025", "Perturb_icCITE_seq_FOXP3_regulators.rds")
out_dir <- file.path(root, "_downloads", "iccite_seq_tcell_2025", "background_audit")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

get_counts <- function(obj, assay) {
  tryCatch(
    GetAssayData(obj, assay = assay, layer = "counts"),
    error = function(e) GetAssayData(obj, assay = assay, slot = "counts")
  )
}

obj <- readRDS(rds_path)
assays <- Assays(obj)

feature_rows <- list()
for (assay in assays) {
  mat <- get_counts(obj, assay)
  features <- rownames(mat)
  feature_rows[[assay]] <- data.frame(
    assay = assay,
    feature = features,
    is_control_like = grepl("isotype|igg|control|migg|hmi?gg", features, ignore.case = TRUE),
    stringsAsFactors = FALSE
  )
}
features <- do.call(rbind, feature_rows)
write.table(features, file.path(out_dir, "all_assay_control_like_features.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

intra <- get_counts(obj, "intra")
intra_features <- rownames(intra)
phospho_features <- grep("phospho|Pho|pSTAT|RPS6Pho", intra_features, ignore.case = TRUE, value = TRUE)
control_features <- intra_features[grepl("isotype|igg|control|mIgG|HMIgG", intra_features, ignore.case = TRUE)]

write.table(data.frame(feature = phospho_features), file.path(out_dir, "phospho_features.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(feature = control_features), file.path(out_dir, "intra_control_like_features.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

if (length(control_features) == 0) {
  stop("No intra control-like feature found")
}

control_mat <- as.matrix(t(intra[control_features, , drop = FALSE]))
phospho_mat <- as.matrix(t(intra[phospho_features, , drop = FALSE]))

control_summary <- data.frame(
  cell_id = colnames(intra),
  control_mean = rowMeans(control_mat),
  control_median = apply(control_mat, 1, median),
  control_max = apply(control_mat, 1, max),
  stringsAsFactors = FALSE
)
write.table(control_summary, file.path(out_dir, "cell_control_background_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

qc_rows <- list()
for (feat in phospho_features) {
  x <- phospho_mat[, feat]
  y <- control_summary$control_mean
  corrected <- pmax(x - y, 0)
  qc_rows[[feat]] <- data.frame(
    feature = feat,
    raw_mean = mean(x),
    raw_median = median(x),
    raw_nonzero_rate = mean(x > 0),
    control_mean_mean = mean(y),
    spearman_control_mean = suppressWarnings(cor(x, y, method = "spearman")),
    corrected_mean = mean(corrected),
    corrected_median = median(corrected),
    corrected_nonzero_rate = mean(corrected > 0),
    fraction_removed_by_mean_subtraction = 1 - (sum(corrected) / sum(x)),
    stringsAsFactors = FALSE
  )
}
qc <- do.call(rbind, qc_rows)
write.table(qc, file.path(out_dir, "phospho_background_qc.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

corrected <- phospho_mat
for (j in seq_len(ncol(corrected))) {
  corrected[, j] <- pmax(corrected[, j] - control_summary$control_mean, 0)
}
corrected_sparse <- Matrix(t(corrected), sparse = TRUE)
rownames(corrected_sparse) <- phospho_features
colnames(corrected_sparse) <- colnames(intra)

Matrix::writeMM(corrected_sparse, file.path(out_dir, "phospho_counts_control_mean_subtracted.mtx"))
write.table(phospho_features, file.path(out_dir, "phospho_counts_control_mean_subtracted_features.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(colnames(intra), file.path(out_dir, "phospho_counts_control_mean_subtracted_barcodes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)

cat("control features\n")
print(control_features)
cat("qc\n")
print(qc)
