suppressPackageStartupMessages({
  library(SeuratObject)
  library(Matrix)
})

root <- "/data/lsy/phospho_project"
rds_path <- file.path(root, "raw", "Perturb_icCITE_seq_FOXP3_regulators.rds")
out_dir <- file.path(root, "export", "iccite_background_corrected")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

get_counts <- function(obj, assay) {
  DefaultAssay(obj) <- assay
  GetAssayData(obj, assay = assay, layer = "counts")
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
    is_control_like = grepl("isotype|igg|control|migg|hmigg", features, ignore.case = TRUE),
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

control <- as.matrix(intra[control_features, , drop = FALSE])
control_mean <- Matrix::colMeans(control)
control_median <- apply(control, 2, median)
control_max <- apply(control, 2, max)

control_summary <- data.frame(
  cell_id = colnames(intra),
  control_mean = as.numeric(control_mean),
  control_median = as.numeric(control_median),
  control_max = as.numeric(control_max),
  stringsAsFactors = FALSE
)
write.table(control_summary, file.path(out_dir, "cell_control_background_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

qc_rows <- list()
corrected <- intra[phospho_features, , drop = FALSE]
for (i in seq_along(phospho_features)) {
  feat <- phospho_features[[i]]
  x <- as.numeric(intra[feat, ])
  y <- as.numeric(control_mean)
  z <- pmax(x - y, 0)
  corrected[i, ] <- z
  qc_rows[[feat]] <- data.frame(
    feature = feat,
    raw_mean = mean(x),
    raw_median = median(x),
    raw_nonzero_rate = mean(x > 0),
    control_mean_mean = mean(y),
    spearman_control_mean = suppressWarnings(cor(x, y, method = "spearman")),
    corrected_mean = mean(z),
    corrected_median = median(z),
    corrected_nonzero_rate = mean(z > 0),
    fraction_removed_by_mean_subtraction = 1 - (sum(z) / sum(x)),
    stringsAsFactors = FALSE
  )
}
qc <- do.call(rbind, qc_rows)
write.table(qc, file.path(out_dir, "phospho_background_qc.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

Matrix::writeMM(corrected, file.path(out_dir, "phospho_counts_control_mean_subtracted.mtx"))
write.table(phospho_features, file.path(out_dir, "phospho_counts_control_mean_subtracted_features.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
write.table(colnames(intra), file.path(out_dir, "phospho_counts_control_mean_subtracted_barcodes.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)

summary <- data.frame(
  item = c("n_cells", "n_phospho_features", "n_control_features"),
  value = c(ncol(intra), length(phospho_features), length(control_features))
)
write.table(summary, file.path(out_dir, "background_correction_summary.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

cat("done\n")
print(summary)
print(control_features)
print(qc)
