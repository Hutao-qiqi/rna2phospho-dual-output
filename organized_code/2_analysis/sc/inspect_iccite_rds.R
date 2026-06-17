suppressPackageStartupMessages({
  library(Seurat)
})

root <- "E:/data/gongke/TCGA-TCPA"
rds_path <- file.path(root, "_downloads", "iccite_seq_tcell_2025", "Perturb_icCITE_seq_FOXP3_regulators.rds")
out_dir <- file.path(root, "_downloads", "iccite_seq_tcell_2025", "inspect")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

obj <- readRDS(rds_path)

sink(file.path(out_dir, "object_summary.txt"))
print(class(obj))
print(obj)
cat("\nassays\n")
print(Assays(obj))
cat("\nreductions\n")
print(Reductions(obj))
cat("\nmeta columns\n")
print(colnames(obj@meta.data))
cat("\nmeta head\n")
print(utils::head(obj@meta.data[, seq_len(min(30, ncol(obj@meta.data))), drop = FALSE]))
for (assay in Assays(obj)) {
  cat("\nassay:", assay, "\n")
  a <- obj[[assay]]
  print(class(a))
  print(dim(a))
  print(utils::head(rownames(a), 40))
}
sink()

writeLines(Assays(obj), file.path(out_dir, "assays.txt"))
write.table(colnames(obj@meta.data), file.path(out_dir, "metadata_columns.tsv"), sep = "\t", quote = FALSE, row.names = FALSE, col.names = "metadata_column")

for (assay in Assays(obj)) {
  features <- rownames(obj[[assay]])
  write.table(data.frame(assay = assay, feature = features), file.path(out_dir, paste0("features_", assay, ".tsv")), sep = "\t", quote = FALSE, row.names = FALSE)
}

cat("inspection written to", out_dir, "\n")
