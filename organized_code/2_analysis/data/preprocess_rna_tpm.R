#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse); library(SummarizedExperiment); library(data.table)
  library(GenomicFeatures); library(org.Hs.eg.db); library(biomaRt); library(arrow)
})

opt_list <- list(
  make_option("--counts", type="character"),
  make_option("--index", type="character"),
  make_option("--out", type="character"),
  make_option("--low", type="double", default=1),
  make_option("--prop", type="double", default=0.2),
  make_option("--log2", type="logical", default=TRUE)
)
opt <- parse_args(OptionParser(option_list=opt_list))

se <- readRDS(opt$counts)
expr_df <- as.data.frame(assay(se))  # HTSeq counts: genes x samples

# Strip Ensembl version and collapse duplicate gene IDs by summation to avoid duplicate rownames error
dt <- as.data.table(expr_df, keep.rownames = "gene")
dt[, gene := sub("\\..*$", "", gene)]
dt <- dt[, lapply(.SD, sum, na.rm = TRUE), by = gene]
expr <- as.data.frame(dt)
rownames(expr) <- expr$gene
expr <- expr[, setdiff(colnames(expr), "gene"), drop = FALSE]

# Filter: keep genes expressed in >= prop samples with counts >= low
keep <- rowMeans(expr >= opt$low) >= opt$prop
expr <- expr[keep, , drop=FALSE]

# Gene length via biomaRt (approximate); guard against temporary failures
mart <- biomaRt::useEnsembl(biomart="genes", dataset="hsapiens_gene_ensembl", version = NULL)
anno <- biomaRt::getBM(attributes=c("ensembl_gene_id","transcript_length"),
                       filters="ensembl_gene_id",
                       values=rownames(expr), mart=mart)
len <- tapply(anno$transcript_length, anno$ensembl_gene_id, median)
len <- len[rownames(expr)]
len[is.na(len)] <- median(len, na.rm=TRUE)

# TPM: per-gene length normalization then per-sample scaling
rate <- sweep(expr, 1, len/1000, `/`)
per_sample_sum <- colSums(rate, na.rm=TRUE)/1e6
tpm <- sweep(rate, 2, per_sample_sum, `/`)
if (isTRUE(opt$log2)) tpm <- log2(tpm + 1)

# Keep only samples present in master index; if index empty, stop with message
index <- tryCatch(data.table::fread(opt$index), error = function(e) data.table())
if (nrow(index) == 0 || !("tcga_barcode_full" %in% colnames(index))) {
  stop("Master index 为空或缺少列 tcga_barcode_full：", opt$index)
}
keep_samples <- intersect(colnames(tpm), index$tcga_barcode_full)
if (length(keep_samples) == 0) {
  stop("与 master index 无交集的 RNA 样本，请检查匹配与条码格式。")
}
tpm <- tpm[, keep_samples, drop=FALSE]

# Write parquet as a data frame with first column as gene id
dt <- data.frame(gene = rownames(tpm), tpm, check.names = FALSE)
dir.create(dirname(opt$out), showWarnings = FALSE, recursive = TRUE)
arrow::write_parquet(dt, sink = opt$out)
