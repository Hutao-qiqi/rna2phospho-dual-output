#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse); library(jsonlite); library(TCGAbiolinks); library(data.table)
})

opt_list <- list(
  make_option("--project", type="character", default="TCGA-LUAD"),
  make_option("--workflow", type="character", default="STAR - Counts"),
  make_option("--sample_types", type="character", default="Primary Tumor"),
  make_option("--download_method", type="character", default="api"),
  make_option("--files_per_chunk", type="integer", default=100),
  make_option("--out_counts", type="character"),
  make_option("--out_clinical", type="character"),
  make_option("--out_meta", type="character"),
  make_option("--out_barcodes", type="character")
)
opt <- parse_args(OptionParser(option_list=opt_list))

# Ensure TCGAbiolinks is installed
if (!requireNamespace("TCGAbiolinks", quietly=TRUE)) {
  BiocManager::install("TCGAbiolinks", update=FALSE, ask=FALSE)
}

options(timeout = max(3600, getOption("timeout")))

sample_types <- unlist(strsplit(opt$sample_types, ","))
sample_types <- trimws(sample_types)

query <- GDCquery(project = opt$project,
                  data.category = "Transcriptome Profiling",
                  data.type = "Gene Expression Quantification",
                  workflow.type = opt$workflow,
                  sample.type = sample_types)

# 更稳健的下载：多次重试 + 可配置 chunk/method
ok <- FALSE
for (i in 1:5) {
  try({
    GDCdownload(query, files.per.chunk = opt$files_per_chunk, method = opt$download_method)
    ok <- TRUE; break
  }, silent = TRUE)
  message(sprintf("GDCdownload 重试 %d/5 ...", i))
  Sys.sleep(5 * i)
}
if (!ok) stop("GDCdownload 多次重试仍失败，请检查网络或稍后重试")
se <- GDCprepare(query)         # SummarizedExperiment

# Clinical retrieval with robust fallback (disable marker paper augmentation when available)
clin <- tryCatch({
  fun <- GDCquery_clinic
  fm <- names(formals(fun))
  if ("add.paper" %in% fm) {
    fun(project = opt$project, type = "clinical", save.csv = FALSE, add.paper = FALSE)
  } else if ("save.csv" %in% fm) {
    fun(project = opt$project, type = "clinical", save.csv = FALSE)
  } else {
    fun(project = opt$project, type = "clinical")
  }
}, error = function(e) {
  message("GDCquery_clinic 失败，使用 SE 的 colData 最小临床信息：", conditionMessage(e))
  cd <- as.data.frame(SummarizedExperiment::colData(se))
  cols <- intersect(c("submitter_id","patient","sample_type","short_letter_code"), colnames(cd))
  unique(cd[, cols, drop = FALSE])
})

# GDC Data Release metadata
status_json <- tryCatch({
  jsonlite::fromJSON("https://api.gdc.cancer.gov/status")
}, error=function(e) list())
meta <- list(
  gdc_data_release = if (is.list(status_json) && length(status_json)>0) status_json$data_release else NA,
  date = as.character(Sys.time()),
  project = opt$project,
  workflow = opt$workflow
)

dir.create(dirname(opt$out_counts), showWarnings = FALSE, recursive = TRUE)
saveRDS(se, opt$out_counts)
dir.create(dirname(opt$out_clinical), showWarnings = FALSE, recursive = TRUE)
saveRDS(clin, opt$out_clinical)
dir.create(dirname(opt$out_meta), showWarnings = FALSE, recursive = TRUE)
write(jsonlite::toJSON(meta, auto_unbox=TRUE, pretty=TRUE), file=opt$out_meta)

# Export barcodes + sample types for downstream matching (robust for different metadata fields)
cd <- as.data.frame(SummarizedExperiment::colData(se))
barcodes <- colnames(se)
patient_id <- substr(barcodes, 1, 12)
sample_type <- if ("sample_type" %in% colnames(cd)) cd$sample_type else NA_character_
short_code <- if ("short_letter_code" %in% colnames(cd)) cd$short_letter_code else NA_character_

# If sample_type is missing, infer from short_code or numeric code in barcode
infer_type <- function(bc) {
  code <- substr(bc, 14, 15) # 01 = Primary Tumor, 11 = Solid Tissue Normal
  if (code == "01") return("Primary Tumor")
  if (code == "11") return("Solid Tissue Normal")
  return(NA_character_)
}
if (all(is.na(sample_type))) {
  if (!all(is.na(short_code))) {
    sample_type <- ifelse(short_code == "TP", "Primary Tumor",
                          ifelse(short_code == "NT", "Solid Tissue Normal", NA_character_))
  } else {
    sample_type <- vapply(barcodes, infer_type, FUN.VALUE = character(1))
  }
}

bc_df <- data.frame(
  barcode = barcodes,
  patient_id = patient_id,
  sample_type = sample_type,
  stringsAsFactors = FALSE
)
dir.create(dirname(opt$out_barcodes), showWarnings = FALSE, recursive = TRUE)
data.table::fwrite(bc_df, opt$out_barcodes, sep='\t')
