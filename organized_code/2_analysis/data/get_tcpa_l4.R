#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(optparse); library(rvest); library(httr); library(readr); library(jsonlite); library(xml2); library(readxl); library(tools); library(dplyr) })

opt_list <- list(
  make_option("--disease", type="character", default="LUAD"),
  make_option("--out", type="character"),
  make_option("--out_meta", type="character"),
  make_option("--local", type="character", default="")
)
opt <- parse_args(OptionParser(option_list=opt_list))

pages <- c(
  "https://tcpaportal.org/tcpa/download.html",
  "https://tcpa.drbioright.org/rppa500/download.html",
  "https://tcpaportal.org/misc/download.html"
)

read_any <- function(path) {
  ext <- tolower(file_ext(path))
  if (ext %in% c('xls','xlsx')) return(readxl::read_excel(path))
  if (ext %in% c('tsv','txt')) return(readr::read_tsv(path, show_col_types = FALSE))
  if (ext == 'csv') return(readr::read_csv(path, show_col_types = FALSE))
  if (ext == 'zip') {
    uz <- unzip(path, exdir = file.path(tmp_dir, 'unz'))
    # pick first tabular file (use character class for dot to avoid escape issues)
    cand <- uz[grepl('[.](xlsx?|tsv|txt|csv)$', tolower(uz))]
    if (length(cand) == 0) stop('ZIP 内未找到表格文件')
    return(read_any(cand[1]))
  }
  stop(paste('不支持的文件格式:', ext))
}

dir.create(dirname(opt$out), showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(opt$out_meta), showWarnings = FALSE, recursive = TRUE)

meta <- list(source_page = pages, final_url = NA, date = as.character(Sys.time()), note="TCPA RPPA Level-4")

tmp_dir <- tempdir()
df <- NULL

# If local path provided and exists, use it
if (nzchar(opt$local) && file.exists(opt$local)) {
  df <- read_any(opt$local)
  meta$final_url <- paste0("local:", opt$local)
} else {
  # Try to discover a candidate URL and download
  candidate <- NULL
  for (pg in pages) {
    try({
      html <- read_html(pg)
      links <- html %>% html_elements("a") %>% html_attr("href")
      cand <- links[grepl("L(4|evel[ _-]?4)", links, ignore.case = TRUE) & grepl(opt$disease, links, ignore.case = TRUE)]
      if (length(cand) > 0) { candidate <- xml2::url_absolute(cand[1], pg); break }
    }, silent=TRUE)
  }
  if (is.null(candidate)) {
    stop("未能自动定位 TCPA Level-4 下载链接；请手动下载 LUAD L4，并将文件路径填入 config.rppa.local_path 或放到 ", opt$out)
  }
  fname <- basename(candidate)
  tmp_path <- file.path(tmp_dir, fname)
  GET(candidate, write_disk(tmp_path, overwrite=TRUE))
  meta$final_url <- candidate
  df <- read_any(tmp_path)
}

meta_cols <- c('TCGA_ID','Tumor','Set','Sample_Source','Sample_description','UUID')
coln <- colnames(df)

if (any(coln %in% meta_cols)) {
  # Orientation: samples are rows, proteins are columns
  # Choose sample identifier: prefer Sample_description (contains full barcode), else TCGA_ID
  sid_col <- if ('Sample_description' %in% coln) 'Sample_description' else if ('TCGA_ID' %in% coln) 'TCGA_ID' else coln[1]
  prot_cols <- setdiff(coln, meta_cols)
  stopifnot(length(prot_cols) > 0)
  # Build protein x sample table
  samp_ids <- as.character(df[[sid_col]])
  mat <- t(as.matrix(df[, prot_cols, drop = FALSE]))
  colnames(mat) <- samp_ids
  out_df <- data.frame(protein = rownames(mat), mat, check.names = FALSE)
} else {
  # Orientation: proteins are rows, samples are columns
  tcga_cols <- grep('^TCGA-', coln, ignore.case = TRUE, value = TRUE)
  if (length(tcga_cols) == 0) {
    # sometimes columns may have lowercase or spaces; try loosen rule
    tcga_cols <- coln[grepl('TCGA', coln, ignore.case = TRUE)]
  }
  protein_col <- coln[1]
  out_df <- df[, c(protein_col, tcga_cols), drop = FALSE]
  colnames(out_df)[1] <- 'protein'
  out_df$protein <- as.character(out_df$protein)
}

readr::write_tsv(out_df, opt$out)
write(jsonlite::toJSON(meta, auto_unbox=TRUE, pretty=TRUE), file=opt$out_meta)
