#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse); library(data.table); library(arrow)
  library(GSVA); library(AnnotationDbi); library(org.Hs.eg.db)
})

opt_list <- list(
  make_option("--X", type="character", help="gene x sample parquet; first column = gene (Ensembl ID)"),
  make_option("--gmt", type="character", default="", help="space-separated GMT files (HGNC symbols)"),
  make_option("--method", type="character", default="gsva"),
  make_option("--minsz", type="integer", default=10),
  make_option("--maxsz", type="integer", default=500),
  make_option("--out_gsva", type="character", help="output parquet: pathway x sample; first column 'pathway'"),
  make_option("--out_symbols", type="character", help="output parquet: gene(symbol) x sample; first column 'gene_symbol'")
)
opt <- parse_args(OptionParser(option_list=opt_list))

# Read gene x sample matrix
Xdt <- as.data.frame(arrow::read_parquet(opt$X))
stopifnot(ncol(Xdt) >= 2)
gene_col <- colnames(Xdt)[1]
rownames(Xdt) <- Xdt[[gene_col]]
Xdt <- Xdt[, setdiff(colnames(Xdt), gene_col), drop = FALSE]

# Map Ensembl -> HGNC symbols (may be one-to-many; collapse by median)
ens <- sub("\\..*$", "", rownames(Xdt))
map <- suppressWarnings(AnnotationDbi::select(org.Hs.eg.db, keys = unique(ens), keytype = "ENSEMBL", columns = c("SYMBOL")))
map <- map[!is.na(map$SYMBOL) & nzchar(map$SYMBOL), ]
rownames(Xdt) <- ens
Xsym <- Xdt[rownames(Xdt) %in% map$ENSEMBL, , drop = FALSE]
sym <- map$SYMBOL[match(rownames(Xsym), map$ENSEMBL)]
Xsym <- rowsum(as.matrix(Xsym), group = sym, reorder = FALSE)
Xsym <- as.data.frame(Xsym)

# Write symbol-level expression parquet
out_sym_df <- data.frame(gene_symbol = rownames(Xsym), Xsym, check.names = FALSE)
dir.create(dirname(opt$out_symbols), showWarnings = FALSE, recursive = TRUE)
arrow::write_parquet(out_sym_df, sink = opt$out_symbols)

# Parse GMT files into list of gene sets (symbols)
parse_gmt <- function(path) {
  if (!nzchar(path) || !file.exists(path)) return(list())
  ln <- readLines(path, warn = FALSE)
  lst <- lapply(ln, function(x) {
    parts <- strsplit(x, "\t", fixed = TRUE)[[1]]
    if (length(parts) < 3) return(NULL)
    gs_name <- parts[1]
    genes <- unique(parts[-c(1,2)])
    genes[nzchar(genes)]
  })
  names(lst) <- vapply(strsplit(ln, "\t", fixed = TRUE), function(v) v[[1]], FUN.VALUE = character(1))
  lst <- lst[!vapply(lst, is.null, logical(1))]
  lst
}

gmt_paths <- strsplit(opt$gmt, " ")[[1]]
gmt_paths <- gmt_paths[nzchar(gmt_paths)]
sets <- list()
for (g in gmt_paths) {
  if (file.exists(g)) {
    gs <- parse_gmt(g)
    if (length(gs) > 0) sets <- append(sets, gs)
  }
}

if (length(sets) > 0) {
  # Filter gene sets by size after intersecting with available symbols
  gssz <- vapply(sets, function(v) length(intersect(v, rownames(Xsym))), integer(1))
  keep_idx <- which(gssz >= opt$minsz & gssz <= opt$maxsz)
  if (length(keep_idx) > 0) sets <- sets[keep_idx] else sets <- list()
}

# Helper to support GSVA >= 2.x (param objects) and legacy API
run_gsva_compat <- function(emat, sets, method, minsz, maxsz) {
  res <- NULL
  # Prefer new param-based API if available
  has_gsvaParam <- "gsvaParam" %in% getNamespaceExports("GSVA")
  has_ssgseaParam <- "ssgseaParam" %in% getNamespaceExports("GSVA")
  if (has_gsvaParam) {
    try({
      if (tolower(method) == "ssgsea" && has_ssgseaParam) {
        param <- do.call(GSVA::ssgseaParam, list(exprData = emat, geneSets = sets,
                                                 minSize = as.integer(minsz), maxSize = as.integer(maxsz)))
        res <- GSVA::gsva(param)
      } else {
        # default to GSVA kernel
        param <- do.call(GSVA::gsvaParam, list(exprData = emat, geneSets = sets, kcdf = "Gaussian",
                                              minSize = as.integer(minsz), maxSize = as.integer(maxsz)))
        res <- GSVA::gsva(param)
      }
    }, silent = TRUE)
  }
  # Legacy fallback (pre‑2.0 signature)
  if (is.null(res)) {
    ok <- FALSE
    try({
      res <- GSVA::gsva(emat, sets, method = method, kcdf = "Gaussian",
                        min.sz = as.integer(minsz), max.sz = as.integer(maxsz),
                        parallel.sz = 1, verbose = FALSE)
      ok <- TRUE
    }, silent = TRUE)
    if (!ok) {
      res <- GSVA::gsva(emat, sets, method = method, kcdf = "Gaussian",
                        minSize = as.integer(minsz), maxSize = as.integer(maxsz),
                        parallel.sz = 1, verbose = FALSE)
    }
  }
  res
}

Sdf <- data.frame(pathway = character(0))
if (length(sets) > 0) {
  emat <- as.matrix(Xsym)  # genes x samples
  sc <- run_gsva_compat(emat, sets, method = opt$method, minsz = opt$minsz, maxsz = opt$maxsz)
  if (is.null(sc)) stop("GSVA 运行失败：请检查 GSVA 版本或基因集/表达矩阵输入")
  Sdf <- data.frame(pathway = rownames(sc), sc, check.names = FALSE)
}

dir.create(dirname(opt$out_gsva), showWarnings = FALSE, recursive = TRUE)
arrow::write_parquet(Sdf, sink = opt$out_gsva)
