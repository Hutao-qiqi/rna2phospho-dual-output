#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(AnnotationDbi)
  library(org.Hs.eg.db)
})

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) {
    return(NA_character_)
  }
  args[[i + 1]]
}

infile <- get_arg("--in")
outfile <- get_arg("--out")

if (is.na(infile) || is.na(outfile) || infile == "" || outfile == "") {
  stop("Usage: map_ensembl_to_symbol.R --in <ids.txt> --out <map.tsv>")
}

ens <- readLines(infile, warn = FALSE)
ens <- unique(trimws(ens))
ens <- ens[ens != ""]
ens <- sub("\\.[0-9]+$", "", ens)

if (length(ens) == 0) {
  write.table(
    data.frame(ENSEMBL = character(), SYMBOL = character()),
    file = outfile,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    col.names = TRUE
  )
  cat("WROTE", outfile, "rows=0\n")
  quit(status = 0)
}

map <- AnnotationDbi::select(org.Hs.eg.db, keys = ens, keytype = "ENSEMBL", columns = c("SYMBOL"))
map <- map[!is.na(map$SYMBOL) & map$SYMBOL != "", , drop = FALSE]
# Keep first SYMBOL per ENSEMBL to simplify downstream aggregation.
map <- map[!duplicated(map$ENSEMBL), c("ENSEMBL", "SYMBOL"), drop = FALSE]

write.table(map, file = outfile, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE)
cat("WROTE", outfile, "rows=", nrow(map), "\n")
