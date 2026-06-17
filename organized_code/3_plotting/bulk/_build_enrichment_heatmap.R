#!/usr/bin/env Rscript
# Build enrichment heatmap PDF.
suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

source("panel_enrichment_heatmap.R")

g <- make_panel_enrichment_heatmap()
grDevices::cairo_pdf("panel_enrichment_heatmap_test.pdf",
                     width = 110 / 25.4,
                     height = 90 / 25.4,
                     family = "Arial")
print(g)
grDevices::dev.off()
cat("wrote panel_enrichment_heatmap_test.pdf\n")
