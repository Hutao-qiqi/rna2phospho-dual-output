#!/usr/bin/env Rscript
# Build panel c v9 (3 external cohorts, FPKM removed).
suppressPackageStartupMessages({ library(ggplot2) })
source("panel_c.R")
g <- make_panel_c()
grDevices::cairo_pdf("panel_c_v10_test.pdf",
                     width = 150 / 25.4, height = 58 / 25.4, family = "Arial")
print(g)
grDevices::dev.off()
cat("wrote panel_c_v9_test.pdf\n")
