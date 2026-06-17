script_dir <- "E:/data/gongke/TCGA-TCPA/02_results/figure_outputs/fig4_v3/scripts"
source(file.path(script_dir, "01_config.R"))
suppressPackageStartupMessages({ library(png); library(grid) })
img <- png::readPNG(file.path(FINAL_DIR, "Fig4_v1.0.png"))
h <- dim(img)[1]; w <- dim(img)[2]
tw <- 820; th <- as.integer(h * (tw / w))
png(file.path(FINAL_DIR, "Fig4_v1.0_thumb.png"), width = tw, height = th)
par(mar = c(0,0,0,0)); plot.new(); rasterImage(img, 0, 0, 1, 1); dev.off()
cat("thumb", tw, "x", th, "\n")
