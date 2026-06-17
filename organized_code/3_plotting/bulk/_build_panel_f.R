#!/usr/bin/env Rscript
# Build panel f v10: ComplexHeatmap (左) + connector (中, 渐变梯形) + bar chart (右).
suppressPackageStartupMessages({
  library(ComplexHeatmap)
  library(circlize)
  library(RColorBrewer)
  library(grid)
  library(ggplot2)
  library(cowplot)
})

source("panel_f.R")

ht <- make_panel_f()

ht_grob <- grid::grid.grabExpr(
  ComplexHeatmap::draw(
    ht,
    heatmap_legend_side    = "left",
    annotation_legend_side = "left",
    merge_legends          = TRUE,
    padding                = grid::unit(c(3, 4, 3, 4), "mm"),
    column_title           = "SCP682 per-site ρ across 12 CPTAC cancer types — top vs bottom 1,000 phosphosites, 6 k-means clusters",
    column_title_gp        = grid::gpar(fontsize = 8, fontfamily = "Arial",
                                        col = "#222222", fontface = "bold"),
    column_title_side      = "top"))

bar       <- make_panel_f_barchart()
connector <- make_panel_f_connector()

# 三层 cowplot 布局：heatmap (左) + bar (右) + connector (overlay 全画布)
combined <- cowplot::ggdraw() +
  cowplot::draw_plot(ht_grob,  x = 0.00, y = 0.00, width = 0.555, height = 1.00) +
  cowplot::draw_plot(bar,      x = 0.690, y = 0.00, width = 0.310, height = 1.00) +
  cowplot::draw_plot(connector, x = 0.00, y = 0.00, width = 1.00, height = 1.00)

grDevices::cairo_pdf("panel_f_test.pdf",
                     width  = 280 / 25.4,
                     height = 200 / 25.4,
                     family = "Arial")
print(combined)
grDevices::dev.off()
cat("wrote panel_f_test.pdf\n")
