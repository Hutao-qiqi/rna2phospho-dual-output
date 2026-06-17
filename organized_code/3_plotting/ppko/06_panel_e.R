# Panel e: per-drug performance — clean rectangular pheatmap (sci-plot canonical)
# 16 drugs (rows, ranked by responsive top20 direction) x 4 metrics (cols)
# drug-class row annotation; diverging blue-lilac-peach ramp; values in cells.

panel_e_gtable <- function(d) {
  suppressPackageStartupMessages({ library(pheatmap); library(grid) })

  df <- d$panel_e_drug %>%
    mutate(drug_class = recode(drug_class, !!!DRUG_CLASS_EN)) %>%
    arrange(desc(responsive_top20_direction_accuracy))

  # unique, clean drug names (title-case first, then de-duplicate)
  cn <- make.unique(stringr::str_to_title(df$perturbation))

  # transposed: rows = 4 metrics, cols = 16 drugs (landscape)
  mat <- t(as.matrix(df[, c("all_sites_cosine", "responsive_top20_cosine",
                            "all_sites_direction_accuracy",
                            "responsive_top20_direction_accuracy")]))
  colnames(mat) <- cn
  rownames(mat) <- c("Cos all", "Cos top20", "Dir all", "Dir top20")

  num <- matrix(sprintf("%.2f", mat), nrow = nrow(mat),
                dimnames = dimnames(mat))

  ann_col <- data.frame(Class = df$drug_class)
  rownames(ann_col) <- cn
  present <- intersect(DRUG_CLASS_ORDER_EN, unique(df$drug_class))
  ann_col$Class <- factor(ann_col$Class, levels = present)
  ann_colors <- list(Class = DRUG_CLASS_COLORS[present])

  ph <- pheatmap(
    mat,
    cluster_rows = FALSE, cluster_cols = FALSE,
    display_numbers = num, number_color = "grey15", fontsize_number = 5.5,
    annotation_col = ann_col, annotation_colors = ann_colors,
    color = RDBU_RAMP,
    breaks = seq(0.3, 1.0, length.out = 101),
    fontsize = 7, fontsize_row = 7, fontsize_col = 6.5,
    border_color = "white", angle_col = 45,
    main = "Per-drug performance",
    legend = TRUE, silent = TRUE,
    fontfamily = BASE_FONT
  )
  ph$gtable
}

draw_panel_e <- function(d, file_pdf, file_png,
                         width_in = 7.8, height_in = 2.8) {
  suppressPackageStartupMessages({ library(grid) })
  gt <- panel_e_gtable(d)
  cairo_pdf(file_pdf, width = width_in, height = height_in, family = BASE_FONT)
  grid::grid.newpage(); grid::grid.draw(gt); dev.off()
  png(file_png, width = width_in * 300, height = height_in * 300, res = 300)
  grid::grid.newpage(); grid::grid.draw(gt); dev.off()
}
