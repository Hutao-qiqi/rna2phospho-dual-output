source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_common.R")

make_panel_h_signed_ptmsea <- function() {
  mat <- read_matrix_tsv(file.path(.FIG5_DATA, "fig5d_ptmsea_signature_heatmap_matrix.tsv"))
  meta <- read.delim(file.path(.FIG5_DATA, "fig5d_ptmsea_signature_heatmap_rows.tsv"),
                     sep = "\t", stringsAsFactors = FALSE)
  meta <- meta[match(rownames(mat), meta$signature), ]
  meta$signature_label <- ifelse(is.na(meta$signature_label), rownames(mat), meta$signature_label)
  meta$signature_label <- substr(meta$signature_label, 1, 52)
  row_levels <- rev(rownames(mat))
  meta$signature <- factor(meta$signature, levels = row_levels)

  long <- matrix_to_long(mat, "signature", "cancer", "nes")
  long$signature <- factor(long$signature, levels = row_levels)
  long$cancer <- factor(long$cancer, levels = colnames(mat))

  pal <- fig2_heat_palette(101)
  class_cols <- c(
    "Kinase" = FIG2_CLUSTER_COLORS[1],
    "Perturbation" = FIG2_CLUSTER_COLORS[2],
    "Pathway" = FIG2_CLUSTER_COLORS[3],
    "Disease" = FIG2_CLUSTER_COLORS[4],
    "Other" = FIG2_CLUSTER_COLORS[8]
  )

  label_df <- meta
  label_df$signature <- factor(label_df$signature, levels = row_levels)
  label_plot <- ggplot(label_df, aes(1, signature)) +
    geom_text(aes(label = signature_label), hjust = 1, family = "Arial", size = 1.65, color = COL_TEXT) +
    xlim(0, 1.03) +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(0, 2, 0, 0, "pt"))

  strip <- ggplot(label_df, aes(1, signature, fill = signature_class)) +
    geom_tile(width = 0.75, height = 0.95) +
    scale_fill_manual(values = class_cols, guide = "none") +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(0, 0, 0, 0, "pt"))

  heat <- ggplot(long, aes(cancer, signature, fill = nes)) +
    geom_tile(color = "white", linewidth = 0.18) +
    scale_fill_gradientn(colors = pal, limits = c(-6, 6), oob = scales::squish, name = "signed NES") +
    labs(title = "Signed PTM-SEA of clinical phosphosite ranks",
         subtitle = "High NES: PTMsigDB up-sites align with risk and down-sites align with protection",
         x = NULL, y = NULL) +
    theme_fig5(7) +
    theme(axis.text.x = element_text(angle = 55, hjust = 1, vjust = 1, size = 6.3, face = "bold"),
          axis.text.y = element_blank(),
          axis.ticks = element_blank(),
          axis.line = element_blank(),
          panel.grid = element_blank(),
          legend.position = "bottom",
          legend.key.size = unit(3.0, "mm"),
          plot.margin = margin(3, 4, 2, 4, "pt"))

  class_summary <- as.data.frame(table(meta$signature_class), stringsAsFactors = FALSE)
  names(class_summary) <- c("signature_class", "n")
  class_summary$signature_class <- factor(class_summary$signature_class,
                                          levels = c("Kinase", "Perturbation", "Pathway", "Disease", "Other"))
  right <- ggplot(class_summary, aes(n, signature_class, fill = signature_class)) +
    geom_col(width = 0.58) +
    geom_text(aes(label = n), hjust = -0.15, family = "Arial", size = 2.1, color = COL_TEXT) +
    scale_fill_manual(values = class_cols, guide = "none") +
    scale_x_continuous(limits = c(0, max(class_summary$n, na.rm = TRUE) * 1.25), expand = c(0, 0)) +
    labs(title = "classes", x = NULL, y = NULL) +
    theme_fig5(7) +
    theme(axis.text.y = element_text(size = 6.2), axis.text.x = element_blank(),
          axis.ticks = element_blank(), axis.line = element_blank(),
          panel.grid = element_blank(), plot.title = element_text(size = 6.5, color = "#555555"))

  cowplot::plot_grid(label_plot, strip, heat, right, nrow = 1,
                     rel_widths = c(2.7, 0.12, 7.1, 1.0), align = "h")
}
