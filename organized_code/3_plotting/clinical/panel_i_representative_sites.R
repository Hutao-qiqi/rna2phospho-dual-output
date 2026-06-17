source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_common.R")

clip_value <- function(x, upper) {
  pmin(pmax(as.numeric(x), 0), upper)
}

wrap_label <- function(x, width = 34) {
  vapply(strwrap(x, width = width, simplify = FALSE), paste, collapse = "\n", FUN.VALUE = character(1))
}

make_panel_i_representative_sites <- function() {
  cancer_mat <- read_matrix_tsv(file.path(.FIG5_DATA, "panel_i_site_specificity_cluster_cancer_matrix.tsv"))
  kegg_mat <- read_matrix_tsv(file.path(.FIG5_DATA, "panel_i_site_specificity_kegg_matrix.tsv"))
  cluster_meta <- read.delim(file.path(.FIG5_DATA, "panel_i_site_specificity_cluster_meta.tsv"),
                             sep = "\t", stringsAsFactors = FALSE)
  pathway_meta <- read.delim(file.path(.FIG5_DATA, "panel_i_site_specificity_kegg_pathway_labels.tsv"),
                             sep = "\t", stringsAsFactors = FALSE)

  cluster_order <- rownames(cancer_mat)
  cancer_order <- colnames(cancer_mat)
  kegg_mat <- kegg_mat[, cluster_order, drop = FALSE]
  pathway_meta <- pathway_meta[match(rownames(kegg_mat), pathway_meta$kegg_pathway), ]
  pathway_meta$display_label <- wrap_label(pathway_meta$kegg_label, width = 38)
  pathway_levels <- rev(rownames(kegg_mat))

  cluster_meta <- cluster_meta[match(cluster_order, cluster_meta$site_cluster), ]
  cluster_meta$cluster_label <- paste0(
    cluster_meta$site_cluster, "  ",
    format(cluster_meta$n_sites, big.mark = ","), " sites"
  )
  cluster_label_map <- setNames(cluster_meta$cluster_label, cluster_meta$site_cluster)
  cluster_meta$site_cluster <- factor(cluster_meta$site_cluster, levels = cluster_order)

  cancer_long <- matrix_to_long(cancer_mat, "site_cluster", "cancer", "signal")
  cancer_long$site_cluster <- factor(cancer_long$site_cluster, levels = rev(cluster_order))
  cancer_long$cancer <- factor(cancer_long$cancer, levels = cancer_order)
  cancer_long$signal_plot <- clip_value(cancer_long$signal, 3.5)

  kegg_long <- matrix_to_long(kegg_mat, "kegg_pathway", "site_cluster", "neglog10_q")
  kegg_long$kegg_pathway <- factor(kegg_long$kegg_pathway, levels = pathway_levels)
  kegg_long$site_cluster <- factor(kegg_long$site_cluster, levels = cluster_order)
  kegg_long$neglog10_q_plot <- clip_value(kegg_long$neglog10_q, 8)

  site_pal <- grDevices::colorRampPalette(c("#F7F7F7", "#C1D8E9", "#92B1D9", "#315F8D"))(101)
  kegg_pal <- grDevices::colorRampPalette(c("#F7F7F7", "#F6C8B6", "#ED8D5A", "#9A4E45"))(101)

  cancer_heat <- ggplot(cancer_long, aes(cancer, site_cluster, fill = signal_plot)) +
    geom_tile(color = "white", linewidth = 0.22) +
    scale_fill_gradientn(colors = site_pal, limits = c(0, 3.5), oob = scales::squish,
                         name = "site signal") +
    scale_y_discrete(labels = cluster_label_map) +
    labs(title = "Cancer-specific phosphosite clusters",
         subtitle = "All parent-mRNA-independent site-cancer rows; color is unsigned clinical signal, no direction used",
         x = NULL, y = NULL) +
    theme_fig5(7) +
    theme(axis.text.x = element_text(angle = 55, hjust = 1, vjust = 1, size = 6.2, face = "bold"),
          axis.text.y = element_text(size = 6.2, color = COL_TEXT),
          axis.ticks = element_blank(),
          axis.line = element_blank(),
          panel.grid = element_blank(),
          legend.position = "bottom",
          legend.key.width = unit(12, "mm"),
          plot.margin = margin(3, 5, 2, 5, "pt"))

  kegg_heat <- ggplot(kegg_long, aes(site_cluster, kegg_pathway, fill = neglog10_q_plot)) +
    geom_tile(color = "white", linewidth = 0.22) +
    scale_fill_gradientn(colors = kegg_pal, limits = c(0, 8), oob = scales::squish,
                         name = "KEGG -log10 FDR") +
    scale_y_discrete(labels = setNames(pathway_meta$display_label, pathway_meta$kegg_pathway)) +
    labs(title = "KEGG pathway clustering of site clusters",
         subtitle = "Gene-level over-representation; KEGG_MEDICUS terms removed; rows clustered by enrichment profile",
         x = NULL, y = NULL) +
    theme_fig5(7) +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5, size = 6.2, face = "bold"),
          axis.text.y = element_text(size = 5.6, color = COL_TEXT, lineheight = 0.86),
          axis.ticks = element_blank(),
          axis.line = element_blank(),
          panel.grid = element_blank(),
          legend.position = "bottom",
          legend.key.width = unit(12, "mm"),
          plot.margin = margin(2, 5, 2, 5, "pt"))

  cowplot::plot_grid(cancer_heat, kegg_heat, ncol = 1, rel_heights = c(1.0, 1.65), align = "v")
}
