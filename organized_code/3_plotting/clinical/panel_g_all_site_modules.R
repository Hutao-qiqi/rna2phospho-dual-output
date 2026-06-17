source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_common.R")

make_panel_g_all_site_modules <- function() {
  centers <- read_matrix_tsv(file.path(.FIG5_DATA, "fig5b_all_site_module_risk_minus_protective_fraction.tsv"))
  density <- read_matrix_tsv(file.path(.FIG5_DATA, "fig5b_all_site_module_significance_density.tsv"))
  modules <- read.delim(file.path(.FIG5_DATA, "fig5b_all_site_module_summary.tsv"),
                        sep = "\t", stringsAsFactors = FALSE)
  cancer_info <- read.delim(file.path(.FIG5_DATA, "fig5b_all_site_module_cancer_order.tsv"),
                            sep = "\t", stringsAsFactors = FALSE)
  modules$module_id <- factor(modules$module_id, levels = rownames(centers))
  cancers <- colnames(centers)

  heat <- matrix_to_long(centers, "module_id", "cancer", "directional_fraction")
  sig <- matrix_to_long(density, "module_id", "cancer", "sig_density")
  heat <- merge(heat, sig, by = c("module_id", "cancer"), all.x = TRUE)
  heat$module_id <- factor(heat$module_id, levels = rev(rownames(centers)))
  heat$cancer <- factor(heat$cancer, levels = cancers)
  heat$dot <- cut(heat$sig_density, breaks = c(-Inf, 0.10, 0.20, Inf),
                  labels = c("none", "q<0.10 >=10%", "q<0.10 >=20%"))

  events <- cancer_info[match(cancers, cancer_info$short), ]
  events$cancer <- factor(events$short, levels = cancers)
  events$events_scaled <- events$events / max(events$events, na.rm = TRUE)

  module_order <- rev(rownames(centers))
  module_meta <- modules[match(module_order, modules$module_id), ]
  module_meta$module_id <- factor(module_meta$module_id, levels = module_order)
  module_meta$size_scaled <- module_meta$n_sites / max(module_meta$n_sites, na.rm = TRUE)

  pal <- fig2_heat_palette(101)
  module_cols <- c("risk" = FIG2_CLUSTER_COLORS[2], "protective" = FIG2_CLUSTER_COLORS[3], "mixed" = FIG2_CLUSTER_COLORS[1])

  top <- ggplot(events, aes(cancer, events_scaled)) +
    geom_col(fill = COL_GREY, width = 0.72) +
    geom_text(aes(label = events), y = 1.08, family = "Arial", size = 1.75, color = "#555555") +
    scale_y_continuous(limits = c(0, 1.22), expand = c(0, 0)) +
    labs(title = "All-site clinical phosphosite effect modules",
         subtitle = "Color: risk-site fraction minus protective-site fraction",
         x = NULL, y = NULL) +
    theme_fig5(7) +
    theme(axis.text.x = element_blank(), axis.text.y = element_blank(),
          axis.ticks = element_blank(), axis.line = element_blank(),
          panel.grid = element_blank(), plot.margin = margin(3, 4, 0, 4, "pt"))

  label <- ggplot(module_meta, aes(1, module_id)) +
    geom_text(aes(label = paste0(as.character(module_id), "\n", direction, " | ", format(n_sites, big.mark = ","))),
              hjust = 1, family = "Arial", size = 2.2, lineheight = 0.88, color = COL_TEXT) +
    xlim(0, 1.05) +
    labs(x = NULL, y = NULL) +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(0, 2, 0, 0, "pt"))

  strip <- ggplot(module_meta, aes(1, module_id, fill = direction)) +
    geom_tile(width = 0.75, height = 0.82) +
    scale_fill_manual(values = module_cols, guide = "none") +
    labs(x = NULL, y = NULL) +
    theme_void(base_family = "Arial") +
    theme(plot.margin = margin(0, 0, 0, 0, "pt"))

  heatmap <- ggplot(heat, aes(cancer, module_id, fill = directional_fraction)) +
    geom_tile(color = "white", linewidth = 0.28) +
    scale_fill_gradientn(colors = pal, limits = c(-0.8, 0.8), oob = scales::squish,
                         name = "risk - protective") +
    labs(x = NULL, y = NULL) +
    theme_fig5(7) +
    theme(axis.text.x = element_text(angle = 55, hjust = 1, vjust = 1, size = 6.4, face = "bold"),
          axis.text.y = element_blank(),
          axis.ticks = element_blank(),
          axis.line = element_blank(),
          panel.grid = element_blank(),
          legend.position = "bottom",
          legend.key.size = unit(3.2, "mm"),
          plot.margin = margin(0, 3, 1, 3, "pt"))

  bars <- ggplot(module_meta, aes(size_scaled, module_id)) +
    geom_col(fill = COL_GREY, width = 0.56) +
    geom_text(aes(x = 1.04, label = paste0(round(shared_site_fraction * 100), "%")),
              hjust = 0, family = "Arial", size = 2.0, color = "#555555") +
    scale_x_continuous(limits = c(0, 1.32), expand = c(0, 0)) +
    labs(x = NULL, y = NULL, title = "size / shared") +
    theme_fig5(7) +
    theme(axis.text = element_blank(), axis.ticks = element_blank(), axis.line = element_blank(),
          panel.grid = element_blank(), plot.title = element_text(size = 6.4, color = "#555555"))

  body <- cowplot::plot_grid(label, strip, heatmap, bars, nrow = 1,
                             rel_widths = c(1.15, 0.14, 6.0, 1.0), align = "h")
  cowplot::plot_grid(top, body, ncol = 1, rel_heights = c(0.20, 1.0), align = "v")
}
