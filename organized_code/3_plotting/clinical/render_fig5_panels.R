#!/usr/bin/env Rscript

suppressMessages({
  library(ggplot2)
  library(cowplot)
  library(grid)
  library(svglite)
})

PANELS <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels"

source(file.path(PANELS, "panel_common.R"))
source(file.path(PANELS, "panel_a.R"))
source(file.path(PANELS, "panel_b.R"))
source(file.path(PANELS, "panel_c.R"))
source(file.path(PANELS, "panel_d.R"))
source(file.path(PANELS, "panel_e.R"))
source(file.path(PANELS, "panel_f.R"))
source(file.path(PANELS, "panel_g_all_site_modules.R"))
source(file.path(PANELS, "panel_h_signed_ptmsea.R"))
source(file.path(PANELS, "panel_i_representative_sites.R"))

panel_specs <- list(
  list(stem = "panel_a_tcga_predicted_ps6_state_raincloud", plot = make_panel_a(), width = 10.8, height = 6.2),
  list(stem = "panel_b_tcga_kirc_survival_mrna_adjusted", plot = make_panel_b(), width = 8.2, height = 3.5),
  list(stem = "panel_c_cptac_target_site_match", plot = make_panel_c(), width = 7.4, height = 3.4),
  list(stem = "panel_d_cptac_target_mtor_axis_bubble", plot = make_panel_d(), width = 8.7, height = 3.8),
  list(stem = "panel_e_tcga_kirc_site_over_parent_waterfall", plot = make_panel_e(), width = 9.6, height = 3.6),
  list(stem = "panel_f_ccrcc_mtori_response", plot = make_panel_f(), width = 5.7, height = 3.4),
  list(stem = "panel_g_all_site_clinical_effect_modules", plot = make_panel_g_all_site_modules(), width = 13.8, height = 5.6),
  list(stem = "panel_h_signed_ptmsea_heatmap", plot = make_panel_h_signed_ptmsea(), width = 13.8, height = 11.0),
  list(stem = "panel_i_representative_site_effect_heatmap", plot = make_panel_i_representative_sites(), width = 16.2, height = 9.4)
)

for (spec in panel_specs) {
  save_panel(spec$plot, spec$stem, spec$width, spec$height)
}

message("Done.")
