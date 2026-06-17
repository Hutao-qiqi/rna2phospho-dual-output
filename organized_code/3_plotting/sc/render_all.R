#!/usr/bin/env Rscript
# render_all.R — Fig 3 全部图逐张渲染（5 主 panel + ~16 扩展图），写回源文件夹。
# 不拼图。每张独立出 svg + pdf + png（矢量字体可在 Illustrator 编辑）。
#
# 运行：
#   Rscript render_all.R            # 渲染全部已定义的图
#   Rscript render_all.R fig1 fig5  # 只渲染 stem 含 fig1 / fig5 的图（便于迭代）
#
# 结构（模仿 fig2 scripts）：
#   panels/theme_fig3.R   共享主题 + 调色板 + helper
#   panels/panel_a..e.R   主图 Fig3 五 panel → fig3_panel_*
#   figures/figNN.R       扩展图 → figN_*

suppressMessages({
  library(ggplot2); library(cowplot); library(grid); library(scales)
  library(ragg); library(svglite)
})

SCRIPTS <- "E:/data/gongke/TCGA-TCPA/paper_final/fig3/scripts"
PANELS  <- file.path(SCRIPTS, "panels")
FIGURES <- file.path(SCRIPTS, "figures")
OUT     <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/04_figure_source_data/fig3"

source(file.path(PANELS, "theme_fig3.R"))
for (f in list.files(PANELS, "^panel_.*\\.R$", full.names = TRUE)) source(f)
if (dir.exists(FIGURES))
  for (f in list.files(FIGURES, "\\.R$", full.names = TRUE)) source(f)

save_fig <- function(p, stem, w_mm, h_mm) {
  grDevices::cairo_pdf(file.path(OUT, paste0(stem, ".pdf")),
                       width = w_mm / 25.4, height = h_mm / 25.4, family = "Arial")
  print(p); invisible(grDevices::dev.off())
  svglite::svglite(file.path(OUT, paste0(stem, ".svg")),
                   width = w_mm / 25.4, height = h_mm / 25.4)
  print(p); invisible(grDevices::dev.off())
  ragg::agg_png(file.path(OUT, paste0(stem, ".png")),
                width = w_mm, height = h_mm, units = "mm", res = 350)
  print(p); invisible(grDevices::dev.off())
  message("  ", stem)
}

# 注册表：函数名 / 输出 stem / 宽 mm / 高 mm
reg <- list(
  list("make_panel_a", "fig3_panel_a", 180,  50),
  list("make_panel_b", "fig3_panel_b",  84,  64),
  list("make_panel_c", "fig3_panel_c", 118,  64),
  list("make_panel_d", "fig3_panel_d",  96, 110),
  list("make_panel_e", "fig3_panel_e", 122,  66),
  list("make_fig01", "fig1_pathway_attention_heatmap", 235, 96),
  list("make_fig02", "fig2_clone_sensitivity_dumbbell", 118, 58),
  list("make_fig03", "fig3_hela_umap_error",            185, 112),
  list("make_fig04", "fig4_cross_cohort_spearman_matrix", 108, 190),
  list("make_fig05", "fig5_gnn_residual_contribution",  120, 165),
  list("make_fig06", "fig6_qurie_ibrutinib_delta_polar", 130, 130),
  list("make_fig07", "fig7_expanded_gnn_network",        155, 150),
  list("make_fig08", "fig8_phospho_periodic_table",      235, 82),
  list("make_fig09", "fig9_hela_nmf_components",         150, 90),
  list("make_fig10", "fig10_nmf_vs_error",               165, 110),
  list("make_fig11", "fig11_nmf_classic_heatmap",        180, 120),
  list("make_fig12", "fig12_nmf3_phenotype",             185, 80),
  list("make_fig13", "fig13_multi_cohort_nmf",           185, 200),
  list("make_fig14", "fig14_rna_phospho_crossval",       160, 78),
  list("make_fig15", "fig15_cross_cohort_hallmark",      160, 150),
  list("make_fig16", "fig16_cell_cycle_validation",      128, 66),
  list("make_fig17", "fig17_cross_cohort_phospho_rna",   165, 120),
  list("make_fig18", "fig18_cohort_landscape",           185, 120),
  list("make_fig19", "fig19_per_readout_difficulty",     178, 180),
  list("make_fig20", "fig20_qurie_pathway_breakdown",    125, 92),
  list("make_fig21", "fig21_pdo_caf_diagnostic",         150, 92),
  # —— reviewer-requested 扩展图（fig22–26）——
  list("make_fig22", "fig22_component_ablation_paired",  178, 78),
  list("make_fig23", "fig23_gnn_vs_site_aware_mlp",       96, 104),
  list("make_fig24", "fig24_fivefold_stability",         120, 150),
  list("make_fig25", "fig25_calibration_reliability",    120, 120),
  list("make_fig26", "fig26_attention_negative_control",  92, 96)
)

args <- commandArgs(trailingOnly = TRUE)
match_arg <- function(stem) length(args) == 0 ||
  any(vapply(args, function(a) grepl(a, stem, fixed = TRUE), logical(1)))

message("Rendering → ", OUT)
for (item in reg) {
  fname <- item[[1]]; stem <- item[[2]]
  if (!match_arg(stem)) next
  if (!exists(fname, mode = "function")) { message("  skip (todo) ", stem); next }
  ok <- tryCatch({
    p <- get(fname)(); save_fig(p, stem, item[[3]], item[[4]]); TRUE
  }, error = function(e) { message("  FAIL ", stem, ": ", conditionMessage(e)); FALSE })
}
message("Done.")
