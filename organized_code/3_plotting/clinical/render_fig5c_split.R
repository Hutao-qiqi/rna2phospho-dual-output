#!/usr/bin/env Rscript
# 把 fig5c (KIRC 生存) 拆成两张独立图:
#   fig5c1_kirc_survival_km     —— KM 曲线 + at-risk 表
#   fig5c2_kirc_cox_mrna_adjusted —— mRNA 校正 Cox forest
# 复用 panel_b.R 的 make_panel_b_km() / make_panel_b_forest()，数值不变。
suppressMessages({ library(ggplot2); library(cowplot); library(grid); library(svglite) })

PANELS <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels"
source(file.path(PANELS, "panel_common.R"))
source(file.path(PANELS, "panel_b.R"))

save_panel(make_panel_b_km(),     "fig5c1_kirc_survival_km",        width = 4.7, height = 3.5)
save_panel(make_panel_b_forest(), "fig5c2_kirc_cox_mrna_adjusted",  width = 3.8, height = 3.5)

message("fig5c split done.")
