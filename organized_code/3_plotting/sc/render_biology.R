#!/usr/bin/env Rscript
# render_biology.R — Fig 3（生物学版）：SCP682-SC recovers pathway-organized
#   phospho-signalling states in single cells。临床/增殖内容移到 Fig 5，不在此图。
#
# 4 个数据 panel（a 架构在 Illustrator）：
#   b  GSE300551 逐读数外部验证，按通路上色，p38 两克隆分别标注
#   c  HeLa + Vivo 跨平台/跨物种验证
#   d  通路注意力路由（14 代表读数 × 9 token，显示名映射，逐读数最强 token 描边）
#   e  代表读数单细胞 predicted vs observed（NFE2L2 / LCP2 / CTNND1 / STAT3）
#
# 数据：main_figure_biology_v1/source_data/*.tsv（用户准备）。输出回 main_figure_biology_v1/。

suppressMessages({
  library(ggplot2); library(grid); library(scales); library(hexbin)
  library(ragg); library(svglite)
})

SCRIPTS <- "E:/data/gongke/TCGA-TCPA/paper_final/fig3/scripts"
source(file.path(SCRIPTS, "panels", "theme_fig3.R"))

BIO <- "E:/data/gongke/TCGA-TCPA/paper_final/fig3/main_figure_biology_v1"
SD  <- file.path(BIO, "source_data")
OUTDIR <- "E:/data/gongke/TCGA-TCPA/paper_final/fig3"   # 主图 a–f 面板输出位置
EDDIR  <- "E:/data/gongke/TCGA-TCPA/paper_final/fig3/extended_data"  # Extended Data 面板

# 候选补充面板的数据源
SUPP    <- "E:/data/gongke/TCGA-TCPA/paper_final/fig3/supp_tables"
GABL    <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/sc_kirc_rps6_validation/tables"
FIG3SRC <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/04_figure_source_data/fig3"
RVDIR   <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/04_figure_source_data/reviewer_requested_tables_v2"
rdp <- function(dir, f) utils::read.delim(file.path(dir, f), sep = "\t",
                                          stringsAsFactors = FALSE, check.names = FALSE)

# ---- 通路 family 配色（low-sat，与 house style 协调）----
FAM_ORDER <- c("TCR/BCR receptor", "JAK/STAT", "MAPK/stress", "NF-kB",
               "adhesion/PI3K", "other")
FAM_FILL <- c("TCR/BCR receptor" = "#4F86C6", "JAK/STAT" = "#51999F",
              "MAPK/stress" = "#ED8D5A", "NF-kB" = "#C0504D",
              "adhesion/PI3K" = "#9C8FC4", "other" = "#B0B0B0")

rd <- function(f) utils::read.delim(file.path(SD, f), sep = "\t",
                                    stringsAsFactors = FALSE, check.names = FALSE)

# 比 house style 大一号的字体（终稿 Fig 3 面板用；不改全局 theme_fig3，避免影响其它图）
theme_fig3_big <- function() {
  theme_fig3() + ggplot2::theme(
    plot.title    = ggplot2::element_text(size = 9),
    plot.subtitle = ggplot2::element_text(size = 7),
    axis.title.x  = ggplot2::element_text(size = 8.5),
    axis.title.y  = ggplot2::element_text(size = 8.5),
    axis.text.x   = ggplot2::element_text(size = 7.5),
    axis.text.y   = ggplot2::element_text(size = 7.5),
    strip.text    = ggplot2::element_text(size = 8),
    legend.text   = ggplot2::element_text(size = 7.5),
    legend.title  = ggplot2::element_text(size = 7.5))
}

# p38 双克隆诚实标注（不挑好的）
relabel_p38 <- function(df) {
  df$disp <- df$target_display
  df$disp[df$target_id == "MAPK14_pSitePending"]      <- "p38 MAPK (A16016A)"
  df$disp[df$target_id == "MAPK14_pSitePending_D3F9"] <- "p38 MAPK (D3F9)"
  df
}

save_fig <- function(p, stem, w, h, dir = OUTDIR) {
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  grDevices::cairo_pdf(file.path(dir, paste0(stem, ".pdf")),
                       width = w / 25.4, height = h / 25.4, family = "Arial")
  print(p); invisible(grDevices::dev.off())
  svglite::svglite(file.path(dir, paste0(stem, ".svg")), width = w / 25.4, height = h / 25.4)
  print(p); invisible(grDevices::dev.off())
  ragg::agg_png(file.path(dir, paste0(stem, ".png")), width = w, height = h, units = "mm", res = 350)
  print(p); invisible(grDevices::dev.off())
  message("  ", stem, "  → ", basename(dir))
}

# ============================================================= panel b
make_bio_b <- function() {
  d <- relabel_p38(rd("fig3_biology_gse300551_ranked_readouts.tsv"))
  d$spearman <- as.numeric(d$spearman)
  n_total <- nrow(d)
  d <- d[d$spearman >= 0.30, ]                              # 主图只放预测得好的读数（全集见 Supp）
  n_keep <- nrow(d)
  d$fam <- factor(d$family_display, levels = FAM_ORDER)
  d <- d[order(d$spearman), ]
  d$disp <- factor(d$disp, levels = d$disp)

  ggplot2::ggplot(d, ggplot2::aes(y = disp)) +
    ggplot2::geom_vline(xintercept = 0, color = COL_ZERO, linewidth = 0.3) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, xend = spearman, yend = disp),
                          color = "#C8C8C8", linewidth = 0.5) +
    ggplot2::geom_point(ggplot2::aes(x = spearman, fill = fam),
                        shape = 21, color = "black", size = 2.0, stroke = 0.3) +
    ggplot2::geom_text(ggplot2::aes(x = spearman + 0.014, label = sprintf("%.2f", spearman)),
                       hjust = 0, size = 6.4 / PT, family = "Arial", color = COL_SUB) +
    ggplot2::scale_fill_manual(values = FAM_FILL, breaks = FAM_ORDER,
                               drop = TRUE, name = NULL) +
    ggplot2::scale_x_continuous(limits = c(0, 0.78), breaks = seq(0, 0.6, 0.2),
                                expand = c(0, 0)) +
    ggplot2::labs(
      x = "Single-cell Spearman ρ (predicted vs observed)", y = NULL,
      title = "GSE300551 — well-predicted single-cell readouts",
      subtitle = sprintf("ρ ≥ 0.30 (%d of %d readouts); full per-readout set in Supplementary", n_keep, n_total)) +
    theme_fig3_big() +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 8),
                   axis.ticks.y = ggplot2::element_blank(),
                   legend.position = "bottom") +
    ggplot2::guides(fill = ggplot2::guide_legend(nrow = 2, byrow = TRUE,
                                                 override.aes = list(size = 2.4)))
}

# ============================================================= panel c
make_bio_c <- function() {
  d <- rd("fig3_biology_external_by_readout.tsv")
  d <- d[d$cohort %in% c("HeLa", "Vivo-Th17"), ]
  d$spearman <- as.numeric(d$spearman)
  d <- d[d$spearman >= 0.30, ]                              # 主图只放迁移成功的读数（弱的见 Supp）
  d$fam <- factor(d$family_display, levels = FAM_ORDER)
  d$coh <- factor(d$cohort, levels = c("HeLa", "Vivo-Th17"),
                  labels = c("SIGNAL-seq HeLa (human)", "Vivo-seq Th17 (mouse)"))
  d <- d[order(d$coh, d$spearman), ]
  # y = target_display 直接做因子（HeLa/Vivo 读数名无重叠），facet free_y 自动取子集，避免标签错位
  d$disp <- factor(d$target_display, levels = d$target_display)

  ggplot2::ggplot(d, ggplot2::aes(y = disp)) +
    ggplot2::geom_vline(xintercept = 0, color = COL_ZERO, linewidth = 0.3) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, xend = spearman, yend = disp),
                          color = "#C8C8C8", linewidth = 0.5) +
    ggplot2::geom_point(ggplot2::aes(x = spearman, fill = fam),
                        shape = 21, color = "black", size = 2.0, stroke = 0.3) +
    ggplot2::geom_text(ggplot2::aes(x = spearman + 0.014, label = sprintf("%.2f", spearman)),
                       hjust = 0, size = 6.4 / PT, family = "Arial", color = COL_SUB) +
    ggplot2::scale_fill_manual(values = FAM_FILL, breaks = FAM_ORDER, drop = TRUE, name = NULL) +
    ggplot2::scale_x_continuous(limits = c(0, 0.74), breaks = seq(0, 0.6, 0.2),
                                expand = c(0, 0)) +
    ggplot2::facet_grid(coh ~ ., scales = "free_y", space = "free_y") +
    ggplot2::labs(
      x = "Single-cell Spearman ρ", y = NULL,
      title = "Cross-platform & cross-species transfer",
      subtitle = "ρ ≥ 0.30; weaker readouts in Supplementary") +
    theme_fig3_big() +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 8),
                   axis.ticks.y = ggplot2::element_blank(),
                   legend.position = "bottom",
                   strip.text.y = ggplot2::element_text(size = 8, angle = 0)) +
    ggplot2::guides(fill = ggplot2::guide_legend(nrow = 2, byrow = TRUE,
                                                 override.aes = list(size = 2.4)))
}

# ============================================================= panel d
make_bio_d <- function() {
  a <- rd("fig3_biology_pathway_attention_selected.tsv")
  a$mean_attention <- as.numeric(a$mean_attention)
  pdisp <- c(BCR_BTK_axis = "BCR/BTK", BTK_PLCG2_axis = "BTK/PLCG2",
             ERK_axis = "ERK/MAPK", AKT_mTOR_S6_axis = "AKT axis",
             NFkB_axis = "NF-kB", stress_ifn = "stress/IFN",
             cell_cycle = "cell cycle", ribosomal = "ribosomal",
             random_control = "random")
  col_order <- c("BCR/BTK", "BTK/PLCG2", "ERK/MAPK", "AKT axis", "NF-kB",
                 "stress/IFN", "cell cycle", "ribosomal", "random")
  a$pw <- factor(pdisp[a$pathway], levels = col_order)

  # 读数按 family 分组排序
  fam_rank <- c("receptor" = 1, "stat" = 2, "stress" = 3, "nfkb" = 4, "adhesion" = 5, "other" = 6)
  ro <- unique(a[, c("target_display", "family")])
  ro <- ro[order(fam_rank[ro$family], ro$target_display), ]
  a$rd_ <- factor(a$target_display, levels = rev(ro$target_display))

  # 每读数最强 token 描边
  mx <- do.call(rbind, lapply(split(a, a$target_display), function(s) s[which.max(s$mean_attention), ]))

  ggplot2::ggplot(a, ggplot2::aes(x = pw, y = rd_, fill = mean_attention)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.4) +
    ggplot2::geom_tile(data = mx, fill = NA, color = "black", linewidth = 0.5) +
    ggplot2::geom_text(ggplot2::aes(label = sub("^0", "", sprintf("%.2f", mean_attention))),
                       size = 4 / PT, family = "Arial", color = "#222222") +
    ggplot2::scale_fill_gradientn(colours = FIG3_SEQ_TEAL, name = "mean attn",
                                  breaks = c(0.10, 0.13, 0.16),
                                  guide = ggplot2::guide_colorbar(
                                    barwidth = ggplot2::unit(24, "mm"),
                                    barheight = ggplot2::unit(1.8, "mm"))) +
    ggplot2::scale_x_discrete(position = "top", expand = c(0, 0)) +
    ggplot2::scale_y_discrete(expand = c(0, 0)) +
    ggplot2::labs(
      x = NULL, y = NULL,
      title = "Pathway-attention routing (representative readouts)",
      subtitle = "Mean site→pathway attention; black outline = top token per readout; 'random' = negative control") +
    theme_fig3() +
    ggplot2::theme(
      axis.text.x = ggplot2::element_text(size = 5.8, angle = 40, hjust = 0),
      axis.text.y = ggplot2::element_text(size = 6.0),
      axis.ticks = ggplot2::element_blank(),
      axis.line = ggplot2::element_blank(),
      legend.position = "bottom",
      panel.border = ggplot2::element_rect(fill = NA, color = "#D0D0D0", linewidth = 0.3))
}

# ============================================================= panel e
make_bio_e <- function() {
  s <- rd("fig3_biology_scatter_clean.tsv")
  s$predicted <- as.numeric(s$predicted); s$observed <- as.numeric(s$observed)
  s$spearman  <- as.numeric(s$spearman)
  ord <- c("NFE2L2 / stress", "LCP2 / receptor", "CTNND1 / adhesion", "STAT3 / cytokine")
  s$facet <- factor(s$panel_label, levels = ord,
                    labels = vapply(ord, function(z) {
                      r <- unique(s$spearman[s$panel_label == z])[1]
                      sprintf("%s\nρ = %.2f", z, r)          # 两行，避免大字号下分面标题被截断
                    }, character(1)))

  ggplot2::ggplot(s, ggplot2::aes(x = predicted, y = observed)) +
    ggplot2::geom_hex(bins = 34) +
    ggplot2::geom_smooth(method = "lm", se = FALSE, color = "#ED8D5A",
                         linewidth = 0.5, formula = y ~ x) +
    ggplot2::scale_fill_gradientn(colours = FIG3_SEQ_TEAL, name = "cells",
                                  trans = "log10",
                                  guide = ggplot2::guide_colorbar(
                                    barwidth = ggplot2::unit(14, "mm"),
                                    barheight = ggplot2::unit(1.8, "mm"))) +
    ggplot2::facet_wrap(~ facet, nrow = 1, scales = "free") +
    ggplot2::labs(
      x = "Predicted phosphorylation (z)", y = "Observed (z)",
      title = "Single-cell predicted vs observed",
      subtitle = "Representative readouts across cohorts; orange = linear fit") +
    theme_fig3_big() +
    ggplot2::theme(legend.position = "bottom",
                   panel.spacing = ggplot2::unit(3, "mm"))
}

# ============================================================= benchmark
make_bio_bench <- function() {
  d <- rd("fig3_benchmark_gse300551_leaderboard.tsv")
  for (c in c("median", "q25", "q75")) d[[c]] <- as.numeric(d[[c]])
  d <- d[d$method != "scFoundation (LR)", ]                 # 同一 backbone 的线性 head=消融，移到 Supp，不当竞争方法
  d <- d[order(d$median), ]
  d$method <- factor(d$method, levels = d$method)         # 升序 → SCP682-SC 最右
  cols <- c("tGPT (LR)" = "#D8D8D8", "Cognate mRNA" = "#BFDFD2", "scGPT (LR)" = "#7BC0CD",
            "UCE (LR)" = "#51999F", "Geneformer (LR)" = "#4198AC", "SCimilarity (LR)" = "#DBCB92",
            "scFoundation (LR)" = "#ECB66C", "SCP682-SC" = "#ED8D5A")
  scp_med <- d$median[d$method == "SCP682-SC"]
  md <- ifelse(abs(d$median) < 0.005, 0, d$median)        # 避免 -0.00
  d$lab <- ifelse(d$method == "SCP682-SC", sprintf("%.2f", md),
                  sprintf("%.2f %s", md, ifelse(d$signif == "", "ns", d$signif)))

  ggplot2::ggplot(d, ggplot2::aes(x = method, y = median, fill = method)) +
    ggplot2::geom_hline(yintercept = 0, color = COL_ZERO, linewidth = 0.3, linetype = "dashed") +
    ggplot2::geom_hline(yintercept = scp_med, color = "#ED8D5A", linewidth = 0.4, linetype = "dashed") +
    ggplot2::geom_col(width = 0.7, color = "black", linewidth = 0.25) +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = q25, ymax = q75),
                           width = 0.25, linewidth = 0.3, color = "#333333") +
    ggplot2::geom_text(ggplot2::aes(y = pmax(q75, median) + 0.02, label = lab),
                       vjust = 0, size = 6 / PT, family = "Arial", color = COL_TEXT) +
    ggplot2::scale_fill_manual(values = cols, guide = "none") +
    ggplot2::scale_y_continuous(limits = c(-0.13, 0.56), breaks = seq(0, 0.5, 0.1),
                                expand = c(0, 0)) +
    ggplot2::labs(
      x = NULL, y = "Per-readout median Spearman ρ",
      title = "Single-cell benchmark (GSE300551)",
      subtitle = "11 readouts, 55,231 cells; median ± IQR") +
    theme_fig3_big() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 40, hjust = 1, size = 7),
                   axis.ticks.x = ggplot2::element_blank())
}

# ============================================================= win-scatter grid（scProTrans 同款）：SCP682-SC vs 每个 baseline 逐读数
make_bio_winscatter <- function() {
  d <- rd("fig3_benchmark_gse300551_per_readout.tsv")
  d$base_rho <- as.numeric(d$base_rho); d$scp_rho <- as.numeric(d$scp_rho)
  ord <- c("tGPT (LR)", "Cognate mRNA", "scGPT (LR)", "UCE (LR)", "Geneformer (LR)", "SCimilarity (LR)")
  d$baseline <- factor(d$baseline, levels = ord)
  d$win <- factor(d$scp_rho > d$base_rho, levels = c("TRUE", "FALSE"))
  ws <- do.call(rbind, lapply(split(d, d$baseline), function(s)
    data.frame(baseline = s$baseline[1], lab = sprintf("%d/%d", sum(s$scp_rho > s$base_rho), nrow(s)))))
  lim <- c(-0.46, 0.60)                                     # 含 baseline 负 ρ 的完胜点

  ggplot2::ggplot(d, ggplot2::aes(base_rho, scp_rho)) +
    ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#9A9A9A", linewidth = 0.35) +
    ggplot2::geom_hline(yintercept = 0, color = COL_ZERO, linewidth = 0.25) +
    ggplot2::geom_vline(xintercept = 0, color = COL_ZERO, linewidth = 0.25) +
    ggplot2::geom_point(ggplot2::aes(fill = win), shape = 21, color = "black",
                        size = 1.7, stroke = 0.25, alpha = 0.92) +
    ggplot2::geom_text(data = ws, ggplot2::aes(x = lim[2] - 0.02, y = lim[1] + 0.02, label = lab),
                       inherit.aes = FALSE, hjust = 1, vjust = 0, size = 7.5 / PT,
                       family = "Arial", fontface = "bold", color = "#1F5A66") +
    ggplot2::scale_fill_manual(values = c("TRUE" = "#3F8E9C", "FALSE" = "#C0504D"),
                               labels = c("TRUE" = "SCP682-SC better", "FALSE" = "baseline better"),
                               breaks = c("TRUE", "FALSE"), name = NULL, drop = FALSE) +
    ggplot2::scale_x_continuous(limits = lim, breaks = seq(-0.4, 0.4, 0.4)) +
    ggplot2::scale_y_continuous(limits = lim, breaks = seq(-0.4, 0.4, 0.4)) +
    ggplot2::facet_wrap(~ baseline, nrow = 1) +
    ggplot2::coord_equal() +
    ggplot2::labs(x = "Baseline per-readout Spearman ρ", y = "SCP682-SC ρ",
                  title = "Head-to-head: SCP682-SC vs each baseline, per readout",
                  subtitle = "GSE300551 matched readouts (n = 11); points above the diagonal = SCP682-SC wins (count per panel)") +
    theme_fig3_big() +
    ggplot2::theme(legend.position = "bottom", panel.spacing.x = ggplot2::unit(4, "pt"))
}

# ============================================================= f: 覆盖/迁移能力
make_bio_f <- function() {
  p <- rdp(SUPP, "supp_per_site_sc_vs_cognate.tsv")
  d <- p[p$cohort_name == "SIGNAL-seq HeLa" & p$site_class == "transfer_only", ]
  d$scp682_sc <- as.numeric(d$scp682_sc)
  nice <- c(CTNND1_T310 = "CTNND1 T310", PDPK1_S241 = "PDPK1 S241",
            MAP2K4_S257 = "MAP2K4 S257", NDRG1_T346 = "NDRG1 T346")
  fam  <- c(CTNND1_T310 = "adhesion/PI3K", PDPK1_S241 = "adhesion/PI3K",
            MAP2K4_S257 = "MAPK/stress", NDRG1_T346 = "adhesion/PI3K")
  d$lab <- ifelse(d$target_id %in% names(nice), nice[d$target_id], d$target_id)
  d$fam <- factor(fam[d$target_id], levels = FAM_ORDER)
  d <- d[order(d$scp682_sc), ]; d$lab <- factor(d$lab, levels = d$lab)

  ggplot2::ggplot(d, ggplot2::aes(y = lab, x = scp682_sc, fill = fam)) +
    ggplot2::geom_col(width = 0.62, color = "black", linewidth = 0.25) +
    ggplot2::geom_text(ggplot2::aes(x = scp682_sc + 0.015, label = sprintf("%.2f", scp682_sc)),
                       hjust = 0, size = 6.4 / PT, family = "Arial", color = COL_SUB) +
    ggplot2::scale_fill_manual(values = FAM_FILL, breaks = FAM_ORDER, drop = TRUE, name = NULL) +
    ggplot2::scale_x_continuous(limits = c(0, 0.78), breaks = seq(0, 0.6, 0.2), expand = c(0, 0)) +
    ggplot2::labs(
      x = "Single-cell Spearman ρ (external)", y = NULL,
      title = "Transfer to antibody-free phosphosites",
      subtitle = "HeLa sites with no single-cell label; same-name-supervised baselines: no model (n=0)") +
    theme_fig3_big() +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 8),
                   axis.ticks.y = ggplot2::element_blank(), legend.position = "bottom")
}

# ============================================================= g: 位点图消融（Internal 训练重建 + 5 个外部 SC 队列）
# Internal = 分布内训练重建（n=38，最 powered）；其余 5 个 = 外部 OOD。两组用点线分隔、分别标注。
make_bio_g <- function() {
  d <- rd("fig3_graph_ablation_by_cohort.tsv")
  d$n <- suppressWarnings(as.integer(d$n))
  d <- d[!is.na(d$n) & d$n >= 1, ]
  d$full <- as.numeric(d$full); d$no_graph <- as.numeric(d$no_graph)
  d <- d[d$cohort != "SIGNAL-seq PDO/CAF", ]                # 已从论文移除（用户决定 2026-06-14）
  short <- c("Internal (in-dist.)" = "Internal", "SIGNAL-seq HeLa" = "HeLa", "Blair" = "Blair",
             "GSE300551" = "GSE300551", "Vivo-seq Th17" = "Vivo-Th17")
  intern <- d[d$scope == "in_dist", ]
  ext    <- d[d$scope == "external", ]; ext <- ext[order(-ext$full), ]
  d <- rbind(intern, ext)
  d$clab <- factor(sprintf("%s\n(n=%d)", short[d$cohort], d$n),
                   levels = sprintf("%s\n(n=%d)", short[d$cohort], d$n))
  d$top  <- pmax(d$full, d$no_graph)
  d$star <- ifelse(is.na(d$signif) | d$signif %in% c("n/a", "NaN", ""), "", d$signif)
  sep_x  <- nrow(intern) + 0.5                              # Internal 与外部之间的分隔
  m <- rbind(data.frame(clab = d$clab, cond = "with graph", rho = d$full),
             data.frame(clab = d$clab, cond = "no graph",  rho = d$no_graph))
  m$cond <- factor(m$cond, levels = c("with graph", "no graph"))

  ggplot2::ggplot(m, ggplot2::aes(x = clab, y = rho, fill = cond)) +
    ggplot2::geom_hline(yintercept = 0, color = COL_ZERO, linewidth = 0.3) +
    ggplot2::geom_vline(xintercept = sep_x, linetype = "dotted", color = "#9A9A9A", linewidth = 0.35) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.7),
                      width = 0.62, color = "black", linewidth = 0.25) +
    ggplot2::geom_text(data = d, ggplot2::aes(x = clab, y = top + 0.03, label = star),
                       inherit.aes = FALSE, size = 6.4 / PT, family = "Arial", color = COL_TEXT) +
    ggplot2::annotate("text", x = nrow(intern) / 2 + 0.5, y = 0.56, label = "in-dist.",
                      family = "Arial", size = 6 / PT, fontface = "italic", color = "#555555") +
    ggplot2::annotate("text", x = (sep_x + nrow(d) + 0.5) / 2, y = 0.56, label = "external (OOD)",
                      family = "Arial", size = 6 / PT, fontface = "italic", color = "#555555") +
    ggplot2::scale_fill_manual(values = c("with graph" = "#ED8D5A", "no graph" = "#C4C4C4"), name = NULL) +
    ggplot2::scale_y_continuous(limits = c(-0.08, 0.60), breaks = seq(0, 0.5, 0.1), expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = "Per-readout median Spearman ρ",
                  title = "Site-graph ablation: internal + external cohorts",
                  subtitle = "With vs without expanded site graph; Internal (n=38) and GSE300551 significant") +
    theme_fig3_big() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(size = 7, lineheight = 0.9),
                   axis.ticks.x = ggplot2::element_blank(), legend.position = "bottom")
}

# ============================================================= f（小图·竖向紧凑版）：位点图消融，主图用
make_bio_g_small <- function() {
  d <- rd("fig3_graph_ablation_by_cohort.tsv")
  d$n <- suppressWarnings(as.integer(d$n)); d <- d[!is.na(d$n) & d$n >= 1, ]
  d$full <- as.numeric(d$full); d$no_graph <- as.numeric(d$no_graph)
  d <- d[d$cohort != "SIGNAL-seq PDO/CAF", ]                # 主图小图不放 PDO/CAF（失败队列，仅在 ED 全图披露）
  short <- c("Internal (in-dist.)" = "Int", "SIGNAL-seq HeLa" = "HeLa", "Blair" = "Blair",
             "GSE300551" = "GSE", "Vivo-seq Th17" = "Vivo")
  intern <- d[d$scope == "in_dist", ]; ext <- d[d$scope == "external", ]; ext <- ext[order(-ext$full), ]
  d <- rbind(intern, ext)
  d$clab <- factor(sprintf("%s (%d)", short[d$cohort], d$n), levels = sprintf("%s (%d)", short[d$cohort], d$n))
  d$top  <- pmax(d$full, d$no_graph)
  d$star <- ifelse(is.na(d$signif) | d$signif %in% c("n/a", "NaN", ""), "ns", d$signif)
  sep_x  <- nrow(intern) + 0.5
  m <- rbind(data.frame(clab = d$clab, cond = "with graph", rho = d$full),
             data.frame(clab = d$clab, cond = "no graph",  rho = d$no_graph))
  m$cond <- factor(m$cond, levels = c("with graph", "no graph"))

  ggplot2::ggplot(m, ggplot2::aes(x = clab, y = rho, fill = cond)) +
    ggplot2::geom_hline(yintercept = 0, color = COL_ZERO, linewidth = 0.3) +
    ggplot2::geom_vline(xintercept = sep_x, linetype = "dotted", color = "#9A9A9A", linewidth = 0.35) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.72), width = 0.64,
                      color = "black", linewidth = 0.2) +
    ggplot2::geom_text(data = d, ggplot2::aes(x = clab, y = top + 0.025, label = star),
                       inherit.aes = FALSE, size = 5.5 / PT, family = "Arial", color = COL_TEXT) +
    ggplot2::scale_fill_manual(values = c("with graph" = "#ED8D5A", "no graph" = "#C4C4C4"), name = NULL) +
    ggplot2::scale_y_continuous(limits = c(-0.05, 0.56), breaks = seq(0, 0.4, 0.2), expand = c(0, 0)) +
    ggplot2::labs(x = NULL, y = "Median Spearman ρ", title = "Site-graph ablation",
                  subtitle = "with vs without GNN") +
    theme_fig3_big() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(size = 6.5, angle = 45, hjust = 1),
                   axis.ticks.x = ggplot2::element_blank(), legend.position = "bottom",
                   plot.title = ggplot2::element_text(size = 8.5),
                   plot.subtitle = ggplot2::element_text(size = 6.8),
                   legend.text = ggplot2::element_text(size = 6.8),
                   legend.key.size = ggplot2::unit(3, "mm")) +
    ggplot2::guides(fill = ggplot2::guide_legend(nrow = 1))
}

# ============================================================= h: 校准（z 化可靠性曲线）
make_bio_h <- function() {
  z <- rdp(FIG3SRC, "fig25_calibration_curves_z.tsv")
  s <- rdp(FIG3SRC, "fig25_calibration_cohort_summary.tsv")
  z$pred_z <- as.numeric(z$pred_z); z$obs_z <- as.numeric(z$obs_z)
  nice <- c(signal_seq_gse256403_hela_2024 = "SIGNAL-seq HeLa",
            gse300551_iccite_plex_kinase_2025 = "GSE300551")   # 只留两个 powered 队列；Vivo 进 Supp
  z <- z[z$cohort_id %in% names(nice), ]
  z$coh <- factor(nice[z$cohort_id], levels = unname(nice))
  z$grp <- interaction(z$cohort_id, z$target_id, drop = TRUE)
  s <- s[s$cohort_id %in% names(nice), ]; s$coh <- factor(nice[s$cohort_id], levels = unname(nice))
  s$lab <- sprintf("ρ̃ = %.2f", as.numeric(s$median_bin_spearman))
  lim <- c(-3.05, 3.05)
  ggplot2::ggplot(z, ggplot2::aes(pred_z, obs_z)) +
    ggplot2::annotate("segment", x = lim[1], y = lim[1], xend = lim[2], yend = lim[2],
                      color = "#B0B0B0", linewidth = 0.35, linetype = "dashed") +
    ggplot2::geom_line(ggplot2::aes(group = grp), color = "#51999F", linewidth = 0.3, alpha = 0.45) +
    ggplot2::geom_text(data = s, ggplot2::aes(x = lim[1] + 0.1, y = lim[2] - 0.1, label = lab),
                       hjust = 0, vjust = 1, size = 6 / PT, family = "Arial",
                       color = COL_TEXT, inherit.aes = FALSE) +
    ggplot2::facet_wrap(~ coh, nrow = 1) +
    ggplot2::scale_x_continuous(limits = lim, breaks = c(-2, 0, 2)) +
    ggplot2::scale_y_continuous(limits = lim, breaks = c(-2, 0, 2)) +
    ggplot2::coord_equal() +
    ggplot2::labs(x = "Predicted bin mean (z)", y = "Observed (z)",
                  title = "Calibration: binned predicted vs observed (z-scored)",
                  subtitle = "Each readout's 10 bins vs y = x; Vivo in Supplementary") +
    theme_fig3_big() +
    ggplot2::theme(panel.spacing = ggplot2::unit(2.5, "mm"))
}

# ============================================================= 5 折稳定（e 候选）
make_bio_stab <- function() {
  d <- rdp(RVDIR, "fivefold_stability_by_readout.tsv")
  d <- d[d$test_dataset == "all", ]
  for (c in c("median_spearman", "sd_spearman", "min_spearman", "max_spearman"))
    d[[c]] <- as.numeric(d[[c]])
  d <- d[is.finite(d$median_spearman), ]
  d <- d[order(d$median_spearman), ]
  d$tl <- fig3_short(d$target_id, 20); d$ord <- factor(seq_len(nrow(d)))
  ggplot2::ggplot(d, ggplot2::aes(y = ord)) +
    ggplot2::geom_vline(xintercept = 0, color = COL_ZERO, linewidth = 0.3, linetype = "dashed") +
    ggplot2::geom_linerange(ggplot2::aes(xmin = min_spearman, xmax = max_spearman),
                            color = "#9FB6BC", linewidth = 0.5) +
    ggplot2::geom_point(ggplot2::aes(x = median_spearman, fill = sd_spearman),
                        shape = 21, color = "black", size = 1.7, stroke = 0.25) +
    ggplot2::scale_fill_gradientn(colours = c("#1F5A66", "#4198AC", "#BFDFD2", "#ECB66C", "#ED8D5A"),
                                  name = "fold SD",
                                  guide = ggplot2::guide_colorbar(barwidth = ggplot2::unit(18, "mm"),
                                                                  barheight = ggplot2::unit(2, "mm"))) +
    ggplot2::scale_y_discrete(labels = d$tl, expand = ggplot2::expansion(add = 0.7)) +
    ggplot2::scale_x_continuous(breaks = seq(0, 0.5, 0.1)) +
    ggplot2::labs(x = "Per-readout Spearman ρ (5-fold CV)", y = NULL,
                  title = "5-fold cross-validation stability",
                  subtitle = "Point = median across folds; bar = min–max; colour = fold SD") +
    theme_fig3_big() +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 6.4),
                   axis.ticks.y = ggplot2::element_blank(), legend.position = "bottom")
}

# ============================================================= 注意力负对照（e 候选）
make_bio_neg <- function() {
  d <- rdp(RVDIR, "random_control_attention_contrast.tsv")
  d <- d[d$dataset == "all", ]
  d$max_biological_attention <- as.numeric(d$max_biological_attention)
  d$random_control_attention <- as.numeric(d$random_control_attention)
  s <- rdp(RVDIR, "random_control_attention_summary.tsv"); sa <- s[s$dataset == "all", ]
  med <- as.numeric(sa$median_delta)
  long <- rbind(data.frame(target = d$target_id, x = 1, v = d$max_biological_attention),
                data.frame(target = d$target_id, x = 2, v = d$random_control_attention))
  npos <- sum(d$max_biological_attention > d$random_control_attention); ntot <- nrow(d)
  lab <- paste0("atop(Delta[median]=='", sprintf("%+.3f", med),
                "', atop(italic(P)=='7.5'%*%10^-11, 'biol > control: ", npos, "/", ntot, "'))")
  ggplot2::ggplot() +
    ggplot2::geom_line(data = long, ggplot2::aes(x = x, y = v, group = target),
                       color = "#51999F", linewidth = 0.3, alpha = 0.5) +
    ggplot2::geom_boxplot(data = long, ggplot2::aes(x = x, y = v, group = x),
                          width = 0.22, outlier.shape = NA, fill = NA, color = "#333333", linewidth = 0.4) +
    ggplot2::annotate("text", x = 1.5, y = max(long$v) + 0.005, label = lab, parse = TRUE,
                      hjust = 0.5, vjust = 1, size = 6 / PT, family = "Arial", color = COL_TEXT) +
    ggplot2::scale_x_continuous(breaks = c(1, 2), labels = c("max\nbiological", "random\ncontrol"),
                                limits = c(0.6, 2.4)) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0.03, 0.16))) +
    ggplot2::labs(x = NULL, y = "Mean attention",
                  title = "Attention negative control",
                  subtitle = "56 readouts; biological pathway token vs shuffled random-control token") +
    theme_fig3_big() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(size = 7.5, lineheight = 0.9))
}

# ============================================================= c（合并版·横向）：全队列外部验证，13 读出一行、组间分隔、按通路
make_bio_c_merged <- function() {
  d <- rd("fig3_biology_external_by_readout.tsv")
  d$spearman <- as.numeric(d$spearman)
  d <- d[d$spearman >= 0.30, ]                              # 主图只放预测得好的（全集见 Supp）
  d$fam <- factor(d$family_display, levels = FAM_ORDER)
  coh_ord <- c("GSE300551", "HeLa", "Vivo-Th17")
  d <- d[d$cohort %in% coh_ord, ]
  d$coh <- factor(d$cohort, levels = coh_ord)
  d <- d[order(d$coh, -d$spearman), ]                       # 队列顺序 + 队列内 ρ 降序
  d$xf <- factor(d$target_display, levels = d$target_display)  # 13 个读出名互不重复
  n_by <- table(d$coh)[coh_ord]
  ends <- cumsum(as.integer(n_by))                          # 各组右端索引
  sep  <- head(ends, -1) + 0.5                              # 组间分隔位置
  cen  <- (c(0, head(ends, -1)) + ends) / 2 + 0.5           # 各组中心
  clab <- data.frame(x = cen, y = 0.80,
                     lab = c("GSE300551\n(icCITE, human T)", "SIGNAL-seq HeLa\n(human)", "Vivo-seq Th17\n(mouse)"))

  ggplot2::ggplot(d, ggplot2::aes(x = xf, y = spearman, fill = fam)) +
    ggplot2::geom_vline(xintercept = sep, linetype = "dotted", color = "#9A9A9A", linewidth = 0.35) +
    ggplot2::geom_col(width = 0.72, color = "black", linewidth = 0.25) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.2f", spearman)),
                       vjust = -0.45, size = 6 / PT, family = "Arial", color = COL_SUB) +
    ggplot2::geom_text(data = clab, ggplot2::aes(x = x, y = y, label = lab), inherit.aes = FALSE,
                       size = 6.2 / PT, family = "Arial", color = "#555555", lineheight = 0.9, vjust = 1) +
    ggplot2::scale_fill_manual(values = FAM_FILL, breaks = FAM_ORDER, drop = TRUE, name = NULL) +
    ggplot2::scale_y_continuous(limits = c(0, 0.84), breaks = seq(0, 0.6, 0.2), expand = c(0, 0)) +
    ggplot2::scale_x_discrete(expand = ggplot2::expansion(add = c(0.7, 1.4))) +
    ggplot2::labs(x = NULL, y = "Single-cell Spearman ρ",
                  title = "Single-cell external validation across cohorts",
                  subtitle = "Well-predicted readouts (ρ ≥ 0.30) by pathway; full set in Supplementary") +
    theme_fig3_big() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(size = 7, angle = 45, hjust = 1),
                   axis.ticks.x = ggplot2::element_blank(), legend.position = "bottom") +
    ggplot2::guides(fill = ggplot2::guide_legend(nrow = 2))
}

# ============================================================= UMAP 空间一致性（4 个已验证读数 pred|obs 网格）
make_bio_umap <- function() {
  u <- rd("fig3_hela_umap_4readouts.tsv")                  # UMAP 坐标 + 4 读数 pred/obs（合并表）
  ro <- c(CTNND1_T310 = "CTNND1 T310 (ρ=0.63)", MAP2K4_S257 = "MAP2K4 S257 (ρ=0.50)",
          NDRG1_T346 = "NDRG1 T346 (ρ=0.50)", PDPK1_S241 = "PDPK1 S241 (ρ=0.48)")
  u <- u[u$target_id %in% names(ro), ]
  for (c in c("predicted", "observed", "umap1", "umap2")) u[[c]] <- as.numeric(u[[c]])
  ncell <- length(unique(u$cell_id))
  L <- rbind(
    data.frame(umap1 = u$umap1, umap2 = u$umap2, tid = u$target_id, panel = "Predicted", val = u$predicted),
    data.frame(umap1 = u$umap1, umap2 = u$umap2, tid = u$target_id, panel = "Observed",  val = u$observed))
  L$panel <- factor(L$panel, levels = c("Predicted", "Observed"))
  L$rd_   <- factor(ro[L$tid], levels = unname(ro))
  L$valz  <- ave(L$val, L$tid, L$panel, FUN = function(x) as.numeric(scale(x)))  # 每(读数,面板)内 z 化
  L$valz  <- pmin(pmax(L$valz, -2.5), 2.5)

  ggplot2::ggplot(L[order(abs(L$valz)), ], ggplot2::aes(umap1, umap2, color = valz)) +
    ggplot2::geom_point(size = 0.95, alpha = 0.9, stroke = 0) +
    ggplot2::scale_color_gradient2(low = "#2166AC", mid = "#E3E3E3", high = "#B2182B",
                                   midpoint = 0, limits = c(-2.5, 2.5), name = "phospho (z)",
                                   guide = ggplot2::guide_colorbar(barwidth = ggplot2::unit(18, "mm"),
                                                                   barheight = ggplot2::unit(2, "mm"))) +
    ggplot2::facet_grid(panel ~ rd_) +
    ggplot2::coord_equal(clip = "off") +                      # 保持 UMAP 真实形状（不变形）
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = 0.02)) +
    ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = 0.02)) +
    ggplot2::labs(x = "UMAP1", y = "UMAP2",
                  title = sprintf("Spatial coherence on the HeLa UMAP — antibody-free transfer sites (n = %d cells)", ncell)) +
    theme_fig3_big() +
    ggplot2::theme(legend.position = "bottom",
                   axis.text.x = ggplot2::element_blank(), axis.text.y = ggplot2::element_blank(),
                   axis.ticks = ggplot2::element_blank(),
                   axis.title = ggplot2::element_text(size = 7, color = COL_SUB),
                   panel.grid = ggplot2::element_blank(),
                   panel.spacing = ggplot2::unit(2.5, "pt"),
                   plot.title = ggplot2::element_text(size = 8.5),
                   strip.text = ggplot2::element_text(size = 7.5),
                   plot.margin = ggplot2::margin(2, 3, 2, 2))
}

# ============================================================= render
# 主图 Fig 3 = a 架构(Illustrator) + b benchmark + c GSE 验证 + d 跨平台 + e 单细胞散点 + f 覆盖能力。
# Extended Data = 消融 / 校准 / 5 折稳定 / 注意力（路由 + 负对照）。第 5 列 = 输出目录。
message("Rendering Fig 3 (main → fig3/, ED → fig3/extended_data/)")
reg <- list(
  # —— 主图 a–e（a 架构在 Illustrator）。尺寸=180mm 宽 4 行版式：a整宽 / b+c并排 / d整宽 / e整宽 ——
  list(make_bio_bench,      "fig3_panel_b_benchmark",            78,  80, OUTDIR),
  list(make_bio_c_merged,   "fig3_panel_c_validation_all",   109.02, 49.42, OUTDIR),
  list(make_bio_winscatter, "fig3_panel_d_winscatter",         178,  58, OUTDIR),
  list(make_bio_umap,       "fig3_panel_e_umap_coherence",   206.77, 103.4, OUTDIR),
  list(make_bio_g_small,    "fig3_panel_f_graph_ablation",    57.89, 75.54, OUTDIR),
  list(make_bio_e,          "EDfig_single_cell_scatter",       178,  56, EDDIR),
  # —— Extended Data ——
  list(make_bio_g,     "EDfig_site_graph_ablation",         124,  84, EDDIR),
  list(make_bio_h,     "EDfig_calibration",                 108,  74, EDDIR),
  list(make_bio_stab,  "EDfig_5fold_stability",              92, 124, EDDIR),
  list(make_bio_neg,   "EDfig_attention_negcontrol",         92,  88, EDDIR)
)
args <- commandArgs(trailingOnly = TRUE)
for (it in reg) {
  if (length(args) == 0 || any(vapply(args, function(a) grepl(a, it[[2]], fixed = TRUE), logical(1)))) {
    ok <- tryCatch({ save_fig(it[[1]](), it[[2]], it[[3]], it[[4]], it[[5]]); TRUE },
                   error = function(e) { message("  FAIL ", it[[2]], ": ", conditionMessage(e)); FALSE })
  }
}
message("Done.")
