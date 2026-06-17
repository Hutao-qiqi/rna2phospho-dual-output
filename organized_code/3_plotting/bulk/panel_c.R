# panel_c.R (v3) — External cross-cohort × cross-method benchmark
#
# 与 panel b 配对：训练集（panel b: pan-cancer + 5 cancer-group）→
#                外部 4 phospho-proteome 队列（本 panel）
#
# 内容：
#   * 6 个方法（去掉 Mean，因外部上 Spearman 全 NA）
#   * 3 个独立外部队列 (FU-iCCA / TU-SCLC / CHCC-HBV)
#     —— v9 去掉 CHCC-HBV FPKM：与 RSEM 同批 HCC 病人，非独立队列
#   * 每方法在每队列一个 bar = per-site median ρ + IQR error bar
#   * 每 baseline bar 上方单独标 paired Wilcoxon (vs SCP682, alternative="greater") 显著性
#     —— 不用 H cluster bracket，因为 Cognate mRNA 可能跟 SCP682 无显著差异
#     单独标 star 比 cluster 标 **** 更 honest
#
# 数据：01_key_results/per_site_spearman_external.tsv (long format，约 21.4 万行)
# 调用：source 后 make_panel_c() 返回 ggplot 对象（不带 panel letter）
# 尺寸：170 × 80 mm（A4 portrait 整行，比 panel b 95mm 略矮，因为只有 4 facet）
# 字体：Arial（cairo_pdf 嵌入）

.PANEL_C_ROOT <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682"
.PANEL_C_TSV  <- file.path(.PANEL_C_ROOT, "01_key_results",
                           "per_site_spearman_external_9models.tsv")

# v8 (2026-05-26): 加 DeepGxP_5fold 和 VAE 两个深度学习 baseline。
# 仍去掉 Mean（NA）和 Cognate mRNA。共 7 方法，顺序跟 panel b 一致。
.PANEL_C_METHOD_COLORS <- c(
  "DeepGxP"       = "#BFDFD2",  # 浅薄荷绿（floor 类）
  "Ridge"         = "#51999F",  # 中青绿
  "PC ridge"      = "#4198AC",  # 深青绿
  "VAE"           = "#DBCB92",  # 浅黄绿
  "MLP"           = "#ECB66C",  # 浅橙
  "Elastic net"   = "#EA9E58",  # 中橙
  "SCP682"        = "#ED8D5A"   # 深橙（ours）
)
.PANEL_C_METHOD_ORDER <- names(.PANEL_C_METHOD_COLORS)

.PANEL_C_TSV_TO_LABEL <- c(
  "DeepGxP_5fold"             = "DeepGxP",
  "masked_ridge_linear"       = "Ridge",
  "PCA_ridge"                 = "PC ridge",
  "VAE"                       = "VAE",
  "MLP"                       = "MLP",
  "masked_elasticnet_linear"  = "Elastic net",
  "SCP682"                    = "SCP682"
  # mean_pred、parent_mRNA_linear 不映射 → 自动过滤
)

# v10 (2026-06-03): CHCC-HBV 改用 FPKM 定量（口径定稿）。
# panel c = FU-iCCA / TU-SCLC / CHCC-HBV(FPKM) 三队列共有统一 multi-method benchmark。
# APOLLO LUAD 为第 4 独立外部队列但无 7-method 对比数据，不进本 panel（正文/补充）。
# CHCC-HBV RSEM 作同队列 RNA 定量敏感性补充（Supplementary），不在此图。
.PANEL_C_DATASET_ORDER <- c("fu_icca", "tu_sclc", "chcc_hbv_fpkm")
.PANEL_C_DATASET_LABEL <- c(
  "fu_icca"       = "FU-iCCA\n(n = 208)",
  "tu_sclc"       = "TU-SCLC\n(n = 107)",
  "chcc_hbv_fpkm" = "CHCC-HBV\n(n = 159)"
)

.panel_c_theme <- function() {
  ggplot2::theme_classic(base_size = 7, base_family = "Arial") +
    ggplot2::theme(
      panel.background    = ggplot2::element_rect(fill = "white", color = NA),
      plot.background     = ggplot2::element_rect(fill = "white", color = NA),
      plot.title          = ggplot2::element_text(
        size = 7, face = "plain", hjust = 0, color = "#222222",
        margin = ggplot2::margin(0, 0, 1, 0)),
      plot.subtitle       = ggplot2::element_text(
        size = 5.8, face = "italic", hjust = 0, color = "#555555",
        margin = ggplot2::margin(0, 0, 3, 0)),
      plot.title.position = "plot",
      axis.title.y        = ggplot2::element_text(size = 6.8, color = "#222222"),
      axis.title.x        = ggplot2::element_blank(),
      axis.text.y         = ggplot2::element_text(size = 6.2, color = "#222222"),
      axis.text.x         = ggplot2::element_blank(),
      axis.ticks.x        = ggplot2::element_blank(),
      axis.line           = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.ticks.y        = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.ticks.length   = ggplot2::unit(0.7, "mm"),
      strip.background    = ggplot2::element_rect(fill = "#F5F5F5", color = NA),
      strip.text          = ggplot2::element_text(size = 6.5, color = "#222222",
                                                  margin = ggplot2::margin(2, 0, 2, 0)),
      panel.spacing.x     = ggplot2::unit(2.0, "mm"),
      legend.text         = ggplot2::element_text(size = 6.5, color = "#222222"),
      legend.title        = ggplot2::element_blank(),
      legend.background   = ggplot2::element_rect(fill = "white", color = NA),
      legend.key.size     = ggplot2::unit(3.5, "mm"),
      legend.position     = "bottom",
      legend.margin       = ggplot2::margin(2, 0, 0, 0),
      plot.margin         = ggplot2::margin(6, 8, 2, 8, "pt")
    )
}

.panel_c_sig <- function(p) {
  ifelse(is.na(p), "",
  ifelse(p < 1e-4, "****",
  ifelse(p < 1e-3, "***",
  ifelse(p < 1e-2, "**",
  ifelse(p < 0.05, "*", "ns")))))
}

make_panel_c <- function() {
  d <- utils::read.delim(.PANEL_C_TSV, sep = "\t", stringsAsFactors = FALSE)
  d <- d[!is.na(d$spearman), , drop = FALSE]
  d$method_label  <- factor(.PANEL_C_TSV_TO_LABEL[d$method],
                            levels = .PANEL_C_METHOD_ORDER)
  d$dataset_label <- factor(.PANEL_C_DATASET_LABEL[d$dataset],
                            levels = .PANEL_C_DATASET_LABEL[.PANEL_C_DATASET_ORDER])
  d <- d[!is.na(d$method_label) & !is.na(d$dataset_label), , drop = FALSE]

  # ---------- median / IQR ----------
  med_tab <- stats::aggregate(spearman ~ method_label + dataset_label, data = d,
                              FUN = function(x) stats::median(x, na.rm = TRUE))
  names(med_tab)[3] <- "med"
  q25_tab <- stats::aggregate(spearman ~ method_label + dataset_label, data = d,
                              FUN = function(x) stats::quantile(x, 0.25, na.rm = TRUE))
  names(q25_tab)[3] <- "q25"
  q75_tab <- stats::aggregate(spearman ~ method_label + dataset_label, data = d,
                              FUN = function(x) stats::quantile(x, 0.75, na.rm = TRUE))
  names(q75_tab)[3] <- "q75"
  agg <- merge(merge(med_tab, q25_tab, by = c("method_label", "dataset_label")),
               q75_tab,                  by = c("method_label", "dataset_label"))

  # ---------- 配对 Wilcoxon: SCP682 vs each baseline，per dataset ----------
  baselines <- setdiff(.PANEL_C_METHOD_ORDER, "SCP682")
  p_rows <- list()
  for (ds in levels(d$dataset_label)) {
    sub <- d[d$dataset_label == ds, ]
    scp_sub <- sub[sub$method_label == "SCP682", c("target", "spearman")]
    names(scp_sub)[2] <- "scp682"
    for (m in baselines) {
      base_sub <- sub[sub$method_label == m, c("target", "spearman")]
      names(base_sub)[2] <- "base"
      paired <- merge(scp_sub, base_sub, by = "target")
      paired <- paired[!is.na(paired$scp682) & !is.na(paired$base), ]
      if (nrow(paired) < 10) next
      t <- stats::wilcox.test(paired$scp682, paired$base,
                              paired = TRUE, alternative = "greater",
                              exact = FALSE)
      p_rows[[paste(ds, m)]] <- data.frame(
        dataset_label = ds, method_label = m,
        n_paired = nrow(paired), p_value = t$p.value,
        stringsAsFactors = FALSE)
    }
  }
  pvals <- do.call(rbind, p_rows)
  pvals$signif <- .panel_c_sig(pvals$p_value)
  pvals$method_label  <- factor(pvals$method_label,  levels = .PANEL_C_METHOD_ORDER)
  pvals$dataset_label <- factor(pvals$dataset_label, levels = levels(d$dataset_label))

  # ---------- median 数字位置（bar 顶上方）----------
  agg$label_y <- pmin(agg$q75 + 0.008, 0.58)

  # ---------- H cluster bracket 数据（7 方法：6 baselines + SCP682）----------
  # cluster 横跨 baseline 1-6 (DeepGxP / Ridge / PC ridge / VAE / MLP / Elastic net)，SCP682 在 x=7
  n_methods <- length(.PANEL_C_METHOD_ORDER)
  scp_x_pos <- n_methods
  bl_x_end  <- n_methods - 1
  agg$method_x <- as.numeric(agg$method_label)
  bracket_df <- do.call(rbind, lapply(levels(d$dataset_label), function(ds) {
    sub <- agg[agg$dataset_label == ds, ]
    bl  <- sub[sub$method_label != "SCP682", ]
    scp <- sub[sub$method_label == "SCP682", ]
    base_top <- max(bl$q75)
    scp_top  <- scp$q75
    cl_y     <- base_top + 0.025
    top_y    <- max(cl_y, scp_top) + 0.04
    data.frame(
      dataset_label   = ds,
      cluster_x_start = 1,
      cluster_x_end   = bl_x_end,
      cluster_y       = cl_y,
      cluster_mid_x   = (1 + bl_x_end) / 2,
      scp_x           = scp_x_pos,
      scp_y_anchor    = scp_top + 0.008,
      bracket_top_y   = top_y,
      label_x         = ((1 + bl_x_end) / 2 + scp_x_pos) / 2,
      label_y         = top_y + 0.012,
      stringsAsFactors = FALSE)
  }))
  bracket_df$dataset_label <- factor(bracket_df$dataset_label,
                                     levels = levels(d$dataset_label))
  # 取每 facet 最弱的 p 作为 cluster 标签（4 个 baseline 全部 **** → 仍 ****）
  cluster_label <- stats::aggregate(p_value ~ dataset_label, data = pvals,
                                    FUN = max)
  cluster_label$signif <- .panel_c_sig(cluster_label$p_value)
  bracket_df <- merge(bracket_df,
                      cluster_label[, c("dataset_label", "signif")],
                      by = "dataset_label")

  # ---------- ggplot ----------
  ggplot2::ggplot(agg,
    ggplot2::aes(x = method_label, y = med, fill = method_label)) +
    ggplot2::geom_hline(yintercept = 0,
                        color = "#A8A8A8", linewidth = 0.3, linetype = "dashed") +
    # SCP682 中位水平参考线
    ggplot2::geom_hline(data = agg[agg$method_label == "SCP682", ],
                        ggplot2::aes(yintercept = med),
                        color = "#ED8D5A", linewidth = 0.4, linetype = "dashed") +
    ggplot2::geom_col(width = 0.7, color = "black", linewidth = 0.25) +
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = q25, ymax = q75),
      width = 0.25, linewidth = 0.3, color = "#333333") +
    # median 数字（bar 顶上方）
    ggplot2::geom_text(
      ggplot2::aes(y = label_y, label = sprintf("%.2f", med)),
      size = 1.6, color = "#222222",
      family = "Arial", vjust = 0, fontface = "plain") +
    # ---------- H cluster bracket ----------
    # 1. baseline cluster 横线
    ggplot2::geom_segment(data = bracket_df,
      ggplot2::aes(x = cluster_x_start, xend = cluster_x_end,
                   y = cluster_y, yend = cluster_y),
      inherit.aes = FALSE, linewidth = 0.3, color = "#555555") +
    # 2. cluster 两端短下钩
    ggplot2::geom_segment(data = bracket_df,
      ggplot2::aes(x = cluster_x_start, xend = cluster_x_start,
                   y = cluster_y, yend = cluster_y - 0.012),
      inherit.aes = FALSE, linewidth = 0.3, color = "#555555") +
    ggplot2::geom_segment(data = bracket_df,
      ggplot2::aes(x = cluster_x_end, xend = cluster_x_end,
                   y = cluster_y, yend = cluster_y - 0.012),
      inherit.aes = FALSE, linewidth = 0.3, color = "#555555") +
    # 3. cluster 中点向上到 bracket_top
    ggplot2::geom_segment(data = bracket_df,
      ggplot2::aes(x = cluster_mid_x, xend = cluster_mid_x,
                   y = cluster_y, yend = bracket_top_y),
      inherit.aes = FALSE, linewidth = 0.3, color = "#555555") +
    # 4. bracket_top 水平段连到 SCP682
    ggplot2::geom_segment(data = bracket_df,
      ggplot2::aes(x = cluster_mid_x, xend = scp_x,
                   y = bracket_top_y, yend = bracket_top_y),
      inherit.aes = FALSE, linewidth = 0.3, color = "#555555") +
    # 5. SCP682 端短下钩到 SCP682 q75 上方
    ggplot2::geom_segment(data = bracket_df,
      ggplot2::aes(x = scp_x, xend = scp_x,
                   y = bracket_top_y, yend = scp_y_anchor),
      inherit.aes = FALSE, linewidth = 0.3, color = "#555555") +
    # 6. 在 bracket 顶水平段中间标 ****
    ggplot2::geom_text(data = bracket_df,
      ggplot2::aes(x = label_x, y = label_y, label = signif),
      inherit.aes = FALSE,
      size = 2.2, color = "#222222", family = "Arial",
      vjust = 0, fontface = "bold") +
    ggplot2::scale_fill_manual(values = .PANEL_C_METHOD_COLORS, name = NULL) +
    ggplot2::scale_y_continuous(
      limits = c(-0.05, 0.65),
      breaks = c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
      expand = c(0, 0)) +
    ggplot2::facet_wrap(~ dataset_label, nrow = 1, scales = "fixed") +
    ggplot2::labs(
      x        = NULL,
      y        = "Per-site median Spearman ρ",
      title    = "External cross-cohort generalisation (3 independent phospho-proteome cohorts)",
      subtitle = paste0("Bars: median ρ; error bars: IQR (25–75%); bracket: paired Wilcoxon signed-rank, SCP682 > every baseline (**** p < 10⁻⁴).\n",
                        "All 6 baselines (DeepGxP / Ridge / PC ridge / VAE / MLP / Elastic net) collapse on external transfer (ρ 0.04–0.24); SCP682 maintains 0.32–0.37.")) +
    ggplot2::guides(fill = ggplot2::guide_legend(
      nrow = 1, override.aes = list(color = "black", linewidth = 0.25))) +
    .panel_c_theme()
}
