# panel_b.R (v6) — Cross-tissue ML benchmark, bar plot + H-shaped significance
#
# 与 v5 的差别：
#   * boxplot → bar plot (high = median; error bar = IQR)
#   * 不显示 per-site 散点
#   * median 数字标在 bar 顶外侧（字号略小于 bar 宽度）
#   * 显著性改为 H-shaped bracket：
#       - 横向 bracket 横跨 6 个 baseline（"baseline cluster"）
#       - 从 cluster 中点向 SCP682 抬一档接到 SCP682 顶
#       - 在 cluster ↔ SCP682 链上标 ****
#     这种形式直观显示"所有 baseline 全部显著弱于 SCP682"
#   * 配色仍用 v5 的 nature-skills NMI Pastel + Primary 调色板

.PANEL_B_ROOT <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682"
.PANEL_B_TSV  <- file.path(.PANEL_B_ROOT, "01_key_results",
                           "per_site_spearman_with_deep_learning.tsv")

# v8 (2026-05-26): 加 DeepGxP_5fold 和 VAE 两个深度学习 baseline，共 9 个方法。
# 顺序按 CPTAC_all median 升序，配色冷→暖渐变。
.PANEL_B_METHOD_COLORS <- c(
  "Mean"          = "#D8D8D8",  # 灰（floor）
  "DeepGxP"       = "#BFDFD2",  # 浅薄荷绿（DeepGxP_5fold，floor 类）
  "Cognate mRNA"  = "#7BC0CD",  # 浅青蓝
  "Ridge"         = "#51999F",  # 中青绿
  "PC ridge"      = "#4198AC",  # 深青绿
  "VAE"           = "#DBCB92",  # 浅黄绿（cool→warm 过渡）
  "MLP"           = "#ECB66C",  # 浅橙
  "Elastic net"   = "#EA9E58",  # 中橙
  "SCP682"        = "#ED8D5A"   # 深橙（ours，最暖强调）
)
.PANEL_B_METHOD_ORDER <- names(.PANEL_B_METHOD_COLORS)

.PANEL_B_TSV_TO_LABEL <- c(
  "mean_pred"                 = "Mean",
  "DeepGxP_5fold"             = "DeepGxP",
  "parent_mRNA_linear"        = "Cognate mRNA",
  "masked_ridge_linear"       = "Ridge",
  "PCA_ridge"                 = "PC ridge",
  "VAE"                       = "VAE",
  "MLP"                       = "MLP",
  "masked_elasticnet_linear"  = "Elastic net",
  "SCP682"                    = "SCP682"
)

.PANEL_B_DATASET_ORDER <- c(
  "CPTAC_all", "CPTAC_kidney", "CPTAC_pancreas_HN",
  "CPTAC_gynecologic", "CPTAC_gi_hepato", "CPTAC_lung"
)
.PANEL_B_DATASET_LABEL <- c(
  "CPTAC_all"         = "Pan-cancer\n(n = 1,431)",
  "CPTAC_kidney"      = "Kidney\n(n = 132)",
  "CPTAC_pancreas_HN" = "Pancreas / H&N\n(n = 172)",
  "CPTAC_gynecologic" = "Gynaecological\n(n = 167)",
  "CPTAC_gi_hepato"   = "GI / hepatobiliary\n(n = 79)",
  "CPTAC_lung"        = "Lung\n(n = 284)"
)

.panel_b_theme <- function() {
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

.panel_b_sig <- function(p) {
  ifelse(is.na(p), "",
  ifelse(p < 1e-4, "****",
  ifelse(p < 1e-3, "***",
  ifelse(p < 1e-2, "**",
  ifelse(p < 0.05, "*", "ns")))))
}

make_panel_b <- function() {
  d <- utils::read.delim(.PANEL_B_TSV, sep = "\t", stringsAsFactors = FALSE)
  d <- d[!is.na(d$spearman), , drop = FALSE]
  d$method_label  <- factor(.PANEL_B_TSV_TO_LABEL[d$method],
                            levels = .PANEL_B_METHOD_ORDER)
  d$dataset_label <- factor(.PANEL_B_DATASET_LABEL[d$dataset],
                            levels = .PANEL_B_DATASET_LABEL[.PANEL_B_DATASET_ORDER])
  d <- d[!is.na(d$method_label) & !is.na(d$dataset_label), , drop = FALSE]

  # ---------- 聚合 median / IQR ----------
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

  # ---------- 配对 Wilcoxon: SCP682 vs each baseline ----------
  baselines <- setdiff(.PANEL_B_METHOD_ORDER, "SCP682")
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
  pvals$signif <- .panel_b_sig(pvals$p_value)
  # 取每个 facet 内 6 个 baseline 的最差 p 作为 cluster 标签（实际全部 < 1e-4 → ****）
  cluster_label <- stats::aggregate(p_value ~ dataset_label, data = pvals,
                                    FUN = max)
  cluster_label$signif <- .panel_b_sig(cluster_label$p_value)
  cluster_label$dataset_label <- factor(cluster_label$dataset_label,
                                        levels = levels(d$dataset_label))

  # ---------- H 形 bracket 数据 ----------
  # 每个 facet 内：
  #   baseline_cluster_y = baselines 中最大 q75 + 0.04
  #   bracket_top_y      = max(baseline_cluster_y, SCP682 q75) + 0.06
  #   SCP682 顶 = SCP682 q75 + 0.01
  agg$method_x <- as.numeric(agg$method_label)
  bracket_df <- do.call(rbind, lapply(levels(d$dataset_label), function(ds) {
    sub <- agg[agg$dataset_label == ds, ]
    bl  <- sub[sub$method_label != "SCP682", ]
    scp <- sub[sub$method_label == "SCP682", ]
    base_top <- max(bl$q75)
    scp_top  <- scp$q75
    cl_y     <- base_top + 0.04
    top_y    <- max(cl_y, scp_top) + 0.06
    data.frame(
      dataset_label   = ds,
      cluster_x_start = 1,
      cluster_x_end   = 6,
      cluster_y       = cl_y,
      cluster_mid_x   = 3.5,
      scp_x           = 7,
      scp_y_anchor    = scp_top + 0.01,
      bracket_top_y   = top_y,
      label_x         = (3.5 + 7) / 2,
      label_y         = top_y + 0.015,
      stringsAsFactors = FALSE)
  }))
  bracket_df$dataset_label <- factor(bracket_df$dataset_label,
                                     levels = levels(d$dataset_label))
  bracket_df <- merge(bracket_df,
                      cluster_label[, c("dataset_label", "signif")],
                      by = "dataset_label")

  # ---------- median 数字标注位置（bar 顶上方 0.02） ----------
  agg$label_y <- pmin(agg$q75 + 0.02, 0.97)
  # bar 顶下方负值的特殊处理：med < 0 时数字标在 0 上方
  agg$label_y <- ifelse(agg$med < 0, 0.02, agg$label_y)
  # label 字色：所有数字都标在 bar 外，统一深字
  agg$label_color <- "#222222"

  # ---------- ggplot ----------
  ggplot2::ggplot(agg,
    ggplot2::aes(x = method_label, y = med, fill = method_label)) +
    ggplot2::geom_hline(yintercept = 0,
                        color = "#A8A8A8", linewidth = 0.3, linetype = "dashed") +
    # SCP682 中位水平参考线
    ggplot2::geom_hline(data = agg[agg$method_label == "SCP682", ],
                        ggplot2::aes(yintercept = med),
                        color = "#ED8D5A", linewidth = 0.4, linetype = "dashed") +
    # bar
    ggplot2::geom_col(width = 0.7, color = "black", linewidth = 0.25) +
    # IQR error bar
    ggplot2::geom_errorbar(
      ggplot2::aes(ymin = q25, ymax = q75),
      width = 0.25, linewidth = 0.3, color = "#333333") +
    # median 数字（缩到 1.6 pt 避免相邻数字粘连）
    ggplot2::geom_text(
      ggplot2::aes(y = label_y, label = sprintf("%.2f", med)),
      size = 1.6, color = "#222222",
      family = "Arial", vjust = 0, fontface = "plain") +
    # ---------- H bracket ----------
    # 1. baseline cluster 横线
    ggplot2::geom_segment(data = bracket_df,
      ggplot2::aes(x = cluster_x_start, xend = cluster_x_end,
                   y = cluster_y, yend = cluster_y),
      inherit.aes = FALSE, linewidth = 0.3, color = "#555555") +
    # 2. cluster 两端短下钩
    ggplot2::geom_segment(data = bracket_df,
      ggplot2::aes(x = cluster_x_start, xend = cluster_x_start,
                   y = cluster_y, yend = cluster_y - 0.015),
      inherit.aes = FALSE, linewidth = 0.3, color = "#555555") +
    ggplot2::geom_segment(data = bracket_df,
      ggplot2::aes(x = cluster_x_end, xend = cluster_x_end,
                   y = cluster_y, yend = cluster_y - 0.015),
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
    ggplot2::scale_fill_manual(values = .PANEL_B_METHOD_COLORS, name = NULL) +
    ggplot2::scale_y_continuous(
      limits = c(-0.15, 1.00),
      breaks = c(0, 0.2, 0.4, 0.6, 0.8, 1.0),
      expand = c(0, 0)) +
    ggplot2::facet_wrap(~ dataset_label, nrow = 1, scales = "fixed") +
    ggplot2::labs(
      x        = NULL,
      y        = "Per-site median Spearman ρ",
      title    = "Cross-tissue ML benchmark (CPTAC OOF, 5-fold CV)",
      subtitle = paste0("Bars: median ρ across all evaluable phosphosites; ",
                        "error bars: IQR (25–75%). ",
                        "Bracket: paired Wilcoxon signed-rank, SCP682 > every baseline ",
                        "(**** p < 10⁻⁴).")) +
    ggplot2::guides(fill = ggplot2::guide_legend(
      nrow = 1, override.aes = list(color = "black", linewidth = 0.25))) +
    .panel_b_theme()
}
