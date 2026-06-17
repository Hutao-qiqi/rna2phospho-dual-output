# panel_f.R — Fig 2 panel f：2,000 个位点 × 12 癌种 mfuzz-style heatmap (v7)
#
# v7 修订（v5 视觉 + cluster sidebar 加显著通路标签）：
#   • 复用 v5 的聚类语义：top/bottom 1000 各自 k-means k=3 on raw ρ → 6 cluster
#   • 聚类 + Hallmark 富集统一由 Python panel_f_v5_cluster_enrichment.py 算好
#   • 主体显示 RAW ρ，色阶 diverging 高对比红蓝 midpoint=0.547
#   • 左 cluster sidebar 用 anno_block：编号 + Hallmark top 通路简称（q<0.1 才显示）
#   • 顶部 cancer-type 色条 (Set3 12 色)
#   • 簇内按 pan-cancer ρ 降序排列
#
# 数据：
#   _scripts/panel_f_heatmap_matrix_cancer_type.tsv     (2,000 × 12 ρ)
#   _scripts/panel_f_v5_cluster_assignment.tsv          (target → cluster_id 1..6)
#   _scripts/panel_f_v5_cluster_enrichment.tsv          (cluster_id → top Hallmark hit)

.PF_ROOT <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682"
.PF_DIR  <- file.path(.PF_ROOT, "04_figure_source_data", "fig2", "_scripts")

.PF_MIDPOINT <- 0.5474
.PF_K        <- 6
.PF_Q_THR    <- 0.1   # cluster 富集 q 阈值，超过则显示 n.s.

# 癌种列顺序 + 显示标签
.PF_DS_ORDER <- c("BRCA", "CCRCC", "ccPRCC", "COAD", "GBM",
                  "HNSCC", "LSCC", "LUAD", "OV", "PDA", "STAD", "UCEC")
.PF_DS_LABEL <- c(
  "BRCA"   = "BRCA\n(n=191)",
  "CCRCC"  = "CCRCC\n(n=91)",
  "ccPRCC" = "ccPRCC\n(n=41)",
  "COAD"   = "COAD\n(n=94)",
  "GBM"    = "GBM\n(n=175)",
  "HNSCC"  = "HNSCC\n(n=86)",
  "LSCC"   = "LSCC\n(n=94)",
  "LUAD"   = "LUAD\n(n=190)",
  "OV"     = "OV\n(n=137)",
  "PDA"    = "PDA\n(n=86)",
  "STAD"   = "STAD\n(n=79)",
  "UCEC"   = "UCEC\n(n=167)"
)

# ---------------------------------------------------------------------------
.pf_load <- function() {
  mat <- utils::read.delim(file.path(.PF_DIR, "panel_f_heatmap_matrix_cancer_type.tsv"),
                           sep = "\t", stringsAsFactors = FALSE)
  ass <- utils::read.delim(file.path(.PF_DIR, "panel_f_v5_cluster_assignment.tsv"),
                           sep = "\t", stringsAsFactors = FALSE)
  enr <- utils::read.delim(file.path(.PF_DIR, "panel_f_v5_cluster_enrichment.tsv"),
                           sep = "\t", stringsAsFactors = FALSE)
  list(mat = mat, ass = ass, enr = enr)
}

# ---------------------------------------------------------------------------
make_panel_f <- function() {
  d <- .pf_load()
  mat <- d$mat
  ass <- d$ass
  enr <- d$enr

  # 合并 cluster_id 到 mat（按 target 对齐）
  mat <- merge(mat, ass[, c("target", "cluster_id")], by = "target", sort = FALSE)
  # 保留 row_order 原顺序 → 再按 cluster_id 升、cluster 内 pan-cancer ρ 降序排
  mat <- mat[order(mat$row_order), ]
  mat <- mat[order(mat$cluster_id, -mat$CPTAC_all), ]

  cluster_id <- mat$cluster_id
  direction  <- mat$direction

  # 数值矩阵
  m_raw <- as.matrix(mat[, .PF_DS_ORDER])
  rownames(m_raw) <- mat$target
  colnames(m_raw) <- .PF_DS_LABEL[.PF_DS_ORDER]

  # ---- cluster sidebar 标签：只显示编号（v8：详细通路信息移到右侧 bar chart）----
  cluster_labels <- as.character(seq_len(.PF_K))

  # ---- Cluster 色板 (Set2 6 色) ----
  cluster_colors <- setNames(
    RColorBrewer::brewer.pal(8, "Set2")[1:.PF_K],
    as.character(1:.PF_K))

  # ---- 顶部 cancer-type 色条 (Set3 12 色) ----
  ds_colors <- setNames(
    RColorBrewer::brewer.pal(12, "Set3"),
    .PF_DS_LABEL[.PF_DS_ORDER])
  ds_levels <- .PF_DS_LABEL[.PF_DS_ORDER]
  # Cancer type legend 被关掉：x 轴 label 已直接写 BRCA/CCRCC/... 12 个癌种名，
  # legend 完全冗余（v12 合成图发现它占 ~36mm，挤崩布局）。
  col_ha <- ComplexHeatmap::HeatmapAnnotation(
    `Cancer type` = factor(colnames(m_raw), levels = ds_levels),
    col = list(`Cancer type` = ds_colors),
    show_legend = FALSE,
    annotation_name_gp = grid::gpar(fontsize = 6, fontfamily = "Arial",
                                    col = "#222222"),
    annotation_name_side = "right",
    simple_anno_size = grid::unit(3, "mm"))

  # ---- 左侧 cluster 色条（anno_block 显示编号+简称）+ Direction（top/bottom）----
  direction_colors <- c("top" = "#CB181D", "bottom" = "#2171B5")
  row_ha <- ComplexHeatmap::rowAnnotation(
    Cluster = ComplexHeatmap::anno_block(
      which   = "row",
      gp      = grid::gpar(fill = cluster_colors, col = NA),
      labels  = cluster_labels,
      labels_gp = grid::gpar(fontsize = 8.0, fontfamily = "Arial",
                             col = "#222222", fontface = "bold"),
      labels_rot = 0,
      width = grid::unit(5, "mm")),
    Direction = factor(direction, levels = c("top", "bottom")),
    col = list(Direction = direction_colors),
    show_legend = c(FALSE, TRUE),
    annotation_name_gp = grid::gpar(fontsize = 6, fontfamily = "Arial",
                                    col = "#222222"),
    annotation_name_side = "bottom",
    simple_anno_size = grid::unit(3, "mm"),
    gap = grid::unit(1, "mm"),
    annotation_legend_param = list(
      Direction = list(
        title    = "Original\ngroup",
        title_gp = grid::gpar(fontsize = 6.0, fontfamily = "Arial",
                              col = "#222222", fontface = "bold"),
        labels_gp= grid::gpar(fontsize = 5.8, fontfamily = "Arial",
                              col = "#222222"),
        labels   = c("Top 1,000", "Bottom 1,000"),
        grid_height = grid::unit(2.4, "mm"),
        grid_width  = grid::unit(2.4, "mm"))))

  # ---- 主体色阶（高对比红蓝，midpoint=0.547）----
  col_fun <- circlize::colorRamp2(
    breaks = c(-0.4, 0, .PF_MIDPOINT, 0.75, 0.95),
    colors = c("#08306B", "#2171B5", "#FFFFFF", "#CB181D", "#67000D"))

  # ---- Heatmap ----
  ht <- ComplexHeatmap::Heatmap(
    m_raw,
    name = "per-site ρ",
    col  = col_fun,
    cluster_rows    = FALSE,
    cluster_columns = FALSE,
    show_row_names  = FALSE,
    show_column_names = TRUE,
    column_names_gp = grid::gpar(fontsize = 6, fontfamily = "Arial",
                                 col = "#222222", lineheight = 0.9),
    column_names_rot = 30,
    top_annotation   = col_ha,
    left_annotation  = row_ha,
    row_split        = factor(cluster_id, levels = 1:.PF_K),
    row_title        = NULL,             # cluster 编号已经塞进 anno_block label
    row_gap          = grid::unit(0.5, "mm"),
    border           = TRUE,
    na_col           = "#EEEEEE",
    use_raster       = FALSE,
    heatmap_legend_param = list(
      title       = "per-site ρ",
      title_gp    = grid::gpar(fontsize = 6.2, fontfamily = "Arial",
                               col = "#222222", fontface = "bold"),
      labels_gp   = grid::gpar(fontsize = 5.8, fontfamily = "Arial",
                               col = "#222222"),
      at          = c(-0.4, 0, .PF_MIDPOINT, 0.75, 0.95),
      labels      = c("-0.4", "0", "0.55 (median)", "0.75", "0.95"),
      grid_height = grid::unit(8, "mm"),
      grid_width  = grid::unit(2.4, "mm"),
      direction   = "vertical"))

  ht
}

# ---------------------------------------------------------------------------
# v9：右侧 Hallmark 条形图，每簇 3 条 bar 限制在 heatmap 该簇的行范围内
# 实现：使用数值 y 轴 row position 与 heatmap 对齐
# ---------------------------------------------------------------------------
make_panel_f_barchart <- function() {
  d <- .pf_load()
  enr <- d$enr
  ass <- d$ass

  # 计算每簇行数 + 起止 y 位置（对齐 heatmap 行号）
  cluster_sizes  <- vapply(1:.PF_K, function(c) sum(ass$cluster_id == c), integer(1))
  cluster_ends   <- cumsum(cluster_sizes)
  cluster_starts <- c(0, cluster_ends[-.PF_K])
  total_rows     <- sum(cluster_sizes)

  # 簇配色（与 heatmap 完全一致）
  cluster_colors <- setNames(
    RColorBrewer::brewer.pal(8, "Set2")[1:.PF_K],
    as.character(1:.PF_K))

  # 行顺序：cluster 1 上 → cluster 6 下；簇内按 rank 升序
  enr <- enr[order(enr$cluster_id, enr$rank), ]
  enr$neg_log10_q <- -log10(pmax(enr$q, 1e-15))

  # 每簇 3 条 bar 在簇内均匀分布（中心点）
  enr$y_pos <- with(enr, {
    cs <- cluster_starts[cluster_id]
    ch <- cluster_sizes[cluster_id]
    pos_frac <- (rank - 0.5) / 3   # rank1=1/6, rank2=1/2, rank3=5/6
    cs + pos_frac * ch
  })

  # bar 高度：每簇 1/4 行数（视觉密度跟簇大小适配）
  enr$bar_h <- cluster_sizes[enr$cluster_id] / 4
  # cluster 3 (n=173) 太小，最小 bar 高度兜底
  enr$bar_h <- pmax(enr$bar_h, 40)

  # y 轴 label：通路简称；重名加 cluster 后缀
  enr$y_label <- enr$short_label
  dup <- duplicated(enr$y_label) | duplicated(enr$y_label, fromLast = TRUE)
  enr$y_label[dup] <- sprintf("%s (c%d)", enr$y_label[dup], enr$cluster_id[dup])
  enr$cluster_factor <- factor(enr$cluster_id, levels = 1:.PF_K)

  # 显著性
  enr$sig_label <- sapply(enr$q, function(q) {
    if (q < 1e-10) "***"
    else if (q < 1e-3) "**"
    else if (q < 0.05) "*"
    else if (q < 0.1)  "+"
    else               "n.s."
  })

  # 簇间隔（visual line，y_pos = cluster_end）
  sep_y <- cluster_ends[-.PF_K] + 0.5

  ggplot2::ggplot(enr,
    ggplot2::aes(x = neg_log10_q, y = y_pos, fill = cluster_factor)) +
    # 簇之间画浅灰横线
    ggplot2::geom_hline(yintercept = sep_y,
                        color = "#CCCCCC", linewidth = 0.25) +
    # bar 用 geom_rect 控制每条高度
    ggplot2::geom_rect(
      ggplot2::aes(xmin = 0, xmax = neg_log10_q,
                   ymin = y_pos - bar_h/2, ymax = y_pos + bar_h/2,
                   fill = cluster_factor),
      color = NA) +
    # 右侧 sig + k + q（紧凑格式，省括号）
    ggplot2::geom_text(
      ggplot2::aes(label = sprintf("%s  k=%d  q=%s",
                                    sig_label, k, format(q, digits = 1, scientific = TRUE))),
      hjust = -0.05, family = "Arial", size = 1.9, color = "#222222") +
    ggplot2::scale_fill_manual(values = cluster_colors, guide = "none") +
    ggplot2::scale_x_continuous(
      limits = c(0, max(enr$neg_log10_q) * 1.7),
      breaks = c(0, 2, 4, 6, 8, 10),
      expand = ggplot2::expansion(mult = c(0, 0.01))) +
    ggplot2::scale_y_reverse(
      breaks = enr$y_pos,
      labels = enr$y_label,
      limits = c(total_rows + 1, -1),
      expand = c(0, 0)) +
    ggplot2::labs(
      x = expression(-log[10]~italic(q)),
      y = NULL,
      title = "Per-cluster Hallmark enrichment (top 3)") +
    ggplot2::theme_classic(base_size = 6.5, base_family = "Arial") +
    ggplot2::theme(
      panel.background    = ggplot2::element_rect(fill = "white", color = NA),
      plot.background     = ggplot2::element_rect(fill = "white", color = NA),
      plot.title          = ggplot2::element_text(
        size = 7, face = "plain", hjust = 0, color = "#222222",
        margin = ggplot2::margin(0, 0, 3, 0)),
      plot.title.position = "plot",
      axis.title.x        = ggplot2::element_text(size = 6.5, color = "#222222"),
      axis.text.x         = ggplot2::element_text(size = 6.0, color = "#222222"),
      axis.text.y         = ggplot2::element_text(size = 5.8, color = "#222222"),
      axis.line.x         = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.line.y         = ggplot2::element_blank(),
      axis.ticks.x        = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.ticks.y        = ggplot2::element_blank(),
      axis.ticks.length   = ggplot2::unit(0.6, "mm"),
      plot.margin         = ggplot2::margin(36, 6, 36, 4, "pt"))   # v10: 加 top/bot 边距压扁高度
}

# ---------------------------------------------------------------------------
# v10：connector ggplot —— 画梯形渐变把 heatmap 簇行连到 bar chart 簇组
# 输出可叠加到 cowplot canvas（透明背景）
# ---------------------------------------------------------------------------
make_panel_f_connector <- function(
  # v10b 实测调校：heatmap body + bar chart 实际 bar 边界的 canvas 坐标
  ht_panel_top   = 0.920,   # 簇 1 sidebar 顶 in canvas
  ht_panel_bot   = 0.075,   # 簇 6 sidebar 底 in canvas
  bar_panel_top  = 0.880,   # 对应 ROW 0 in bar chart
  bar_panel_bot  = 0.100,   # 对应 ROW total_rows in bar chart
  ht_x_right     = 0.555,
  bar_x_left     = 0.690,
  n_strip        = 24,
  alpha_left     = 0.55,
  alpha_right    = 0.00
) {
  d <- .pf_load()
  ass <- d$ass
  cluster_sizes  <- vapply(1:.PF_K, function(c) sum(ass$cluster_id == c), integer(1))
  cluster_ends   <- cumsum(cluster_sizes)
  cluster_starts <- c(0, cluster_ends[-.PF_K])
  total_rows     <- sum(cluster_sizes)

  cluster_colors <- setNames(
    RColorBrewer::brewer.pal(8, "Set2")[1:.PF_K],
    as.character(1:.PF_K))

  # 每簇 bar group 实际边界（rank 1 top + rank 3 bot），与 make_panel_f_barchart 算法一致
  compute_bar_edges <- function(i) {
    cs <- cluster_starts[i]
    cz <- cluster_sizes[i]
    bar_h <- max(cz / 4, 40)
    y_pos_rank1 <- cs + (1 - 0.5) / 3 * cz   # 簇内第 1 条 bar 中心
    y_pos_rank3 <- cs + (3 - 0.5) / 3 * cz   # 簇内第 3 条 bar 中心
    list(top = y_pos_rank1 - bar_h / 2,
         bot = y_pos_rank3 + bar_h / 2)
  }

  # 每簇生成 n_strip 个 sub-polygon，alpha 渐变
  strip_data <- do.call(rbind, lapply(1:.PF_K, function(i) {
    cs <- cluster_starts[i]; ce <- cluster_ends[i]
    be <- compute_bar_edges(i)

    # 左侧（heatmap）：簇 sidebar 上下沿（row cs ↔ ce）
    ht_y_top  <- ht_panel_top  - cs / total_rows * (ht_panel_top - ht_panel_bot)
    ht_y_bot  <- ht_panel_top  - ce / total_rows * (ht_panel_top - ht_panel_bot)
    # 右侧（bar chart）：实际 bar group 上下沿
    bar_y_top <- bar_panel_top - be$top / total_rows * (bar_panel_top - bar_panel_bot)
    bar_y_bot <- bar_panel_top - be$bot / total_rows * (bar_panel_top - bar_panel_bot)

    do.call(rbind, lapply(seq_len(n_strip), function(k) {
      f0 <- (k - 1) / n_strip
      f1 <- k / n_strip
      x0 <- ht_x_right + f0 * (bar_x_left - ht_x_right)
      x1 <- ht_x_right + f1 * (bar_x_left - ht_x_right)
      y_top_0 <- ht_y_top + f0 * (bar_y_top - ht_y_top)
      y_top_1 <- ht_y_top + f1 * (bar_y_top - ht_y_top)
      y_bot_0 <- ht_y_bot + f0 * (bar_y_bot - ht_y_bot)
      y_bot_1 <- ht_y_bot + f1 * (bar_y_bot - ht_y_bot)
      a       <- alpha_left + f0 * (alpha_right - alpha_left)
      data.frame(
        cluster = i, strip = k,
        poly_id = sprintf("c%d_s%d", i, k),
        x = c(x0, x1, x1, x0),
        y = c(y_top_0, y_top_1, y_bot_1, y_bot_0),
        fill  = cluster_colors[as.character(i)],
        alpha = a,
        stringsAsFactors = FALSE)
    }))
  }))

  # 灰线：每簇上沿和下沿（heatmap 簇边界 ↔ bar 组边界）
  line_data <- do.call(rbind, lapply(1:.PF_K, function(i) {
    cs <- cluster_starts[i]; ce <- cluster_ends[i]
    be <- compute_bar_edges(i)

    ht_y_top  <- ht_panel_top  - cs / total_rows * (ht_panel_top - ht_panel_bot)
    ht_y_bot  <- ht_panel_top  - ce / total_rows * (ht_panel_top - ht_panel_bot)
    bar_y_top <- bar_panel_top - be$top / total_rows * (bar_panel_top - bar_panel_bot)
    bar_y_bot <- bar_panel_top - be$bot / total_rows * (bar_panel_top - bar_panel_bot)
    data.frame(
      cluster = i,
      x    = c(ht_x_right, ht_x_right),
      xend = c(bar_x_left, bar_x_left),
      y    = c(ht_y_top, ht_y_bot),
      yend = c(bar_y_top, bar_y_bot),
      stringsAsFactors = FALSE)
  }))

  ggplot2::ggplot() +
    ggplot2::geom_polygon(
      data = strip_data,
      ggplot2::aes(x = x, y = y, group = poly_id, fill = fill, alpha = alpha),
      color = NA) +
    ggplot2::geom_segment(
      data = line_data,
      ggplot2::aes(x = x, xend = xend, y = y, yend = yend),
      color = "#A0A0A0", linewidth = 0.25) +
    ggplot2::scale_fill_identity() +
    ggplot2::scale_alpha_identity() +
    ggplot2::scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    ggplot2::scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    ggplot2::theme_void() +
    ggplot2::theme(
      plot.background = ggplot2::element_rect(fill = NA, color = NA),
      panel.background = ggplot2::element_rect(fill = NA, color = NA),
      plot.margin = ggplot2::margin(0, 0, 0, 0, "pt"))
}
