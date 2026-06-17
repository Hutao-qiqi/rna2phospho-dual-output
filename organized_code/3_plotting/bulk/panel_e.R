# panel_e.R — Fig 2 panel e 模块
#
# 内容：图残差权重 alpha 的收缩扫描双曲线。
#   - X 轴 alpha 从 0.0 到 1.0，步长 0.1，共 11 个测点
#   - 两条评估模式曲线：full_grid_trainmode（暖橘金，圆点）
#                       full_grid_pseudo_external（深蓝，方块）
#   - alpha=1.0 处加黑色 dotted 垂直线 + 右上角 selected 白底标注框
#   - alpha 在 0.5 至 1.0 区间画 plateau 浅橙阴影 + italic 注释
#   - 数字标注：仅在偶数 alpha（0.0/0.2/0.4/0.6/0.8/1.0）标 4 位小数 6 pt，
#               train-mode 在上、pseudo-external 在下；为视觉舒展，
#               原本每点都标的 22 个标注收缩到 12 个
#
# 调用方式：source 本文件后调用 make_panel_e() 返回 ggplot 对象。
# 本文件不在顶部 library() 也不 setwd()，调用方需自行加载 ggplot2。
#
# 输出尺寸约束：170 x 65 mm（A4 portrait 末行整宽）。
# 字体：Arial。无 panel letter。

# ---------------------------------------------------------------------------
# 路径与配色（sci-plot SKILL 低饱和度方案）
# ---------------------------------------------------------------------------
.PANEL_E_ROOT <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682"
.PANEL_E_OUT  <- file.path(.PANEL_E_ROOT, "04_figure_source_data", "fig2")
.PANEL_E_TSV  <- file.path(.PANEL_E_OUT, "fig2_panel_e_data.tsv")

.PANEL_E_COL_DARK_BLUE <- "#3D5A80"  # pseudo-external（square）
.PANEL_E_COL_GOLD      <- "#C97A52"  # train-mode（circle）
.PANEL_E_COL_PLATEAU   <- "#FFE0C2"  # plateau 阴影 fill
.PANEL_E_COL_PLATEAU_T <- "#7F4D1A"  # plateau 注释文字
.PANEL_E_COL_TEXT      <- "#222222"  # 标题与轴

# ---------------------------------------------------------------------------
# 本面板专用主题
# ---------------------------------------------------------------------------
# 字号档位（本轮放宽后）：
#   - 轴标题 7.5 pt（原 6.5）
#   - 轴 tick text 7 pt（原 6.0）
#   - 图例 text 7 pt（原 6.0）
.panel_e_theme <- function() {
  ggplot2::theme_classic(base_size = 7, base_family = "Arial") +
    ggplot2::theme(
      panel.background    = ggplot2::element_rect(fill = "white", color = NA),
      plot.background     = ggplot2::element_rect(fill = "white", color = NA),
      plot.title          = ggplot2::element_text(
        size = 7, face = "plain", hjust = 0,
        color = .PANEL_E_COL_TEXT,
        margin = ggplot2::margin(0, 0, 3, 0)),
      plot.title.position = "plot",
      axis.title          = ggplot2::element_text(size = 7.5,
                                                  color = .PANEL_E_COL_TEXT),
      axis.text           = ggplot2::element_text(size = 7.0,
                                                  color = .PANEL_E_COL_TEXT),
      axis.line           = ggplot2::element_line(linewidth = 0.3,
                                                  color = "black"),
      axis.ticks          = ggplot2::element_line(linewidth = 0.3,
                                                  color = "black"),
      axis.ticks.length   = ggplot2::unit(0.7, "mm"),
      legend.text         = ggplot2::element_text(size = 7.0,
                                                  color = .PANEL_E_COL_TEXT),
      legend.title        = ggplot2::element_blank(),
      legend.background   = ggplot2::element_rect(fill = "white", color = NA),
      legend.key          = ggplot2::element_rect(fill = "white", color = NA),
      legend.key.size     = ggplot2::unit(3.0, "mm"),
      legend.spacing.y    = ggplot2::unit(0.4, "mm"),
      plot.margin         = ggplot2::margin(6, 8, 3, 6, "pt")
    )
}

# ---------------------------------------------------------------------------
# make_panel_e
# ---------------------------------------------------------------------------
# 返回：ggplot 对象。
# 数据：fig2_panel_e_data.tsv，22 行（2 mode x 11 alpha）。
# 布局要点（170 mm 横宽版，本轮放宽）：
#   - X 轴范围 c(-0.02, 1.08)，breaks seq(0, 1, 0.2)
#   - Y 轴范围 c(0.25, 0.65)，breaks seq(0.3, 0.6, 0.1)
#   - 数字标注：仅在 alpha ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} 标 4 位小数 6 pt
#               （减半到 12 个，横向间距由 0.1 → 0.2 单位 ≈ 27 mm，留白足够）
#               train-mode 在点上方 vjust=-1.4，pseudo-external 在点下方 vjust=1.8
#   - alpha=1.0 处避免数字标注与右上 selected 框 / dotted 线重叠：
#     单独把 alpha=1.0 两条曲线的数字 nudge 到点左上方水平位 hjust=1.15
#   - 图例放图内左上 inside c(0.02, 0.98)，避让 plateau 阴影中的注释
#   - 点 size 2.0、折线 linewidth 0.7，整体视觉更舒展
make_panel_e <- function() {
  se <- utils::read.delim(.PANEL_E_TSV,
                          sep = "\t", stringsAsFactors = FALSE)
  se$mode <- ifelse(se$scan_name == "full_grid_trainmode",
                    "train-mode", "pseudo-external")
  se$mode <- factor(se$mode, levels = c("pseudo-external", "train-mode"))

  # 数字标注子集：alpha ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
  # 用 round 防 0.1 浮点尾差，比较保留 1 位小数粒度
  se$alpha_round1 <- round(se$shrinkage, 1)
  alpha_label_set <- c(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
  se$show_label   <- se$alpha_round1 %in% alpha_label_set
  # alpha=0.0 时两条曲线 y 完全相同（0.3053），重复标注会叠
  # 只保留 train-mode 的 0.0 标签（pseudo-external 0.0 标签去掉）
  se$show_label[se$shrinkage < 0.01 & se$mode == "pseudo-external"] <- FALSE

  # 数字标注偏移：train-mode 上方、pseudo-external 下方（4 位小数保留）
  se$label_str <- sprintf("%.4f", se$median_spearman)
  se$hjust_x   <- 0.5
  se$vjust_y   <- ifelse(se$mode == "train-mode", -1.4, 1.8)

  # alpha=1.0 + train-mode 单独把标签拉到点的左下方
  # 避开右上 selected 框 (1.02, 0.62) 和 dotted 垂直线
  mask_tail_train <- (se$shrinkage > 0.99) & (se$mode == "train-mode")
  se$hjust_x[mask_tail_train] <- 1.15
  se$vjust_y[mask_tail_train] <- 1.6

  # alpha=1.0 + pseudo-external 标签拉到点的左下方，避免 dotted 垂直线压住数字
  mask_tail_ext <- (se$shrinkage > 0.99) & (se$mode == "pseudo-external")
  se$hjust_x[mask_tail_ext] <- 1.15
  se$vjust_y[mask_tail_ext] <- 1.6

  # alpha=0.0 处标签向右挑避免越过 Y 轴左边界
  # 该 alpha 只剩 train-mode 一个标签（pseudo-external 已去掉）
  # 由于两曲线 y 重合于 0.3053，把唯一标签放到点的右下方避免越过 Y 轴 + 不挡图例
  mask_head <- (se$shrinkage < 0.01)
  se$hjust_x[mask_head] <- 0
  se$vjust_y[mask_head] <- 1.8

  # 标注子集 dataframe（只画在 alpha_label_set 上的 12 个标签）
  se_lab <- se[se$show_label, , drop = FALSE]

  ggplot2::ggplot(se, ggplot2::aes(x = shrinkage, y = median_spearman,
                                   color = mode, shape = mode)) +
    # plateau 阴影：alpha in [0.5, 1.0]
    ggplot2::annotate("rect",
                      xmin = 0.5, xmax = 1.0,
                      ymin = -Inf, ymax = Inf,
                      fill = .PANEL_E_COL_PLATEAU, alpha = 0.30) +
    # plateau 文字注释（区间中央，y=0.39 远离曲线，字号放宽 5.5 → 6.5 pt）
    ggplot2::annotate("text",
                      x = 0.75, y = 0.39,
                      label = "plateau (α ≥ 0.5)",
                      family = "Arial", size = 6.5 / ggplot2::.pt,
                      color = .PANEL_E_COL_PLATEAU_T,
                      fontface = "italic") +
    # 折线 + 点（折线 0.5 → 0.7，点 1.7 → 2.0）
    ggplot2::geom_line(linewidth = 0.7) +
    ggplot2::geom_point(size = 2.0) +
    # 数字标注：仅在 se_lab（alpha = 0/0.2/0.4/0.6/0.8/1.0，共 12 个）
    # 字号 5 → 6 pt（保留 4 位小数）
    ggplot2::geom_text(
      data = se_lab,
      ggplot2::aes(label = label_str,
                   hjust = hjust_x, vjust = vjust_y),
      family = "Arial", size = 6 / ggplot2::.pt,
      show.legend = FALSE) +
    # alpha=1.0 黑色 dotted 垂直线
    ggplot2::geom_vline(xintercept = 1.0,
                        linetype = "dotted",
                        color = "black",
                        linewidth = 0.4) +
    # 右上 selected 标注（白底黑边小框，7 pt 加粗）—— 位置 (1.02, 0.62) 略偏右
    # 避开 α=0.9/1.0 train-mode 数字标签和 dotted 线
    ggplot2::annotate("label",
                      x = 1.02, y = 0.62,
                      label = "α = 1.0",
                      family = "Arial",
                      size = 7 / ggplot2::.pt,
                      fontface = "bold",
                      color = "black",
                      fill = "white",
                      label.size = 0.25,
                      label.padding = ggplot2::unit(0.8, "mm"),
                      label.r = ggplot2::unit(0.4, "mm"),
                      hjust = 1) +
    # 配色 + 形状
    ggplot2::scale_color_manual(
      values = c("pseudo-external" = .PANEL_E_COL_DARK_BLUE,
                 "train-mode"      = .PANEL_E_COL_GOLD),
      breaks = c("train-mode", "pseudo-external")) +
    ggplot2::scale_shape_manual(
      values = c("pseudo-external" = 15,
                 "train-mode"      = 16),
      breaks = c("train-mode", "pseudo-external")) +
    ggplot2::scale_x_continuous(
      limits = c(-0.02, 1.08),
      breaks = seq(0, 1, 0.2),
      expand = c(0, 0)) +
    ggplot2::scale_y_continuous(
      limits = c(0.25, 0.65),
      breaks = seq(0.3, 0.6, 0.1),
      expand = c(0, 0)) +
    ggplot2::labs(
      x = "Graph residual weight α",
      y = "Median Spearman ρ (n = 18,413)",
      title = "Shrinkage sweep: selected α = 1.0") +
    .panel_e_theme() +
    ggplot2::theme(
      # 图例从右下挪到 plot 内左上 c(0.02, 0.98)，避免压 plateau 阴影注释
      legend.position = "inside",
      legend.position.inside = c(0.02, 0.98),
      legend.justification   = c(0, 1),
      legend.direction       = "vertical",
      legend.margin          = ggplot2::margin(1, 2, 1, 2, "pt")
    )
}
