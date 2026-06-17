# panel_a.R — Fig 2 panel a：SCP682 架构示意图（横长 170 × 55 mm 版本，v2 松弛化）
#
# 用途：source 后调用 make_panel_a() 返回 ggplot 对象。panel letter 不在此处加，
# 由合成层 cowplot::plot_grid(labels = ...) 注入。
#
# v2 松弛化修订要点（相对 v1）：
#   1. 全部 annotate text 改用 `pt / .pt` 写法，字号直接以 pt 标注，体内主文字最小 6 pt
#   2. Site graph 顶框去掉第三行 "CoPheeMap / CoPheeKSA / KSTAR" 细分，框内只留汇总
#   3. Sample graph 底框压扁到约 0.85 单位高，两行字垂直拉开
#   4. Bφ / Gθ 方框内三行文字垂直行距加大（原 0.42 / 0.65 / 0.32 拉到 0.55 / 0.75 / 0.50）
#   5. ŷ 方框宽度从 1.3→1.1 单位略缩，让中间留出更多空白
#
# 设计要点：
#   - 坐标系 xlim=c(0,17) ylim=c(0,5.5)，对应 170×55 mm 横长比例
#   - 顶部 Site graph 横长矩形 / 底部 Sample graph 横长矩形（横向铺满）
#   - 中部三方框 Bφ / Gθ / ŷ phospho 横向拉开间距
#   - 箭头流向：Bφ → Gθ → ŷ；Site graph ↓ Gθ；Sample graph ↑ Gθ
#   - Bφ → ŷ skip 连接（虚线弧）
#
# 配色（sci-plot SKILL 低饱和度）：
#   Site/Sample graph + Gθ  金色系：fill #F5E8D0 / stroke #D4A56B / text #6B4A1F
#   Bφ                       蓝色系：fill #C1D8E9 / stroke #5680B0 / text #1F3A5F
#   ŷ                        深蓝填充：fill #3D5A80 / stroke #1F3A5F / text white
#
# 字体：Arial；标题 7 pt（外部 theme_fig2 注入）；
#       体内文字 9–11 pt 粗体大字符号、6.5–7 pt 主描述、5.8–6 pt 斜体小说明
#       全部用 `pt / .pt` 转换为 mm 单位
#
# 数字来源（详见 fig2_panel_a_components.md）：
#   18,592 sites / 420,102 edges       —— graph_statistics.tsv L2
#   1,431 samples / 21,925 edges        —— graph_statistics.tsv L3
#   ρ = 0.5474                          —— headline_metrics.tsv L10
#
# 注：CoPheeMap 386,224 / CoPheeKSA 6,520 / KSTAR 45,600 三个来源细分
# 在 v2 中移到图注/components.md 中，框内不再展示，避免顶框 3 行文字挤。

make_panel_a <- function() {

  # ---- 配色常量 ----
  COL_GOLD_FILL   <- "#F5E8D0"
  COL_GOLD_STROKE <- "#D4A56B"
  COL_GOLD_TEXT   <- "#6B4A1F"
  COL_GOLD_SUB    <- "#3B2E1A"

  COL_BLUE_FILL   <- "#C1D8E9"
  COL_BLUE_STROKE <- "#5680B0"
  COL_BLUE_TEXT   <- "#1F3A5F"

  COL_OUT_FILL    <- "#3D5A80"
  COL_OUT_STROKE  <- "#1F3A5F"

  COL_TEXT_MAIN   <- "#222222"

  # ---- 几何布局（坐标范围 17 × 5.5） ----
  # Site graph 顶部横条（v2：高度从 0.75 缩到 0.55，两行字垂直拉开）
  site_xmin <- 1.5; site_xmax <- 15.5
  site_ymin <- 4.70; site_ymax <- 5.30

  # Sample graph 底部横条（v2：高度从 0.75 压扁到 0.60，两行字垂直拉开）
  samp_xmin <- 1.5; samp_xmax <- 15.5
  samp_ymin <- 0.30; samp_ymax <- 0.90

  # 中部三方框（同一垂直区间，横向间距拉开）
  # v2：方框上下高度从 2.0 单位扩到 2.05，垂直内边距更大
  mid_ymin  <- 1.65; mid_ymax  <- 3.70

  bphi_xmin   <- 0.40; bphi_xmax   <- 4.30
  gtheta_xmin <- 6.10; gtheta_xmax <- 11.30
  # v2：ŷ 方框宽度从 3.5（13.10–16.60）缩到 3.1（13.30–16.40），留出更多间距
  yhat_xmin   <- 13.30; yhat_xmax  <- 16.40

  # 三方框水平中心 y
  mid_y_center <- (mid_ymin + mid_ymax) / 2  # = 2.675

  # 方框内 3 行文字的 y 坐标（v2：行距加大）
  # 顶行：粗体大字符号 (Bφ / Gθ / ŷ)
  # 中行：主描述
  # 底行：斜体小说明
  row_top    <- mid_ymax - 0.55
  row_middle <- mid_y_center - 0.05
  row_bottom <- mid_ymin + 0.45

  # ---- 绘图 ----
  ggplot() +
    coord_cartesian(xlim = c(0, 17), ylim = c(0, 5.5), expand = FALSE) +

    # ------------------------------------------------------------------
    # 顶部 Site graph 横条（v2：去掉第三行 CoPheeMap/CoPheeKSA/KSTAR 细分）
    # ------------------------------------------------------------------
    annotate("rect",
             xmin = site_xmin, xmax = site_xmax,
             ymin = site_ymin, ymax = site_ymax,
             fill = COL_GOLD_FILL, color = COL_GOLD_STROKE, linewidth = 0.35) +
    annotate("text",
             x = (site_xmin + site_xmax) / 2, y = site_ymax - 0.18,
             label = "Site graph (phospho × phospho)",
             family = "Arial", size = 7 / .pt, fontface = "bold",
             color = COL_GOLD_TEXT) +
    annotate("text",
             x = (site_xmin + site_xmax) / 2, y = site_ymin + 0.17,
             label = "18,592 sites · 420,102 edges",
             family = "Arial", size = 6.5 / .pt,
             color = COL_GOLD_SUB) +

    # ------------------------------------------------------------------
    # Bφ 方框（左）
    # ------------------------------------------------------------------
    annotate("rect",
             xmin = bphi_xmin, xmax = bphi_xmax,
             ymin = mid_ymin, ymax = mid_ymax,
             fill = COL_BLUE_FILL, color = COL_BLUE_STROKE, linewidth = 0.4) +
    annotate("text",
             x = (bphi_xmin + bphi_xmax) / 2, y = row_top,
             label = "Bφ",
             family = "Arial", size = 11 / .pt, fontface = "bold",
             color = COL_BLUE_TEXT) +
    annotate("text",
             x = (bphi_xmin + bphi_xmax) / 2, y = row_middle,
             label = "general expression baseline",
             family = "Arial", size = 6.5 / .pt,
             color = COL_BLUE_TEXT) +
    annotate("text",
             x = (bphi_xmin + bphi_xmax) / 2, y = row_bottom,
             label = "OOF-calibrated on RNA + cancer group",
             family = "Arial", size = 6 / .pt, fontface = "italic",
             color = COL_BLUE_TEXT) +

    # ------------------------------------------------------------------
    # Gθ 方框（中）
    # ------------------------------------------------------------------
    annotate("rect",
             xmin = gtheta_xmin, xmax = gtheta_xmax,
             ymin = mid_ymin, ymax = mid_ymax,
             fill = COL_GOLD_FILL, color = COL_GOLD_STROKE, linewidth = 0.4) +
    annotate("text",
             x = (gtheta_xmin + gtheta_xmax) / 2, y = row_top,
             label = "Gθ",
             family = "Arial", size = 11 / .pt, fontface = "bold",
             color = COL_GOLD_TEXT) +
    annotate("text",
             x = (gtheta_xmin + gtheta_xmax) / 2, y = row_middle,
             label = "graph residual (dual-axis GNN)",
             family = "Arial", size = 6.5 / .pt,
             color = COL_GOLD_TEXT) +
    annotate("text",
             x = (gtheta_xmin + gtheta_xmax) / 2, y = row_bottom,
             label = "site + sample message passing",
             family = "Arial", size = 6 / .pt, fontface = "italic",
             color = COL_GOLD_TEXT) +

    # ------------------------------------------------------------------
    # ŷ phospho 输出方框（右）
    # ------------------------------------------------------------------
    annotate("rect",
             xmin = yhat_xmin, xmax = yhat_xmax,
             ymin = mid_ymin, ymax = mid_ymax,
             fill = COL_OUT_FILL, color = COL_OUT_STROKE, linewidth = 0.4) +
    annotate("text",
             x = (yhat_xmin + yhat_xmax) / 2, y = row_top,
             label = "ŷ",
             family = "Arial", size = 12 / .pt, fontface = "bold", color = "white") +
    annotate("text",
             x = (yhat_xmin + yhat_xmax) / 2, y = row_middle,
             label = "phospho",
             family = "Arial", size = 6.5 / .pt, color = "white") +
    annotate("text",
             x = (yhat_xmin + yhat_xmax) / 2, y = row_bottom,
             label = "ρ = 0.5474",
             family = "Arial", size = 6.5 / .pt, fontface = "bold", color = "white") +

    # ------------------------------------------------------------------
    # 水平主箭头：Bφ → Gθ → ŷ
    # ------------------------------------------------------------------
    annotate("segment",
             x = bphi_xmax + 0.05, xend = gtheta_xmin - 0.05,
             y = mid_y_center, yend = mid_y_center,
             color = COL_BLUE_TEXT, linewidth = 0.45,
             arrow = arrow(length = unit(1.1, "mm"), type = "closed")) +
    annotate("segment",
             x = gtheta_xmax + 0.05, xend = yhat_xmin - 0.05,
             y = mid_y_center, yend = mid_y_center,
             color = COL_GOLD_TEXT, linewidth = 0.45,
             arrow = arrow(length = unit(1.1, "mm"), type = "closed")) +

    # ------------------------------------------------------------------
    # 垂直点线箭头：Site graph ↓ Gθ / Sample graph ↑ Gθ
    # ------------------------------------------------------------------
    annotate("segment",
             x = (gtheta_xmin + gtheta_xmax) / 2,
             xend = (gtheta_xmin + gtheta_xmax) / 2,
             y = site_ymin - 0.02, yend = mid_ymax + 0.05,
             color = COL_GOLD_STROKE, linewidth = 0.35, linetype = "dotted",
             arrow = arrow(length = unit(0.95, "mm"))) +
    annotate("segment",
             x = (gtheta_xmin + gtheta_xmax) / 2,
             xend = (gtheta_xmin + gtheta_xmax) / 2,
             y = samp_ymax + 0.02, yend = mid_ymin - 0.05,
             color = COL_GOLD_STROKE, linewidth = 0.35, linetype = "dotted",
             arrow = arrow(length = unit(0.95, "mm"))) +

    # ------------------------------------------------------------------
    # Skip 连接：Bφ → ŷ（虚线弧形走 Gθ 上方，避开 Site graph 横条）
    # 用两段直线 + 转折点近似弧线
    # Site graph 底边 y=4.70, 中部方框顶 y=3.70；
    # skip 走 y=4.15（位于两者中间，离 Site graph 至少 0.55 单位）
    # ------------------------------------------------------------------
    # 上行：从 Bφ 顶部中央升到 skip 高度
    annotate("segment",
             x = (bphi_xmin + bphi_xmax) / 2,
             xend = (bphi_xmin + bphi_xmax) / 2,
             y = mid_ymax + 0.02, yend = 4.15,
             color = COL_BLUE_STROKE, linewidth = 0.32, linetype = "dashed") +
    # 水平段：跨过 Gθ 上方（在 Site graph 与三方框之间的空白带）
    annotate("segment",
             x = (bphi_xmin + bphi_xmax) / 2,
             xend = (yhat_xmin + yhat_xmax) / 2,
             y = 4.15, yend = 4.15,
             color = COL_BLUE_STROKE, linewidth = 0.32, linetype = "dashed") +
    # 下行：落入 ŷ 顶部
    annotate("segment",
             x = (yhat_xmin + yhat_xmax) / 2,
             xend = (yhat_xmin + yhat_xmax) / 2,
             y = 4.15, yend = mid_ymax + 0.02,
             color = COL_BLUE_STROKE, linewidth = 0.32, linetype = "dashed",
             arrow = arrow(length = unit(0.9, "mm"))) +
    # skip 文字标签（放在水平段下方，让其与水平虚线分离）
    annotate("text",
             x = ((bphi_xmin + bphi_xmax) / 2 +
                  (yhat_xmin + yhat_xmax) / 2) / 2,
             y = 4.38,
             label = "skip (additive)",
             family = "Arial", size = 6 / .pt, fontface = "italic",
             color = COL_BLUE_TEXT) +

    # ------------------------------------------------------------------
    # 底部 Sample graph 横条（v2：压扁高度，两行字垂直拉开）
    # ------------------------------------------------------------------
    annotate("rect",
             xmin = samp_xmin, xmax = samp_xmax,
             ymin = samp_ymin, ymax = samp_ymax,
             fill = COL_GOLD_FILL, color = COL_GOLD_STROKE, linewidth = 0.35) +
    annotate("text",
             x = (samp_xmin + samp_xmax) / 2, y = samp_ymax - 0.18,
             label = "Sample graph (RNA-similarity kNN)",
             family = "Arial", size = 7 / .pt, fontface = "bold",
             color = COL_GOLD_TEXT) +
    annotate("text",
             x = (samp_xmin + samp_xmax) / 2, y = samp_ymin + 0.17,
             label = "1,431 samples · 21,925 edges",
             family = "Arial", size = 6.5 / .pt,
             color = COL_GOLD_SUB) +

    # ------------------------------------------------------------------
    # 标题
    # ------------------------------------------------------------------
    labs(title = "Bφ baseline + Gθ graph residual") +

    # ------------------------------------------------------------------
    # 主题：架构图不需要坐标轴
    # ------------------------------------------------------------------
    theme(
      axis.line   = element_blank(),
      axis.title  = element_blank(),
      axis.text   = element_blank(),
      axis.ticks  = element_blank()
    )
}
