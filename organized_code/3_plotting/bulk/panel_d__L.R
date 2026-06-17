# panel_d.R (v3) — e160 graph ablation, three variants
#
# 内容：在 SCP682 e160 训练预算下做的图约束消融：
#   左 facet (Axis): model baseline / site-only / sample-only / dual / SCP682 main
#   右 facet (Edge-source): CoPheeKSA only / CoPheeMap only / KSTAR only / SCP682 main
#
# 每个 facet 加 SCP682 主模型（panel b/c 用的 main result 0.5474）作 reference，
# 区分 ablation grid 内部 dual (0.5403) 与 main paper SCP682 (0.5474)。
#
# 本文件提供三个变体：
#   make_panel_d_lollipop()  — 水平 lollipop
#   make_panel_d_bar()       — 水平 bar plot
#   make_panel_d_forest()    — forest plot
# 默认 make_panel_d() = make_panel_d_lollipop()。

# ---------------------------------------------------------------------------
# 数据（hardcoded, 来自 fig2_panel_d_data.tsv + headline_metrics SCP682 main）
# ---------------------------------------------------------------------------
.panel_d_data <- function() {
  data.frame(
    facet  = c(rep("Axis ablation", 5), rep("Edge-source ablation", 4)),
    config = c("model baseline", "site-only", "sample-only", "dual (axis grid)",
               "SCP682 main",
               "CoPheeKSA only", "CoPheeMap only", "KSTAR only", "SCP682 main"),
    median = c(0.3053, 0.5127, 0.5517, 0.5403, 0.5474,
               0.5400, 0.5471, 0.5482, 0.5474),
    is_main = c(FALSE, FALSE, FALSE, FALSE, TRUE,
                FALSE, FALSE, FALSE, TRUE),
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# 配色：model baseline 灰；ablation 用 panel b/c 同青绿系；SCP682 main 暖橙
# ---------------------------------------------------------------------------
.PANEL_D_COLORS <- c(
  "model baseline"   = "#D8D8D8",
  "site-only"        = "#7BC0CD",
  "sample-only"      = "#51999F",
  "dual (axis grid)" = "#4198AC",
  "CoPheeKSA only"   = "#7BC0CD",
  "CoPheeMap only"   = "#51999F",
  "KSTAR only"       = "#4198AC",
  "SCP682 main"      = "#ED8D5A"
)

# ---------------------------------------------------------------------------
# 共享主题
# ---------------------------------------------------------------------------
.panel_d_theme <- function() {
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
      axis.title          = ggplot2::element_text(size = 6.8, color = "#222222"),
      axis.text           = ggplot2::element_text(size = 6.2, color = "#222222"),
      axis.line           = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.ticks          = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.ticks.length   = ggplot2::unit(0.7, "mm"),
      strip.background    = ggplot2::element_rect(fill = "#F5F5F5", color = NA),
      strip.text          = ggplot2::element_text(size = 6.5, color = "#222222",
                                                  margin = ggplot2::margin(2, 0, 2, 0)),
      panel.spacing.x     = ggplot2::unit(2.0, "mm"),
      legend.position     = "none",
      plot.margin         = ggplot2::margin(6, 8, 4, 8, "pt")
    )
}

# ---------------------------------------------------------------------------
# 变体 1: 水平 Lollipop
# ---------------------------------------------------------------------------
make_panel_d_lollipop <- function() {
  d <- .panel_d_data()
  d$facet <- factor(d$facet, levels = c("Axis ablation", "Edge-source ablation"))
  d <- d[order(d$facet, d$median), ]
  d$config <- factor(d$config, levels = unique(d$config))

  ggplot2::ggplot(d, ggplot2::aes(x = median, y = config, color = config)) +
    ggplot2::geom_segment(ggplot2::aes(x = 0, xend = median, yend = config),
                          linewidth = 0.6) +
    ggplot2::geom_point(size = 2.4) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", median)),
                       hjust = -0.25, family = "Arial", size = 1.9,
                       color = "#222222") +
    ggplot2::scale_color_manual(values = .PANEL_D_COLORS) +
    ggplot2::scale_x_continuous(limits = c(0, 0.70),
                                breaks = c(0, 0.2, 0.4, 0.6),
                                expand = c(0, 0)) +
    ggplot2::facet_wrap(~ facet, nrow = 1, scales = "free_y") +
    ggplot2::labs(
      x        = "Median Spearman ρ (pseudo-external)",
      y        = NULL,
      title    = "e160 graph ablation: axis and edge-source controls",
      subtitle = "Each panel includes SCP682 main result (0.5474, orange) as cross-figure reference.") +
    .panel_d_theme()
}

# ---------------------------------------------------------------------------
# 变体 2: 水平 Bar plot
# ---------------------------------------------------------------------------
make_panel_d_bar <- function() {
  d <- .panel_d_data()
  d$facet <- factor(d$facet, levels = c("Axis ablation", "Edge-source ablation"))
  d <- d[order(d$facet, d$median), ]
  d$config <- factor(d$config, levels = unique(d$config))

  ggplot2::ggplot(d, ggplot2::aes(x = median, y = config, fill = config)) +
    ggplot2::geom_col(width = 0.65, color = "black", linewidth = 0.25) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", median)),
                       hjust = -0.15, family = "Arial", size = 1.9,
                       color = "#222222") +
    ggplot2::scale_fill_manual(values = .PANEL_D_COLORS) +
    ggplot2::scale_x_continuous(limits = c(0, 0.70),
                                breaks = c(0, 0.2, 0.4, 0.6),
                                expand = c(0, 0)) +
    ggplot2::facet_wrap(~ facet, nrow = 1, scales = "free_y") +
    ggplot2::labs(
      x        = "Median Spearman ρ (pseudo-external)",
      y        = NULL,
      title    = "e160 graph ablation: axis and edge-source controls",
      subtitle = "Each panel includes SCP682 main result (0.5474, orange) as cross-figure reference.") +
    .panel_d_theme()
}

# ---------------------------------------------------------------------------
# 变体 3: Forest plot
# ---------------------------------------------------------------------------
make_panel_d_forest <- function() {
  d <- .panel_d_data()
  d$facet <- factor(d$facet, levels = c("Axis ablation", "Edge-source ablation"))
  d <- d[order(d$facet, d$median), ]
  d$config <- factor(d$config, levels = unique(d$config))

  ggplot2::ggplot(d, ggplot2::aes(x = median, y = config, color = config)) +
    ggplot2::geom_vline(xintercept = 0, color = "#A8A8A8",
                        linewidth = 0.3, linetype = "dashed") +
    ggplot2::geom_segment(ggplot2::aes(x = 0, xend = median, yend = config),
                          linewidth = 0.3, alpha = 0.4) +
    ggplot2::geom_point(size = 3.2) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.3f", median)),
                       hjust = -0.3, family = "Arial", size = 1.9,
                       color = "#222222", fontface = "bold") +
    ggplot2::scale_color_manual(values = .PANEL_D_COLORS) +
    ggplot2::scale_x_continuous(limits = c(0, 0.72),
                                breaks = c(0, 0.2, 0.4, 0.6),
                                expand = c(0, 0)) +
    ggplot2::facet_wrap(~ facet, nrow = 1, scales = "free_y") +
    ggplot2::labs(
      x        = "Median Spearman ρ (pseudo-external)",
      y        = NULL,
      title    = "e160 graph ablation: axis and edge-source controls",
      subtitle = "Each panel includes SCP682 main result (0.5474, orange) as cross-figure reference.") +
    .panel_d_theme()
}

# 默认导出 lollipop
make_panel_d <- make_panel_d_lollipop
