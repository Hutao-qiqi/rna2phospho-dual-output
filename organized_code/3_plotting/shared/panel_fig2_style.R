suppressMessages({
  library(ggplot2)
  library(grid)
  library(scales)
  library(RColorBrewer)
})

# Fig 2 style constants copied into Fig 5 so every panel uses the same visual grammar.
FIG2_COL_TEXT <- "#222222"
FIG2_COL_MUTED <- "#555555"
FIG2_COL_GRID <- "#D8D8D8"

FIG2_METHOD_COLORS <- c(
  "Mean"          = "#D8D8D8",
  "DeepGxP"       = "#BFDFD2",
  "Cognate mRNA"  = "#7BC0CD",
  "Ridge"         = "#51999F",
  "PC ridge"      = "#4198AC",
  "VAE"           = "#DBCB92",
  "MLP"           = "#ECB66C",
  "Elastic net"   = "#EA9E58",
  "SCP682"        = "#ED8D5A"
)

FIG2_BLUE <- "#2171B5"
FIG2_BLUE_DARK <- "#08306B"
FIG2_BLUE_SOFT <- "#C1D8E9"
FIG2_ORANGE <- "#ED8D5A"
FIG2_ORANGE_SOFT <- "#F5E8D0"
FIG2_RED <- "#CB181D"
FIG2_RED_DARK <- "#67000D"
FIG2_NEUTRAL <- "#FFFFFF"

# Fig 2 的原热图色轴很强，这里保留蓝-白-红方向，但降低饱和度用于 Fig 5 的临床效应热图。
FIG2_HEATMAP_COLORS <- c("#2F5F8F", "#9FC4E3", FIG2_NEUTRAL, "#F0B9A7", "#B85A50")
FIG2_CLUSTER_COLORS <- RColorBrewer::brewer.pal(8, "Set2")
FIG2_CANCER_COLORS <- RColorBrewer::brewer.pal(12, "Set3")

fig2_panel_theme <- function(base_size = 7) {
  ggplot2::theme_classic(base_size = base_size, base_family = "Arial") +
    ggplot2::theme(
      panel.background    = ggplot2::element_rect(fill = "white", color = NA),
      plot.background     = ggplot2::element_rect(fill = "white", color = NA),
      plot.title          = ggplot2::element_text(
        size = 7, face = "plain", hjust = 0, color = FIG2_COL_TEXT,
        margin = ggplot2::margin(0, 0, 1, 0)),
      plot.subtitle       = ggplot2::element_text(
        size = 5.8, face = "italic", hjust = 0, color = FIG2_COL_MUTED,
        margin = ggplot2::margin(0, 0, 3, 0)),
      plot.title.position = "plot",
      axis.title          = ggplot2::element_text(size = 6.8, color = FIG2_COL_TEXT),
      axis.text           = ggplot2::element_text(size = 6.2, color = FIG2_COL_TEXT),
      axis.line           = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.ticks          = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.ticks.length   = ggplot2::unit(0.7, "mm"),
      strip.background    = ggplot2::element_rect(fill = "#F5F5F5", color = NA),
      strip.text          = ggplot2::element_text(size = 6.5, color = FIG2_COL_TEXT,
                                                  margin = ggplot2::margin(2, 0, 2, 0)),
      panel.grid.major.y  = ggplot2::element_line(color = "#EFEFEF", linewidth = 0.25),
      panel.grid.major.x  = ggplot2::element_blank(),
      legend.text         = ggplot2::element_text(size = 6.5, color = FIG2_COL_TEXT),
      legend.title        = ggplot2::element_text(size = 6.2, color = FIG2_COL_TEXT),
      legend.background   = ggplot2::element_rect(fill = "white", color = NA),
      legend.key.size     = ggplot2::unit(3.5, "mm"),
      legend.position     = "bottom",
      legend.margin       = ggplot2::margin(2, 0, 0, 0),
      plot.margin         = ggplot2::margin(6, 8, 2, 8, "pt")
    )
}

fig2_heat_palette <- function(n = 101) {
  grDevices::colorRampPalette(FIG2_HEATMAP_COLORS)(n)
}
