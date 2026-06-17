# theme_fig3.R — Fig 3 (SCP682-SC) 共享主题 + 调色板
#
# 与 Fig 2 同一套 house style：theme_classic(base 7, Arial) + 低饱和冷→暖板，
# "ours / 强调" 用暖橙 #ED8D5A。所有 panel_*.R 与主合成脚本都 source 本文件。
#
# 字号约定（pt → mm 用 PT 常量换算；geom_text/annotate 的 size 单位是 mm）：
#   标题 7 / 副标题 5.8 / 轴标题 6.8 / 轴文字 6.2 / 图例 6.2 / 体内数字 5–5.5

suppressMessages(library(ggplot2))

# pt → mm: 1 pt = 1/.pt mm，.pt = 72.27/25.4 = 2.845276
PT <- 2.845276

# ---- 数据路径常量（多张图共用）----
.SC11   <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11"
FIG3_DIR  <- file.path(.SC11, "04_figure_source_data", "fig3")
RNA_SD    <- file.path(.SC11, "04_figure_source_data",
                       "sc11_validation_rna_nmf_v4", "source_data")
VIZ_SD    <- file.path(.SC11, "04_figure_source_data",
                       "sc11_visualization_panels_v1", "source_data")
DT_EXT    <- file.path(.SC11, "02_data_tables", "external_predicted_observed")
RV_DIR    <- file.path(.SC11, "04_figure_source_data", "reviewer_requested_tables_v2")
SRC_CACHE <- file.path("E:/data/gongke/TCGA-TCPA", "remote_scripts",
                       "_paper_extract_sources", "sc11_result", "tables")

# ---- 低饱和冷→暖板（与 Fig 2 panel b/c 同源）----
FIG3_PAL <- c(
  mint      = "#BFDFD2",
  cyan      = "#7BC0CD",
  teal      = "#51999F",
  deepteal  = "#4198AC",
  sand      = "#DBCB92",
  lightorg  = "#ECB66C",
  midorg    = "#EA9E58",
  ours      = "#ED8D5A",
  grey      = "#C4C4C4",
  greyfloor = "#D8D8D8"
)

# ρ / 强度发散板（与 Fig 2 .RHO_COLORS 一致）
FIG3_RHO_COLORS <- c("#BFDFD2", "#7BC0CD", "#DBCB92", "#ECB66C", "#ED8D5A")
# 密度（计数）顺序板：浅 → 深青
FIG3_SEQ_TEAL   <- c("#ECF3F4", "#BFDFD2", "#7BC0CD", "#4198AC", "#1F5A66")

COL_TEXT <- "#222222"
COL_SUB  <- "#555555"
COL_ZERO <- "#A8A8A8"

theme_fig3 <- function(base = 7) {
  ggplot2::theme_classic(base_size = base, base_family = "Arial") +
    ggplot2::theme(
      panel.background    = ggplot2::element_rect(fill = "white", color = NA),
      plot.background     = ggplot2::element_rect(fill = "white", color = NA),
      plot.title          = ggplot2::element_text(
        size = 7, face = "plain", hjust = 0, color = COL_TEXT,
        margin = ggplot2::margin(0, 0, 1, 13)),   # 左 13pt 给 panel letter 留位
      plot.subtitle       = ggplot2::element_text(
        size = 5.8, face = "italic", hjust = 0, color = COL_SUB,
        margin = ggplot2::margin(0, 0, 3, 13)),
      plot.title.position = "plot",
      axis.title.y        = ggplot2::element_text(size = 6.8, color = COL_TEXT),
      axis.title.x        = ggplot2::element_text(size = 6.8, color = COL_TEXT),
      axis.text           = ggplot2::element_text(size = 6.2, color = COL_TEXT),
      axis.line           = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.ticks          = ggplot2::element_line(linewidth = 0.3, color = "black"),
      axis.ticks.length   = ggplot2::unit(0.7, "mm"),
      strip.background    = ggplot2::element_rect(fill = "#F5F5F5", color = NA),
      strip.text          = ggplot2::element_text(size = 6.5, color = COL_TEXT,
                                                  margin = ggplot2::margin(2, 0, 2, 0)),
      legend.text         = ggplot2::element_text(size = 6.0, color = COL_TEXT),
      legend.title        = ggplot2::element_text(size = 6.0, color = COL_TEXT),
      legend.background   = ggplot2::element_rect(fill = "white", color = NA),
      legend.key.size     = ggplot2::unit(3.2, "mm"),
      legend.position     = "bottom",
      legend.margin       = ggplot2::margin(1, 0, 0, 0),
      plot.margin         = ggplot2::margin(6, 8, 2, 8, "pt")
    )
}

# ---- 生物通路分组（与主 Python 脚本 target_group 一致，first-match 优先）----
FIG3_GROUP_ORDER <- c("BCR/BTK", "JAK/STAT", "MAPK/stress", "NFkB",
                      "AKT/mTOR/S6", "cell cycle/DNA", "adhesion", "other")
# 填充用（低饱和，沿用主脚本 PATHWAY_COLORS）
FIG3_GROUP_FILL <- c(
  "BCR/BTK" = "#92B1D9", "JAK/STAT" = "#C1D8E9", "MAPK/stress" = "#D4A56B",
  "NFkB" = "#C97064", "AKT/mTOR/S6" = "#6CBFB5", "cell cycle/DNA" = "#9C8FC4",
  "adhesion" = "#A8A8A8", "other" = "#D4D4D4")
# 文字用（加深，保证轴标签可读）
FIG3_GROUP_TEXT <- c(
  "BCR/BTK" = "#3F6FB0", "JAK/STAT" = "#6FA0C8", "MAPK/stress" = "#B07A2E",
  "NFkB" = "#B0493B", "AKT/mTOR/S6" = "#2E8C80", "cell cycle/DNA" = "#6E5BA6",
  "adhesion" = "#777777", "other" = "#9A9A9A")

fig3_target_group <- function(target) {
  one <- function(tt) {
    t <- toupper(tt)
    anyx <- function(v) any(vapply(v, function(p) grepl(p, t, fixed = TRUE), logical(1)))
    if (anyx(c("BTK","SYK","BLNK","CD79","PLCG","LCK","ZAP70","LAT","LCP2","SRC"))) return("BCR/BTK")
    if (anyx(c("STAT","JAK"))) return("JAK/STAT")
    if (anyx(c("MAPK","MAP2K","JUN","FOS","JNK","P44_42"))) return("MAPK/stress")
    if (anyx(c("RELA","P-P65","IKK","IRAK","NFKB"))) return("NFkB")
    if (anyx(c("RPS6","AKT","TOR","EIF4","PDPK","AMPK","NDRG"))) return("AKT/mTOR/S6")
    if (anyx(c("CDK","RB","HISTON","HISTONE","H2AFX","H3"))) return("cell cycle/DNA")
    if (anyx(c("CTNND"))) return("adhesion")
    "other"
  }
  vapply(as.character(target), one, character(1), USE.NAMES = FALSE)
}

# readout 短名清理（与 Python short_label 一致的核心替换）
fig3_short <- function(x, n = 20) {
  s <- as.character(x)
  s <- gsub("_pSitePending", "", s)
  s <- gsub("_p_site_pending", "", s, ignore.case = TRUE)
  s <- gsub("MAPK1_MAPK3", "ERK1/2", s)
  s <- gsub("RPS6_pSitePending", "RPS6", s)
  s <- gsub("RELA_pSitePending_93H1", "RELA_93H1", s)
  ifelse(nchar(s) > n, paste0(substr(s, 1, n - 1), "."), s)
}

# 科学计数法 → Unicode 上标字符串，如 1.2×10⁻¹⁰（用于图内注标 p 值）
fig3_sci10 <- function(p, digits = 1) {
  p <- as.numeric(p)
  e <- floor(log10(p)); m <- p / 10^e
  sup <- c("0"="⁰","1"="¹","2"="²","3"="³","4"="⁴",
           "5"="⁵","6"="⁶","7"="⁷","8"="⁸","9"="⁹",
           "-"="⁻")
  es <- paste(sup[strsplit(as.character(e), "")[[1]]], collapse = "")
  sprintf(paste0("%.", digits, "f×10%s"), m, es)
}

# 显著性星号（备用）
fig3_sig <- function(p) {
  ifelse(is.na(p), "",
  ifelse(p < 1e-4, "****",
  ifelse(p < 1e-3, "***",
  ifelse(p < 1e-2, "**",
  ifelse(p < 0.05, "*", "ns")))))
}
