# Fig 4 config — sci-plot publication style (clean / restrained)
# Font: Arial. Palette: sci-plot low-saturation. Theme: theme_minimal base.

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(stringr); library(forcats); library(scales); library(patchwork)
  library(viridisLite)
})

# ---------- paths ----------
SRC_DIR  <- "E:/data/gongke/TCGA-TCPA/02_results/figure_sources/20260528_fig4_locked_p100_v10b_cosine_direction"
TBL_DIR  <- file.path(SRC_DIR, "tables")
OUT_DIR  <- "E:/data/gongke/TCGA-TCPA/02_results/figure_outputs/fig4_v3"
PANEL_DIR <- file.path(OUT_DIR, "panels")
FINAL_DIR <- file.path(OUT_DIR, "final")
dir.create(PANEL_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FINAL_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(PANEL_DIR, "thumbs"), showWarnings = FALSE, recursive = TRUE)

# ---------- font ----------
if (.Platform$OS.type == "windows") {
  grDevices::windowsFonts(Arial = grDevices::windowsFont("Arial"))
}
BASE_FONT <- "Arial"
BASE_SIZE <- 8

# ---------- sci-plot palette ----------
PUB <- c("#92B1D9", "#C1D8E9", "#DBDDEF", "#F6C8B6", "#D4D4D4")

# drug class: blue -> lavender -> peach + grey (lightness-graded, 2 hue family)
DRUG_CLASS_EN <- c(
  "EGFR/HER2" = "EGFR/HER2", "ABL/SRC" = "ABL/SRC", "HDAC" = "HDAC",
  "mTOR" = "mTOR", "MEK" = "MEK", "蛋白酶体" = "Proteasome", "其他" = "Other",
  "EGFR" = "EGFR", "RAF/MEK" = "RAF/MEK", "VEGF" = "VEGF", "多激酶" = "Multi-kinase"
)
DRUG_CLASS_ORDER_EN <- c("EGFR/HER2","ABL/SRC","HDAC","mTOR","MEK","Proteasome","Other",
                         "EGFR","RAF/MEK","VEGF","Multi-kinase")
# distinguishable muted qualitative palette (Tableau-muted) for categorical
# drug-class points/annotations (used by panel c rain + panel e annotation)
DRUG_CLASS_COLORS <- c(
  "EGFR/HER2" = "#4E79A7", "ABL/SRC" = "#F28E2B", "HDAC" = "#59A14F",
  "mTOR" = "#E15759", "MEK" = "#B07AA1", "Proteasome" = "#B6992D", "Other" = "#BAB0AC",
  "EGFR" = "#86BCB6", "RAF/MEK" = "#D37295", "VEGF" = "#8CD17D", "Multi-kinase" = "#79706E"
)

# two-group metric palette
METRIC_EN     <- c("余弦" = "Cosine", "方向准确率" = "Direction acc.")
METRIC_COLORS <- c("Cosine" = "#92B1D9", "Direction acc." = "#F6C8B6")

# site set (lightness within blue)
SITE_SET_EN     <- c("all_sites" = "All sites", "responsive_top20" = "Responsive top 20%")
SITE_SET_COLORS <- c("All sites" = "#C1D8E9", "Responsive top 20%" = "#5A89B3")

# response
RESPONSE_EN     <- c("反应者" = "Responder", "非反应者" = "Non-responder")
RESPONSE_COLORS <- c("Responder" = "#F6C8B6", "Non-responder" = "#C1D8E9")

# score groups (panel h / old i)
SCORE_GROUP_COLORS <- c("SCP682-PPKO V10B" = "#5A89B3",
                        "PPKO" = "#5A89B3",
                        "Generic control" = "#D4D4D4")

# TCGA score label translations
SCORE_LABEL_EN <- c(
  "V10B 靶点先验相关位点预测扰动幅度" = "PPKO target-prior |Δ|",
  "V10B 预测前二百位点扰动幅度"       = "PPKO top-200 |Δ|",
  "V10B 外部观测位点扰动幅度"         = "PPKO observed-site |Δ|",
  "ppko_abs_delta_mean"                = "PPKO mean |Δ|",
  "手工靶点通路分数"                   = "Hand-curated pathway score",
  "全局磷酸化均值"                     = "Global phospho mean",
  "全局绝对磷酸化负荷"                 = "Global |phospho|",
  "映射磷酸化抗体均值"                 = "Mapped marker mean",
  "映射磷酸化抗体绝对均值"             = "Mapped marker |mean|",
  "靶点总蛋白均值"                     = "Target protein mean",
  "观测抗体数量"                       = "Observed marker count"
)

# continuous ramps
DIV_RAMP <- colorRampPalette(c("#92B1D9", "#DBDDEF", "#F6C8B6"))(100)
SEQ_RAMP <- viridisLite::viridis(100, option = "D", begin = 0.08, end = 0.95)
# standard high-contrast red-blue diverging (RdBu): low = blue, high = red
RDBU_RAMP <- colorRampPalette(c("#2166AC", "#4393C3", "#92C5DE", "#F7F7F7",
                                "#F4A582", "#D6604D", "#B2182B"))(100)

# ---------- clean publication theme (theme_minimal base) ----------
theme_fig4 <- function(base_size = BASE_SIZE) {
  theme_minimal(base_size = base_size, base_family = BASE_FONT) +
    theme(
      text             = element_text(family = BASE_FONT, color = "black"),
      plot.title       = element_text(size = base_size + 1, face = "bold", hjust = 0,
                                       margin = margin(b = 2)),
      plot.subtitle    = element_text(size = base_size - 1, color = "grey40",
                                       margin = margin(b = 4)),
      plot.tag         = element_text(size = base_size + 4, face = "bold"),
      axis.title       = element_text(size = base_size, face = "bold", color = "black"),
      axis.text        = element_text(size = base_size - 1, color = "grey20"),
      axis.line        = element_line(linewidth = 0.35, color = "grey30"),
      axis.ticks       = element_line(linewidth = 0.3, color = "grey30"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.25, color = "grey92"),
      legend.title     = element_text(size = base_size - 1, face = "bold"),
      legend.text      = element_text(size = base_size - 1),
      legend.key.size  = unit(0.32, "cm"),
      strip.text       = element_text(size = base_size, face = "bold"),
      plot.margin      = margin(6, 8, 6, 6)
    )
}

sig_stars <- function(p) {
  if (is.na(p)) "" else if (p < 0.001) "***" else if (p < 0.01) "**" else
    if (p < 0.05) "*" else "ns"
}

cat("config v4 (clean sci-plot) loaded\n")
