#!/usr/bin/env Rscript
# =============================================================================
# Fig 5a  Pan-cancer entry map  (PLOT ONLY)
# 数据由 panel_a_cancer_entry_data.py 生成，本脚本不重算任何数值，只读表绘图。
# 结构: 顶部双层柱(+内嵌三色 mini 图例) + 中部数值矩阵 + 右侧全局总量条(带单位) + 底部图例
# 无标题。行标签 = 左侧大字号文字(按 direction 上色)。简化行名。
# 矩阵数字: k 缩写 + 浅色小值隐藏。右侧总量带单位(genes/sites/modules/targets)。
# 全英文 / Arial。列序 = cancer_order, 行序 = row_order, 单元格色 = direction。
# =============================================================================
suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(patchwork); library(scales)
})

TAB <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/source_data/tables"
FIG <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/figures"
LOG <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts"
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
FAM <- "Arial"

top <- fread(file.path(TAB, "panel_a_cancer_entry_top_bars.tsv"))
mat <- fread(file.path(TAB, "panel_a_cancer_entry_matrix_long.tsv"))
rt  <- fread(file.path(TAB, "panel_a_cancer_entry_row_totals.tsv"))

# ---- 剔除全零癌种 (12 行合计为 0 的列, 如 CHOL/DLBC/PCPG/THYM) ---------------
col_sum <- mat[, .(s = sum(value)), by = cancer_short]
zero_cancers <- col_sum[s == 0, cancer_short]
if (length(zero_cancers)) {
  mat <- mat[!cancer_short %in% zero_cancers]
  top <- top[!cancer_short %in% zero_cancers]
  message("dropped all-zero cancers: ", paste(zero_cancers, collapse = ", "))
}

# ---- 显示名(压缩) + 单位 ----------------------------------------------------
disp_map <- c(
  r01 = "Parent mRNA clinical genes",
  r02 = "All clinical predicted sites",
  r03 = "Sites beyond parent mRNA, risk",
  r04 = "Sites beyond parent mRNA, protective",
  r05 = "Graph-residual sites, risk",
  r06 = "Graph-residual sites, protective",
  r07 = "Beyond parent mRNA + graph residual",
  r08 = "Sample-graph modules",
  r09 = "Site-graph modules",
  r10 = "Multi-site residual genes",
  r11 = "Strict external anchors",
  r12 = "Confirmed external anchors")
unit_map <- c(r01="genes", r02="sites", r03="sites", r04="sites", r05="sites",
              r06="sites", r07="sites", r08="modules", r09="modules",
              r10="genes", r11="targets", r12="targets")

# ---- 数字格式: >=1000 用 k 缩写 ---------------------------------------------
fmtk <- function(v) ifelse(v >= 1000, paste0(formatC(v/1000, format="f", digits=1), "k"),
                           formatC(v, format="d", big.mark=""))

# ---- shared orders (用 row_id 对齐矩阵/标签/总量三块) ------------------------
clev <- top[order(cancer_order), cancer_short]
rid_order <- rt[order(row_order), row_id]                   # top -> bottom
ylev <- rev(rid_order)                                      # ggplot bottom -> top
top[, cancer_short := factor(cancer_short, levels = clev)]
mat[, cancer_short := factor(cancer_short, levels = clev)]
mat[, yfac := factor(row_id, levels = ylev)]
rt[,  yfac := factor(row_id, levels = ylev)]

# ---- palette : cell fills (by direction) ------------------------------------
GREY_CTX <- "#D4D4D4"; INK <- "#2B2B2B"
R_risk <- colorRamp(c("#FCEAE2","#F6C8B6","#E8927A"))
R_prot <- colorRamp(c("#EAF1F7","#C1D8E9","#92B1D9"))
R_cnt  <- colorRamp(c("#EEEEF5","#DBDDEF","#9092C4"))
hex_for <- function(direction, v) {
  v <- pmax(0, pmin(1, v))^0.72
  f <- switch(direction, risk = R_risk, protective = R_prot, R_cnt)
  m <- f(v); rgb(m[,1], m[,2], m[,3], maxColorValue = 255)
}
base_risk <- "#E8927A"; base_prot <- "#92B1D9"; base_cnt <- "#9092C4"
dir_base  <- function(d) switch(d, risk = base_risk, protective = base_prot, base_cnt)
lab_risk <- "#C0623F"; lab_prot <- "#3F6FA0"; lab_cnt <- INK
dir_lab  <- function(d) switch(d, risk = lab_risk, protective = lab_prot, lab_cnt)

# ---- matrix fills / labels --------------------------------------------------
WHITE_ZERO <- "#FAFAFB"      # 真零格子留白(极浅, 仅勾出网格)
mat[, vmax := max(value), by = row_id]
mat[, v01  := ifelse(vmax > 0, value / vmax, 0)]
# 真零格子留白; 非零格子按 direction + 行内归一化上色
mat[, fill := ifelse(value <= 0, WHITE_ZERO, hex_for(direction[1], v01)), by = .(row_id, cancer_short)]
mat[, txt_col := ifelse(v01 > 0.55, "white", INK)]
# 数字: 仅显示绝对值 >=50 的格子 (其余留色块不写数字, 减少拥挤)
mat[, show := value >= 50]
mat[, lab := ifelse(show, fmtk(value), "")]

bt <- theme_minimal(base_family = FAM, base_size = 11) +
  theme(panel.grid = element_blank(),
        axis.title = element_text(size = 11, color = INK),
        axis.text  = element_text(color = INK),
        plot.margin = margin(2, 2, 2, 2))

# ---- p_mat ------------------------------------------------------------------
p_mat <- ggplot(mat, aes(cancer_short, yfac)) +
  geom_tile(aes(fill = fill), color = "white", linewidth = 0.5) +
  geom_text(aes(label = lab, color = txt_col), size = 2.5, family = FAM) +
  scale_fill_identity() + scale_color_identity() +
  scale_x_discrete(position = "bottom", expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(x = NULL, y = NULL) + bt +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1, size = 10, face = "bold"),
        axis.text.y = element_blank(), axis.ticks.y = element_blank())

grp_td <- rt[order(row_order), .N, by = row_group][, N]
bounds <- head(cumsum(rev(grp_td)), -1) + 0.5
p_mat <- p_mat + geom_hline(yintercept = bounds, color = "white", linewidth = 1.8)

# ---- p_rowlab : 左侧大字号行标签 --------------------------------------------
wrap2max <- function(s, width = 40) vapply(s, function(x) {
  if (nchar(x) <= 34) return(x)
  w <- strwrap(x, width = width)
  if (length(w) > 2) w <- c(paste(w[seq_len(length(w)-1)], collapse = " "), w[length(w)])
  paste(w, collapse = "\n")
}, character(1))
labd <- rt[, .(row_id, direction, row_order)]
labd[, yfac := factor(row_id, levels = ylev)]
labd[, disp := wrap2max(disp_map[row_id], 40)]
labd[, col  := sapply(direction, dir_lab)]
p_rowlab <- ggplot(labd, aes(x = 1, y = yfac)) +
  geom_text(aes(label = disp, color = col), hjust = 1, size = 3.75, family = FAM, lineheight = 0.9) +
  scale_color_identity() +
  scale_y_discrete(expand = c(0, 0)) +
  scale_x_continuous(limits = c(0, 1.04), expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  theme_void(base_family = FAM) + theme(plot.margin = margin(2, 3, 2, 2))

# ---- p_top : grey clinical_total + stacked beyond-mRNA + 内嵌 mini 图例 ------
top_by <- melt(top[, .(cancer_short, Risk = site_over_mrna_risk, Protective = site_over_mrna_protective)],
               id.vars = "cancer_short", variable.name = "effect", value.name = "value")
top_by[, effect := factor(effect, levels = c("Protective", "Risk"))]
ymax_top <- max(top$clinical_total)
my  <- ymax_top * c(0.98, 0.86, 0.74)        # 三行图例 y (右上空区)
mcol <- c(GREY_CTX, base_risk, base_prot)
mlab <- c("All clinical sites", "Beyond parent mRNA, risk", "Beyond parent mRNA, protective")
rh  <- ymax_top * 0.04
ncol_top <- nrow(top)                         # 实际列数(剔零后)
xsq <- ncol_top * 0.62                         # 方块 x: 右侧矮柱区, 随列数自适应
p_top <- ggplot() +
  geom_col(data = top, aes(cancer_short, clinical_total), fill = GREY_CTX, width = 0.84) +
  geom_col(data = top_by, aes(cancer_short, value, fill = effect), width = 0.48, position = position_stack()) +
  scale_fill_manual(values = c(Risk = base_risk, Protective = base_prot), guide = "none") +
  annotate("rect", xmin = xsq, xmax = xsq+0.75, ymin = my-rh, ymax = my+rh, fill = mcol) +
  annotate("text", x = xsq+1.1, y = my, label = mlab, hjust = 0, size = 2.7, family = FAM, color = INK) +
  scale_y_continuous(labels = fmtk, expand = expansion(mult = c(0, 0.04))) +
  scale_x_discrete(expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = "Clinical sites") + bt +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        axis.text.y = element_text(size = 9), axis.title.y = element_text(size = 10))

# ---- p_right : per-row pan-cancer totals (带单位) ----------------------------
rt[, basecol := sapply(direction, dir_base)]
rt[, rlab := paste0(fmtk(total_value), " ", unit_map[row_id])]
p_right <- ggplot(rt, aes(total_value, yfac)) +
  geom_col(aes(fill = basecol), width = 0.72) +
  geom_text(aes(label = rlab), hjust = -0.05, size = 2.75, family = FAM, color = INK) +
  scale_fill_identity() +
  scale_x_sqrt(expand = expansion(mult = c(0, 0.62))) +
  scale_y_discrete(expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  labs(x = "Pan-cancer total", y = NULL) + bt +
  theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        axis.text.x = element_blank(), axis.title.x = element_text(size = 9.5))

# ---- p_leg : bottom legend = 矩阵格子配色含义(顶部已解释柱图三色, 此处专职矩阵) --
leg <- data.table(x = factor(1:3),
  lab = c("Risk effect (HR > 1)", "Protective effect (HR < 1)", "Count / module"),
  mid = c(base_risk, base_prot, base_cnt))
p_leg <- ggplot(leg, aes(x, 1)) +
  geom_tile(aes(fill = mid), width = 0.24, height = 0.55, color = "white") +
  geom_text(aes(label = lab), nudge_y = -0.78, size = 3.0, family = FAM, color = INK) +
  scale_fill_identity() +
  scale_x_discrete(expand = expansion(add = c(0.5, 0.5))) +
  scale_y_continuous(limits = c(0, 1.5), expand = c(0, 0)) +
  coord_cartesian(clip = "off") + theme_void(base_family = FAM) +
  theme(plot.margin = margin(2, 40, 4, 40))

# ---- compose (no title) -----------------------------------------------------
fig <- wrap_plots(plot_spacer(), p_top, plot_spacer(), p_rowlab, p_mat, p_right, p_leg,
                  design = "ABC\nDEF\nGGG",
                  widths = c(0.32, 1, 0.21), heights = c(0.24, 1, 0.085))

# ---- save -------------------------------------------------------------------
W <- 16.8; H <- 8.8
msg <- c()
msg["pdf"] <- tryCatch({ ggsave(file.path(FIG,"panel_a_cancer_entry_map.pdf"), fig, width=W, height=H, device=cairo_pdf); "ok" },
                       error=function(e) tryCatch({ ggsave(file.path(FIG,"panel_a_cancer_entry_map.pdf"), fig, width=W, height=H); "ok-fallback" }, error=function(e2) paste("ERR", conditionMessage(e2))))
msg["png"] <- tryCatch({ if (requireNamespace("ragg", quietly=TRUE)) ggsave(file.path(FIG,"panel_a_cancer_entry_map.png"), fig, width=W, height=H, dpi=400, device=ragg::agg_png) else ggsave(file.path(FIG,"panel_a_cancer_entry_map.png"), fig, width=W, height=H, dpi=400, type="cairo"); "ok" },
                       error=function(e) paste("ERR", conditionMessage(e)))
msg["svg"] <- tryCatch({ if (requireNamespace("svglite", quietly=TRUE)) ggsave(file.path(FIG,"panel_a_cancer_entry_map.svg"), fig, width=W, height=H, device=svglite::svglite) else ggsave(file.path(FIG,"panel_a_cancer_entry_map.svg"), fig, width=W, height=H); "ok" },
                       error=function(e) paste("ERR", conditionMessage(e)))
writeLines(c(paste("pdf:", msg["pdf"]), paste("png:", msg["png"]), paste("svg:", msg["svg"]),
             paste("n_cancers:", length(clev)), paste("n_rows:", nrow(rt)),
             paste("hidden_cells:", mat[show == FALSE & value > 0, .N], "of", mat[value > 0, .N], "nonzero")),
           file.path(LOG, "panel_a_cancer_entry_plot.log"))
cat("PLOT_DONE pdf=", msg["pdf"], " png=", msg["png"], " svg=", msg["svg"], "\n", sep="")
