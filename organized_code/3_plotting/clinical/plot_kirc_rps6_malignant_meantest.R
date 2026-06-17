#!/usr/bin/env Rscript
# =============================================================================
# KIRC SC RPS6 validation — malignant-cell enrichment, mean-based + tests
# 诚实口径: 信号集中在 inferred malignant cells; bulk tumor 整体均值升高但患者级
#           被非恶性细胞组成/患者异质性稀释; 只有限定 malignant 后患者级配对才显著。
# 三面板: a 细胞级 4 组分布 / b 患者级配对斜线 / c 效应量森林
# 全英文 / Arial。所有字体 >=6pt（几何文字 size>=2.3≈6.5pt, 主题文字>=7pt）。
# 数值全部读自既有表, 仅患者级配对线由 cell 表聚合(并回写源表)。
# =============================================================================
suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(patchwork); library(scales)
})

ROOT <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682_SC11/04_figure_source_data/kirc_rps6_single_cell_validation_v1"
TAB <- file.path(ROOT, "tables"); FIG <- file.path(ROOT, "figures"); LOG <- file.path(ROOT, "scripts")
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)
FAM <- "Arial"; INK <- "#2B2B2B"; SUB <- "#555555"
COL_HEALTHY <- "#92B1D9"; COL_TUMALL <- "#9AA7B5"; COL_NONMAL <- "#C1D8E9"; COL_MALIG <- "#E8927A"
GREY <- "#BFBFBF"
# pt -> ggplot geom size (mm): size = pt / 2.845
PT <- function(pt) pt / 2.845

lc <- file(file.path(LOG, "plot_meantest.log"), "w"); L <- function(...) { writeLines(paste0(...), lc); flush(lc) }

cell <- fread(file.path(TAB, "kirc_cell_rps6_prediction.tsv"))
mt   <- fread(file.path(TAB, "kirc_rps6_tumor_healthy_mean_tests.tsv"))
VAL <- "predicted_RPS6_pS235_S236"
setnames(cell, VAL, "val")

cell[, tissue := as.character(tissue)]
cell[, malignant_status := as.character(malignant_status)]
healthy <- cell[tissue == "Healthy"][, grp := "Healthy\nall cells"]
tum_all <- cell[tissue == "Tumor"][, grp := "Tumor\nall cells"]
tum_non <- cell[tissue == "Tumor" & malignant_status == "non_malignant_inferred"][, grp := "Tumor\nnon-malignant"]
tum_mal <- cell[tissue == "Tumor" & malignant_status == "malignant_inferred"][, grp := "Tumor\nmalignant"]
A <- rbindlist(list(healthy, tum_all, tum_non, tum_mal), use.names = TRUE)
glev <- c("Healthy\nall cells", "Tumor\nall cells", "Tumor\nnon-malignant", "Tumor\nmalignant")
A[, grp := factor(grp, levels = glev)]
gcols <- setNames(c(COL_HEALTHY, COL_TUMALL, COL_NONMAL, COL_MALIG), glev)

gstat <- A[, .(n = .N, mean = mean(val), median = median(val)), by = grp]
L("=== cell-level group stats ==="); for (i in seq_len(nrow(gstat)))
  L(sprintf("  %-22s n=%d mean=%.2f median=%.2f", gsub("\n"," ",as.character(gstat$grp[i])), gstat$n[i], gstat$mean[i], gstat$median[i]))

yr <- quantile(A$val, c(0.005, 0.99), na.rm = TRUE)

# ---- base theme: 所有文字 >=6pt; 边距收紧便于后续组图 ----------------------
bt <- theme_minimal(base_family = FAM, base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        axis.title = element_text(size = 9.5, color = INK),
        axis.text  = element_text(size = 8, color = INK),
        plot.title = element_text(face = "bold", size = 11.5, color = INK),
        plot.subtitle = element_text(size = 8.5, color = SUB),
        plot.title.position = "plot",
        plot.margin = margin(3, 4, 3, 4))

pf <- function(p) vapply(p, function(x) if (is.na(x)) "p=NA" else if (x < 1e-4) sprintf("p=%.0e", x)
                         else if (x < 1e-3) sprintf("p=%.1e", x) else sprintf("p=%.3f", x), character(1))

# ============================== Panel A ======================================
mt_cell <- mt[level == "cell"]
p_mal_heal <- mt_cell[comparison == "Tumor malignant cells vs Healthy all cells", welch_p]
p_mal_non  <- mt_cell[comparison == "Tumor malignant cells vs Tumor non-malignant cells", welch_p]
xidx <- setNames(seq_along(glev), glev)

# 长尾极重(~1.5% 细胞 >200); 按箱线须顶 Q3+1.5*IQR 定上界, 裁掉稀疏长尾的大片留白
cap <- A[, { q <- quantile(val, c(.25, .75), na.rm = TRUE); iqr <- q[2] - q[1]
             max(val[val <= q[2] + 1.5 * iqr]) }, by = grp][, max(V1)]
body_hi <- cap
y_br2 <- cap * 1.04; y_br1 <- cap * 1.18               # 两条 p 桥接
y_mean <- cap * 1.36; y_top <- cap * 1.40              # mean/n 标注基线 + y 轴上界

pA <- ggplot(A, aes(grp, val, fill = grp)) +
  geom_violin(width = 0.85, alpha = 0.32, color = NA, scale = "width") +
  geom_boxplot(width = 0.15, outlier.shape = NA, fill = "white", color = INK, linewidth = 0.4, alpha = 0.95) +
  stat_summary(fun = mean, geom = "point", shape = 23, size = 3.0, fill = "white", color = INK, stroke = 0.55) +
  geom_text(data = gstat, aes(grp, y = y_mean, label = sprintf("mean %.1f\nn=%s", mean, comma(n))),
            inherit.aes = FALSE, family = FAM, size = PT(7.5), color = INK, vjust = 1, lineheight = 0.85) +
  scale_fill_manual(values = gcols, guide = "none") +
  coord_cartesian(ylim = c(yr[1], y_top)) +
  labs(title = "Cell-level predicted RPS6 pS235/S236",
       subtitle = "Inferred malignant cells carry the strongest predicted phospho-state",
       x = NULL, y = "Predicted RPS6 pS235/S236") +
  bt + theme(axis.text.x = element_text(size = 8.5, lineheight = 0.9))

br <- function(x1, x2, y, lab) list(
  annotate("segment", x = x1, xend = x2, y = y, yend = y, color = INK, linewidth = 0.35),
  annotate("segment", x = x1, xend = x1, y = y, yend = y - body_hi*0.02, color = INK, linewidth = 0.35),
  annotate("segment", x = x2, xend = x2, y = y, yend = y - body_hi*0.02, color = INK, linewidth = 0.35),
  annotate("text", x = (x1+x2)/2, y = y, label = lab, vjust = -0.45, family = FAM, size = PT(7.5), color = INK))
pA <- pA +
  br(xidx["Healthy\nall cells"], xidx["Tumor\nmalignant"], y_br1, paste0("malignant vs Healthy   ", pf(p_mal_heal))) +
  br(xidx["Tumor\nnon-malignant"], xidx["Tumor\nmalignant"], y_br2, paste0("malignant vs non-malignant   ", pf(p_mal_non)))

# ============================== Panel B ======================================
agg <- function(dt) dt[, .(m = mean(val)), by = patient_id]
heal_p <- agg(healthy); tum_p <- agg(tum_all); mal_p <- agg(tum_mal)
mk_pair <- function(tumdt, lab) {
  m <- merge(heal_p[, .(patient_id, Healthy = m)], tumdt[, .(patient_id, Tumor = m)], by = "patient_id")
  m[, comparison := lab]; m[, dir := ifelse(Tumor >= Healthy, "up", "down")]; m
}
pair_all <- mk_pair(tum_p, "All tumor cells"); pair_mal <- mk_pair(mal_p, "Malignant tumor cells")
B <- rbindlist(list(pair_all, pair_mal))
Blong <- melt(B, id.vars = c("patient_id","comparison","dir"), measure.vars = c("Healthy","Tumor"),
              variable.name = "scope", value.name = "val")
Blong[, scope := factor(scope, levels = c("Healthy","Tumor"))]
Blong[, comparison := factor(comparison, levels = c("All tumor cells","Malignant tumor cells"))]

L(""); L("=== patient-paired check ===")
for (cmp in levels(Blong$comparison)) { sub <- B[comparison == cmp]; pt <- t.test(sub$Tumor, sub$Healthy, paired = TRUE)
  L(sprintf("  %s n=%d shift=%.2f p=%.4f up=%d/%d", cmp, nrow(sub), mean(sub$Tumor-sub$Healthy), pt$p.value, sum(sub$dir=="up"), nrow(sub))) }
p_paired_all <- mt[comparison == "Patient paired mean: Tumor all cells vs Healthy all cells", paired_p]
p_paired_mal <- mt[comparison == "Patient paired mean: Tumor malignant cells vs Healthy all cells", paired_p]
plab <- data.table(comparison = factor(c("All tumor cells","Malignant tumor cells"), levels = c("All tumor cells","Malignant tumor cells")),
                   lab = c(paste0("paired ", pf(p_paired_all), "  (n=8)"), paste0("paired ", pf(p_paired_mal), "  (n=8)")))

pB <- ggplot(Blong, aes(scope, val, group = patient_id, color = dir)) +
  geom_line(linewidth = 0.7, alpha = 0.85) +
  geom_point(size = 2.5, alpha = 0.9) +
  facet_wrap(~ comparison, nrow = 1) +
  scale_color_manual(values = c(up = COL_MALIG, down = COL_HEALTHY),
                     labels = c(up = "Tumor > Healthy", down = "Tumor < Healthy"), name = NULL) +
  geom_text(data = plab, aes(x = 1.5, y = Inf, label = lab), inherit.aes = FALSE,
            family = FAM, size = PT(8), color = INK, vjust = 1.5) +
  labs(title = "Patient-level paired means",
       subtitle = "Tumor-vs-normal rises consistently only after restricting to malignant cells",
       x = NULL, y = "Patient mean predicted\nRPS6 pS235/S236") +
  bt + theme(legend.position = "bottom", legend.text = element_text(size = 8.5),
             legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(2, "pt"),
             strip.text = element_text(face = "bold", size = 10, color = INK),
             panel.spacing = unit(7, "pt"), panel.grid.major.x = element_blank())

# ============================== Panel C ======================================
fr_rows <- list(
  list(cmp="Tumor all cells vs Healthy all cells", lvl="cell", lab="All tumor - Healthy", sub="cell level", tcol="welch_t", pcol="welch_p", df=Inf),
  list(cmp="Patient paired mean: Tumor all cells vs Healthy all cells", lvl="patient_paired", lab="All tumor - Healthy", sub="patient (paired)", tcol="paired_t", pcol="paired_p", df=7),
  list(cmp="Tumor malignant cells vs Healthy all cells", lvl="cell", lab="Malignant - Healthy", sub="cell level", tcol="welch_t", pcol="welch_p", df=Inf),
  list(cmp="Patient paired mean: Tumor malignant cells vs Healthy all cells", lvl="patient_paired", lab="Malignant - Healthy", sub="patient (paired)", tcol="paired_t", pcol="paired_p", df=7),
  list(cmp="Tumor malignant cells vs Tumor non-malignant cells", lvl="cell", lab="Malignant - non-malignant", sub="cell level", tcol="welch_t", pcol="welch_p", df=Inf))
C <- rbindlist(lapply(fr_rows, function(r) {
  row <- mt[comparison == r$cmp & level == r$lvl]
  shift <- row$mean_shift_a_minus_b; tval <- as.numeric(row[[r$tcol]]); p <- as.numeric(row[[r$pcol]])
  se <- abs(shift / tval); crit <- if (is.finite(r$df)) qt(0.975, r$df) else 1.96
  data.table(label = r$lab, sub = r$sub, shift = shift, lo = shift - crit*se, hi = shift + crit*se, p = p, sig = p < 0.05)
}), use.names = TRUE)
C[, row_lab := paste0(label, "\n", sub)]
C[, row_lab := factor(row_lab, levels = rev(row_lab))]
C[, fillc := ifelse(sig, COL_MALIG, GREY)]
C[, plab := ifelse(sig, paste0(pf(p), " *"), pf(p))]

pC <- ggplot(C, aes(shift, row_lab)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#888888", linewidth = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.22, linewidth = 0.45, color = INK) +
  geom_point(aes(fill = fillc), shape = 21, size = 3.2, color = "white", stroke = 0.55) +
  geom_text(aes(x = hi, label = plab), hjust = -0.12, family = FAM, size = PT(8), color = INK) +
  scale_fill_identity() +
  scale_x_continuous(expand = expansion(mult = c(0.04, 0.30))) +
  labs(title = "Effect size across analysis levels",
       subtitle = "Mean shift (A - B), 95% CI; patient-level significance only for malignant cells",
       x = "Mean shift in predicted RPS6 pS235/S236", y = NULL) +
  bt + theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(size = 8, lineheight = 0.9))

# ============================== 独立导出三张图 (不拼组) ======================
# 每张单独 ggsave, 各自尺寸, 字号自然放大; 同时各出一张 <2000px 预览供目检。
save3 <- function(plot, stem, w, h) {
  pdf <- tryCatch({ ggsave(file.path(FIG, paste0(stem,".pdf")), plot, width=w, height=h, device=cairo_pdf); "ok" }, error=function(e) paste("ERR",conditionMessage(e)))
  png <- tryCatch({ if (requireNamespace("ragg",quietly=TRUE)) ggsave(file.path(FIG, paste0(stem,".png")), plot, width=w, height=h, dpi=400, device=ragg::agg_png) else ggsave(file.path(FIG, paste0(stem,".png")), plot, width=w, height=h, dpi=400, type="cairo"); "ok" }, error=function(e) paste("ERR",conditionMessage(e)))
  svg <- tryCatch({ if (requireNamespace("svglite",quietly=TRUE)) ggsave(file.path(FIG, paste0(stem,".svg")), plot, width=w, height=h, device=svglite::svglite) else ggsave(file.path(FIG, paste0(stem,".svg")), plot, width=w, height=h); "ok" }, error=function(e) paste("ERR",conditionMessage(e)))
  # 预览: 限制最长边 <2000px. 取 dpi 使 max(w,h)*dpi <= 1900
  prev_dpi <- floor(1900 / max(w, h))
  ggsave(file.path(FIG, paste0("_preview_", stem, ".png")), plot, width=w, height=h, dpi=prev_dpi, bg="white")
  L(sprintf("%-40s pdf=%s png=%s svg=%s (prev_dpi=%d)", stem, pdf, png, svg, prev_dpi))
  c(pdf=pdf, png=png, svg=svg)
}

save3(pA, "kirc_rps6_celltype_meantest",        w = 6.6, h = 4.6)
save3(pB, "kirc_rps6_patient_paired_slopes",    w = 6.0, h = 4.2)
save3(pC, "kirc_rps6_effectsize_forest",        w = 6.4, h = 3.6)

fwrite(B, file.path(TAB, "kirc_rps6_patient_paired_means_for_plot.tsv"), sep = "\t")
L(""); L("ALL DONE (3 independent panels, no composite)")
close(lc)
cat("DONE — 3 independent panels written\n")
