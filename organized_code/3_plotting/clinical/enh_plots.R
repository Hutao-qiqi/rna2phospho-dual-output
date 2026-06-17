#!/usr/bin/env Rscript
# 两个增强分析图(各自独立, 不拼组):
#  enh1: 恶性细胞 predicted pS6 vs angiogenesis 程序 (诚实修正版: phospho 抓到、转录没抓到)
#  enh2: pathway_family x 方向 富集 (增殖=RISK vs 信号/免疫/mTOR=PROTECTIVE), 双口径
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(scales) })

SC  <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/sc_kirc_rps6_validation/tables"
TAB <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/source_data/tables"
FIG <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/figures/enhancements"
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
FAM <- "Arial"; INK <- "#2B2B2B"; SUB <- "#555555"
COL_MALIG <- "#E8927A"; COL_BLUE <- "#92B1D9"; GREY <- "#BFBFBF"
PT <- function(pt) pt / 2.845

save3 <- function(p, stem, w, h) {
  tryCatch(ggsave(file.path(FIG, paste0(stem,".pdf")), p, width=w, height=h, device=cairo_pdf), error=function(e) ggsave(file.path(FIG, paste0(stem,".pdf")), p, width=w, height=h))
  tryCatch({ if (requireNamespace("ragg",quietly=TRUE)) ggsave(file.path(FIG, paste0(stem,".png")), p, width=w, height=h, dpi=400, device=ragg::agg_png) else ggsave(file.path(FIG, paste0(stem,".png")), p, width=w, height=h, dpi=400) }, error=function(e) message("png err ",conditionMessage(e)))
  tryCatch({ if (requireNamespace("svglite",quietly=TRUE)) ggsave(file.path(FIG, paste0(stem,".svg")), p, width=w, height=h, device=svglite::svglite) else ggsave(file.path(FIG, paste0(stem,".svg")), p, width=w, height=h) }, error=function(e) message("svg err ",conditionMessage(e)))
  pv <- floor(1900/max(w,h)); ggsave(file.path(FIG, paste0("_preview_",stem,".png")), p, width=w, height=h, dpi=pv, bg="white")
}

bt <- theme_minimal(base_family=FAM, base_size=12) +
  theme(panel.grid.minor=element_blank(), plot.title=element_text(face="bold",size=11.5,color=INK),
        plot.subtitle=element_text(size=8.5,color=SUB), axis.title=element_text(size=9.5,color=INK),
        axis.text=element_text(size=8,color=INK), plot.title.position="plot",
        plot.margin=margin(3,5,3,5))

# ============ enh1: pS6 vs angiogenesis (malignant cells) ============
cell <- fread(file.path(SC, "kirc_cell_rps6_prediction.tsv"))
mal <- cell[malignant_status == "malignant_inferred"]
setnames(mal, "predicted_RPS6_pS235_S236", "pS6")
r_ang  <- cor(mal$pS6, mal$angiogenesis_score, method="spearman", use="complete.obs")
r_mtor <- cor(mal$mTOR_S6_score, mal$angiogenesis_score, method="spearman", use="complete.obs")
ycap <- as.numeric(quantile(mal$pS6, 0.985, na.rm=TRUE))
xr <- range(mal$angiogenesis_score, na.rm=TRUE)
p1 <- ggplot(mal, aes(angiogenesis_score, pS6)) +
  geom_hex(bins = 55) +
  scale_fill_gradientn(colours = c("#EAF1F7","#C1D8E9","#92B1D9","#5E86B8"), name="cells", trans="log10") +
  geom_smooth(method="lm", se=FALSE, color=COL_MALIG, linewidth=0.8) +
  annotate("text", x=xr[1], y=ycap*1.03, hjust=0, vjust=1, family=FAM, size=PT(8.5), color=INK,
           label=sprintf("Spearman ρ = %.2f  (n = %s malignant cells)", r_ang, comma(nrow(mal)))) +
  coord_cartesian(ylim=c(as.numeric(quantile(mal$pS6,0.01,na.rm=TRUE)), ycap*1.06)) +
  labs(title="Predicted pS6 tracks the angiogenesis program in malignant ccRCC cells",
       subtitle=sprintf("Predicted phospho-pS6 captures an angiogenesis link the transcript misses\n(mTOR-S6 transcriptional score vs angiogenesis: ρ = %.2f)", r_mtor),
       x="Angiogenesis program score", y="Predicted RPS6 pS235/S236") +
  bt + theme(legend.position=c(0.92,0.28), legend.key.size=unit(3,"mm"), legend.title=element_text(size=7), legend.text=element_text(size=6.5))
save3(p1, "enh1_pS6_angiogenesis_malignant", w=6.4, h=4.8)

# ============ enh2 (panel c): 24 生物学 pathway_module 方向 + 代表基因 ============
md <- fread(file.path(TAB, "fig5c_pathway_module_direction.tsv"))
md <- md[n_clin >= 30 & !is.na(rep3) & rep3 != ""]
md[, short := sub("^.*/ ", "", pathway_module)]
md[, short := factor(short, levels = md[order(risk_frac_clin), short])]
md[, dir2 := factor(ifelse(risk_frac_clin >= 0.5, "Risk-dominant (HR>1 majority)", "Protective-dominant (HR<1 majority)"),
                    levels = c("Risk-dominant (HR>1 majority)", "Protective-dominant (HR<1 majority)"))]
GLOB <- 0.363
p2 <- ggplot(md, aes(risk_frac_clin, short, fill = dir2)) +
  geom_col(width = 0.72) +
  geom_vline(xintercept = GLOB, linetype = "dashed", color = GREY, linewidth = 0.4) +
  geom_text(aes(label = sprintf("%d", n_clin)), hjust = -0.18, family = FAM, size = PT(6), color = INK) +
  geom_text(aes(x = 0.80, label = rep3), hjust = 0, family = FAM, size = PT(6.2), color = SUB) +
  scale_fill_manual(values = c("Risk-dominant (HR>1 majority)" = "#E41A1C",
                               "Protective-dominant (HR<1 majority)" = "#2166AC"), name = NULL) +
  scale_x_continuous(limits = c(0, 1.5), breaks = c(0, 0.25, 0.5, 0.75), expand = expansion(mult = c(0, 0))) +
  labs(title = "Biological identity and risk direction of pan-cancer phospho-modules",
       subtitle = "Proliferation (ribosome biogenesis, cell cycle) = RISK; mTOR-AKT (RICTOR/RPTOR, likely inhibitory sites) / junction / Rho-SRC = PROTECTIVE\nBH-FDR q<0.05; n by each bar; dashed = global risk fraction (0.36); right text = representative genes",
       x = "Risk fraction (HR>1 share of clinically significant sites)", y = NULL) +
  bt + theme(axis.text.y = element_text(size = 7.5), plot.subtitle = element_text(size = 7.5),
             panel.grid = element_blank(), legend.position = "bottom",
             legend.key.height = unit(3, "mm"), legend.key.width = unit(9, "mm"),
             legend.title = element_text(size = 7.5), legend.text = element_text(size = 7))
save3(p2, "enh2_pathway_module_direction", w = 9.0, h = 6.7)

cat("ENH DONE\n")
