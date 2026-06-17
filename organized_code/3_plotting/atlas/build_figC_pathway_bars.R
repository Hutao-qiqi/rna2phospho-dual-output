#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(ggplot2)})
ROOT <- "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
K <- 30
enr  <- read.delim(sprintf("%s/results/module_pathway_enrichment_k%d.tsv",ROOT,K), stringsAsFactors=FALSE)

# 精选 9 个有代表性、通路显著的癌种特异模块(覆盖代谢/缺氧/免疫/EMT/激素/增殖)
sel     <- c("M4","M10","M17","M8","M25","M20","M9","M18","M28")
sel_lab <- c(M4="M4 · KICH", M10="M10 · LIHC", M17="M17 · KIRC", M8="M8 · DLBC",
             M25="M25 · SKCM", M20="M20 · MESO", M9="M9 · BRCA", M18="M18 · PRAD", M28="M28 · TGCT")

top6 <- do.call(rbind, lapply(sel, function(m){ d <- enr[enr$module==m,]; head(d[order(d$p),], 6) }))
pretty <- function(x){ x <- gsub("_"," ",x); paste0(toupper(substr(x,1,1)), tolower(substr(x,2,nchar(x)))) }
top6$pw <- pretty(top6$pathway)
top6$facet <- factor(sel_lab[top6$module], levels = sel_lab[sel])
top6$sig <- ifelse(top6$fdr < 0.05, "FDR < 0.05", "n.s.")

p <- ggplot(top6, aes(x = neg_log10_p, y = reorder(pw, neg_log10_p), fill = sig)) +
  geom_col(width = 0.72, color = "black", linewidth = 0.2) +
  geom_text(aes(label = sprintf("%d", k)), hjust = -0.25, size = 2, family = "Arial", color = "#444444") +
  facet_wrap(~ facet, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = c("FDR < 0.05" = "#ED8D5A", "n.s." = "#C9D6E5"), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.20))) +
  labs(x = expression(-log[10]~italic(p)), y = NULL,
       title = "Hallmark enrichment of representative cancer-specific phospho-modules") +
  theme_classic(base_size = 8, base_family = "Arial") +
  theme(plot.title = element_text(size = 9, face = "bold", color = "#222222"),
        strip.background = element_rect(fill = "#F2F2F2", color = NA),
        strip.text = element_text(size = 7.5, face = "bold", color = "#222222"),
        axis.text.y = element_text(size = 6.5, color = "#333333"),
        axis.text.x = element_text(size = 6.5, color = "#333333"),
        axis.title.x = element_text(size = 7),
        panel.spacing.x = unit(3, "mm"), panel.spacing.y = unit(2, "mm"),
        legend.position = "top", legend.text = element_text(size = 7),
        legend.key.size = unit(3.5, "mm"),
        axis.line = element_line(linewidth = 0.25), axis.ticks = element_line(linewidth = 0.25))
cairo_pdf(sprintf("%s/figures/figC_pathway_bars_k%d.pdf",ROOT,K), width = 170/25.4, height = 120/25.4, family = "Arial")
print(p); invisible(dev.off()); cat(sprintf("wrote figC_pathway_bars_k%d.pdf (9 modules)\n", K))
