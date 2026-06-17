#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(ggplot2)})
ROOT <- "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
K <- 30
enr  <- read.delim(sprintf("%s/results/module_pathway_enrichment_k%d.tsv",ROOT,K), stringsAsFactors=FALSE)
summ <- read.delim(sprintf("%s/results/module_summary_k%d.tsv",ROOT,K), stringsAsFactors=FALSE)

# top 5 Hallmark per module by p
top5 <- do.call(rbind, lapply(split(enr, enr$module), function(d) head(d[order(d$p),], 5)))
# pretty pathway names
pretty <- function(x){ x <- gsub("_"," ", x); paste0(toupper(substr(x,1,1)), tolower(substr(x,2,nchar(x)))) }
top5$pw <- pretty(top5$pathway)

# facet order follows summ (sorted by pref_cancer); label = "M# CANCER"
lab_levels <- paste0(summ$module, " · ", summ$pref_cancer)
top5$facet <- factor(paste0(top5$module, " · ",
                            summ$pref_cancer[match(top5$module, summ$module)]),
                     levels = lab_levels)
top5$sig <- ifelse(top5$fdr < 0.05, "FDR < 0.05", "n.s.")

p <- ggplot(top5, aes(x = neg_log10_p, y = reorder(pw, neg_log10_p), fill = sig)) +
  geom_col(width = 0.72, color = "black", linewidth = 0.18) +
  geom_text(aes(label = sprintf("%d", k)), hjust = -0.25, size = 1.5, family = "Arial", color = "#444444") +
  facet_wrap(~ facet, scales = "free_y", ncol = 6) +
  scale_fill_manual(values = c("FDR < 0.05" = "#ED8D5A", "n.s." = "#C9D6E5"), name = NULL) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.22))) +
  labs(x = expression(-log[10]~italic(p)), y = NULL,
       title = "Hallmark enrichment of each pan-cancer phospho-module (top 5; k = 30)") +
  theme_classic(base_size = 6, base_family = "Arial") +
  theme(plot.title = element_text(size = 8, face = "bold", color = "#222222"),
        strip.clip = "off",
        strip.background = element_rect(fill = "#F2F2F2", color = NA),
        strip.text = element_text(size = 5.3, face = "bold", color = "#222222",
                                  margin = margin(1.2,0,1.2,0)),
        axis.text.y = element_text(size = 4.6, color = "#333333"),
        axis.text.x = element_text(size = 4.8, color = "#333333"),
        axis.title.x = element_text(size = 6),
        panel.spacing.x = unit(2.2, "mm"), panel.spacing.y = unit(1.2, "mm"),
        legend.position = "top", legend.text = element_text(size = 6),
        legend.key.size = unit(3, "mm"),
        axis.line = element_line(linewidth = 0.22), axis.ticks = element_line(linewidth = 0.22))
cairo_pdf(sprintf("%s/figures/figC_pathway_bars_k%d.pdf",ROOT,K), width = 185/25.4, height = 250/25.4, family = "Arial")
print(p); invisible(dev.off()); cat(sprintf("wrote figC_pathway_bars_k%d.pdf\n", K))
