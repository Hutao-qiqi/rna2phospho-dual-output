#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(survival); library(ggplot2)})
ROOT <- "E:/data/gongke/TCGA-TCPA/02_results/model_validation/20260612_tcga_pancancer_nmf_v1"
K <- 30
sma  <- read.delim(sprintf("%s/results/sample_module_activity_k%d.tsv",ROOT,K), check.names=FALSE)
meta <- read.delim(sprintf("%s/results/sample_meta.tsv",ROOT))
summ <- read.delim(sprintf("%s/results/module_summary_k%d.tsv",ROOT,K), stringsAsFactors=FALSE)

df <- merge(sma, meta[,c("sample_id","survival_time","survival_event")], by="sample_id")
df <- df[!is.na(df$survival_time) & df$survival_time>0 & !is.na(df$survival_event), ]
cat("samples with usable OS:", nrow(df), " events:", sum(df$survival_event), "\n")

mods <- paste0("M",1:K)
res <- do.call(rbind, lapply(mods, function(m){
  cx <- tryCatch(coxph(as.formula(sprintf("Surv(survival_time,survival_event)~`%s`+strata(cancer)",m)), data=df),
                 error=function(e) NULL)
  if (is.null(cx)) return(NULL)
  s <- summary(cx)
  data.frame(module=m, HR=s$coef[1,2], lo=s$conf.int[1,3], hi=s$conf.int[1,4], p=s$coef[1,5])
}))
res$fdr <- p.adjust(res$p, "BH")
res <- merge(res, summ[,c("module","pref_cancer","top_hallmark")], by="module")
res$label <- paste0(res$module, " ", res$pref_cancer)
res$sig <- res$fdr < 0.05
res <- res[order(res$HR), ]
res$label <- factor(res$label, levels=res$label)
write.table(res, sprintf("%s/results/module_survival_k%d.tsv",ROOT,K), sep="\t", row.names=FALSE, quote=FALSE)
cat("=== significant prognostic modules (FDR<0.05) ===\n")
print(res[res$sig, c("module","pref_cancer","HR","p","fdr","top_hallmark")], row.names=FALSE)

p <- ggplot(res, aes(HR, label, color=HR>1)) +
  geom_vline(xintercept=1, linetype=2, color="grey60", linewidth=0.3) +
  geom_errorbarh(aes(xmin=lo, xmax=hi), height=0.32, linewidth=0.4) +
  geom_point(aes(shape=sig), size=1.9, fill="white", stroke=0.6) +
  scale_shape_manual(values=c(`TRUE`=16, `FALSE`=21), guide="none") +
  scale_color_manual(values=c(`TRUE`="#CB181D", `FALSE`="#2171B5"), guide="none") +
  scale_x_log10(breaks=c(0.7,1,1.5,2,3)) +
  labs(x="OS hazard ratio per SD (Cox, stratified by cancer)", y=NULL,
       title="Phospho-module activity vs overall survival (n = 9,102)",
       subtitle="filled = FDR < 0.05;  red = higher risk, blue = protective") +
  theme_classic(base_size=7, base_family="Arial") +
  theme(plot.title=element_text(size=8, face="bold", color="#222222"),
        plot.subtitle=element_text(size=5.8, color="#555555"),
        axis.text.y=element_text(size=5.8, color="#333333"),
        axis.text.x=element_text(size=6, color="#333333"),
        axis.title.x=element_text(size=6.5),
        axis.line=element_line(linewidth=0.25), axis.ticks=element_line(linewidth=0.25))
cairo_pdf(sprintf("%s/figures/figD_survival_k%d.pdf",ROOT,K), width=120/25.4, height=135/25.4, family="Arial")
print(p); invisible(dev.off()); cat(sprintf("wrote figD_survival_k%d.pdf\n", K))
