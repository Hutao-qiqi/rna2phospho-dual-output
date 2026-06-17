#!/usr/bin/env Rscript
# 新 panel b: 24 生物学 pathway_module × 癌种 risk-minus-protective 热图
# 行序 = panel c 顺序(risk 在顶), 列序 = panel a 癌种序; 小 n 格子淡化(alpha by n)
suppressPackageStartupMessages({ library(data.table); library(ggplot2); library(scales) })
TAB <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/source_data/tables"
FIG <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5/figures/enhancements"
dir.create(FIG, recursive=TRUE, showWarnings=FALSE)
FAM<-"Arial"; INK<-"#2B2B2B"; SUB<-"#555555"; COL_MALIG<-"#E8927A"; COL_BLUE<-"#92B1D9"; GREY<-"#BFBFBF"
PT<-function(pt)pt/2.845
save3<-function(p,stem,w,h){
  tryCatch(ggsave(file.path(FIG,paste0(stem,".pdf")),p,width=w,height=h,device=cairo_pdf),error=function(e)ggsave(file.path(FIG,paste0(stem,".pdf")),p,width=w,height=h))
  tryCatch({if(requireNamespace("ragg",quietly=TRUE))ggsave(file.path(FIG,paste0(stem,".png")),p,width=w,height=h,dpi=400,device=ragg::agg_png) else ggsave(file.path(FIG,paste0(stem,".png")),p,width=w,height=h,dpi=400)},error=function(e)message(e))
  tryCatch({if(requireNamespace("svglite",quietly=TRUE))ggsave(file.path(FIG,paste0(stem,".svg")),p,width=w,height=h,device=svglite::svglite) else ggsave(file.path(FIG,paste0(stem,".svg")),p,width=w,height=h)},error=function(e)message(e))
  pv<-floor(1900/max(w,h)); ggsave(file.path(FIG,paste0("_preview_",stem,".png")),p,width=w,height=h,dpi=pv,bg="white")
}

rmp <- fread(file.path(TAB,"fig5b_module_cancer_rmp.tsv")); setnames(rmp,1,"pathway_module")
nmat<- fread(file.path(TAB,"fig5b_module_cancer_n.tsv"));   setnames(nmat,1,"pathway_module")
meta<- fread(file.path(TAB,"fig5b_module_meta.tsv"))
canc_levels <- setdiff(colnames(rmp),"pathway_module")
rl <- melt(rmp, id.vars="pathway_module", variable.name="cancer", value.name="rmp")
nl <- melt(nmat,id.vars="pathway_module", variable.name="cancer", value.name="n")
d <- merge(rl,nl,by=c("pathway_module","cancer"))
d <- merge(d, meta[,.(pathway_module,short)], by="pathway_module")
d[, short  := factor(short, levels=meta$short)]            # meta 已按 c 的 risk_frac 升序 -> risk 在顶
d[, cancer := factor(cancer, levels=canc_levels)]
d[, alpha  := pmin(1, n/20)]
d <- d[is.finite(rmp)]

p <- ggplot(d, aes(cancer, short, fill=rmp, alpha=alpha)) +
  geom_tile(color="white", linewidth=0.3) +
  scale_fill_gradient2(low="blue", mid="white", high="red", midpoint=0,
                       limits=c(-1,1), name="risk − protective") +
  scale_alpha(range=c(0.28,1), guide="none") +
  scale_x_discrete(expand=c(0,0)) + scale_y_discrete(expand=c(0,0)) +
  labs(title="Pan-cancer direction of phospho-modules across cancers",
       subtitle="Biological modules (rows; same order & identity as panel c) × cancer. Red = risk-skewed, blue = protective-skewed; faded cells = few significant sites.",
       x=NULL, y=NULL) +
  theme_minimal(base_family=FAM, base_size=12) +
  theme(axis.text.x=element_text(angle=55,hjust=1,vjust=1,size=7.5,face="bold",color=INK),
        axis.text.y=element_text(size=7.5,color=INK),
        panel.grid=element_blank(), plot.title=element_text(face="bold",size=11.5,color=INK),
        plot.subtitle=element_text(size=7.8,color=SUB), plot.title.position="plot",
        legend.position="bottom", legend.key.height=unit(3,"mm"), legend.key.width=unit(7,"mm"),
        legend.title=element_text(size=7.5), legend.text=element_text(size=7),
        plot.margin=margin(3,5,3,5))
save3(p,"fig5b_pathway_module_cancer", w=8.4, h=6.4)
cat("PANEL B (module) DONE\n")
