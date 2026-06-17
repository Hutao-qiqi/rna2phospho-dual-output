source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_common.R")

make_panel_f <- function() {
  d <- read.delim(file.path(.FIG5_DATA, "panel_f_depmap_rcc_mtor_drug_correlations.tsv"),
                  sep = "\t", stringsAsFactors = FALSE)
  d <- d[d$feature == "predicted_rps6_s235_s236", ]
  d$drug_label_short <- sub("^REPURPOSING ", "REP ", d$drug_label)
  d$drug_label_short <- sub("^GDSC2 ", "GDSC2 ", d$drug_label_short)
  d$drug_label <- factor(d$drug_label_short, levels = rev(d$drug_label_short[order(d$spearman_rho)]))
  d$direction <- ifelse(d$spearman_rho < 0, "pS6-high more sensitive", "pS6-high less sensitive")
  d$label <- sprintf("n=%d; p=%.3f", d$n, d$spearman_p)

  ggplot(d, aes(spearman_rho, drug_label)) +
    geom_vline(xintercept = 0, color = "#888888", linetype = "dashed", linewidth = 0.35) +
    geom_col(aes(fill = direction), width = 0.62, color = NA, alpha = 0.92) +
    geom_text(aes(x = -0.035, label = label), hjust = 1,
              family = "Arial", size = 1.75, color = COL_TEXT) +
    scale_fill_manual(values = c("pS6-high more sensitive" = COL_WARM,
                                 "pS6-high less sensitive" = COL_BLUE), name = NULL) +
    scale_x_continuous(limits = c(-0.95, 0.25), breaks = c(-0.8, -0.4, 0, 0.2)) +
    labs(title = "mTORi response",
         subtitle = "AUC lower value means higher sensitivity",
         x = "Spearman rho vs drug AUC", y = NULL) +
    theme_fig5(7) +
    theme(panel.grid.major.y = element_blank(),
          axis.text.y = element_text(size = 6.1),
          legend.position = "none",
          legend.key.size = unit(3, "mm"))
}
