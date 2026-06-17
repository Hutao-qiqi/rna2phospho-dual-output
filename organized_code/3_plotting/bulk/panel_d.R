#!/usr/bin/env Rscript
# Attention enrichment bar in panel_b visual style (ours-orange + clean theme)
# size locked to 81.529 mm (W) x 63.743 mm (H)
suppressPackageStartupMessages({library(ggplot2); library(scales)})
ROOT <- "E:/data/gongke/TCGA-TCPA/SCP682_MAIN/attention_export"
d <- read.delim(file.path(ROOT, "attn_bar_data.tsv"), stringsAsFactors = FALSE)
d$grp <- ifelse(grepl("High", d$group), "High attention (top 5%)", "Other edges")
d$category <- factor(d$category, levels = c("Same protein", "Same pathway"))
d$grp <- factor(d$grp, levels = c("High attention (top 5%)", "Other edges"))

# enrichment-fold + significance annotation, placed above the High bar
ann <- data.frame(
  category = factor(c("Same protein", "Same pathway"), levels = c("Same protein", "Same pathway")),
  frac = c(0.259066, 0.133343), enrich = c(4.73, 1.79))
ann$txt <- sprintf("%.1f×\n****", ann$enrich)

# panel_b palette: ours = warm orange #ED8D5A, baseline = cool grey-blue
COL <- c("High attention (top 5%)" = "#ED8D5A", "Other edges" = "#9FB6C9")

p <- ggplot(d, aes(category, frac, fill = grp)) +
  geom_col(position = position_dodge(0.7), width = 0.62, color = "black", linewidth = 0.25) +
  geom_text(aes(label = sprintf("%.1f%%", 100 * frac)), position = position_dodge(0.7),
            vjust = -0.4, size = 1.9, family = "Arial", color = "#222222") +
  geom_text(data = ann, aes(x = category, y = frac + 0.035, label = txt), inherit.aes = FALSE,
            size = 2.0, family = "Arial", fontface = "bold", color = "#C0392B",
            lineheight = 0.85, vjust = 0) +
  scale_fill_manual(values = COL, name = NULL) +
  scale_y_continuous(limits = c(0, 0.32), breaks = c(0, 0.1, 0.2, 0.3),
                     labels = percent_format(accuracy = 1), expand = c(0, 0)) +
  labs(x = NULL, y = "Fraction of edges with relationship",
       title = "Attention concentrates on functionally coherent site pairs") +
  theme_classic(base_size = 7, base_family = "Arial") +
  theme(
    plot.title       = element_text(size = 6.5, color = "#222222", margin = margin(0, 0, 3, 0)),
    plot.title.position = "plot",
    axis.title.y     = element_text(size = 6.5, color = "#222222"),
    axis.text        = element_text(size = 6.2, color = "#222222"),
    axis.line        = element_line(linewidth = 0.3, color = "black"),
    axis.ticks       = element_line(linewidth = 0.3, color = "black"),
    axis.ticks.length = unit(0.7, "mm"),
    legend.text      = element_text(size = 6, color = "#222222"),
    legend.title     = element_blank(),
    legend.key.size  = unit(3, "mm"),
    legend.position  = "bottom",
    legend.margin    = margin(2, 0, 0, 0),
    plot.margin      = margin(6, 8, 2, 8, "pt"))
OUT <- "E:/data/gongke/TCGA-TCPA/paper_final/fig2/main_figure/panel_d_attention_enrichment.pdf"
cairo_pdf(OUT, width = 81.529 / 25.4, height = 63.743 / 25.4, family = "Arial")
print(p); invisible(dev.off()); cat("wrote panel_d_attention_enrichment.pdf (81.529 x 63.743 mm)\n")
