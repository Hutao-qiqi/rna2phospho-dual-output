#!/usr/bin/env Rscript
suppressPackageStartupMessages({ library(ggplot2); library(cowplot) })
setwd("E:/data/gongke/TCGA-TCPA/SCP682_MAIN/attention_export")

bar <- read.delim("attn_bar_data.tsv", stringsAsFactors = FALSE)
vio <- read.delim("attn_violin_data.tsv", stringsAsFactors = FALSE)

COL_HIGH <- "#ED8D5A"; COL_OTHER <- "#92B1D9"; TXT <- "#222222"
bar$group <- factor(bar$group, levels = c("High attention\n(top 5%)", "Other edges"))
bar$category <- factor(bar$category, levels = c("Same protein", "Same pathway"))

# enrichment annotation per category
lab <- data.frame(
  category = factor(c("Same protein","Same pathway"), levels=levels(bar$category)),
  y = c(0.30, 0.30),
  txt = c("4.7×\n****", "1.8×\n****"))

p_bar <- ggplot(bar, aes(category, frac, fill = group)) +
  geom_col(position = position_dodge(0.72), width = 0.66,
           color = "black", linewidth = 0.25) +
  geom_text(aes(label = sprintf("%.0f%%", 100*frac)),
            position = position_dodge(0.72), vjust = -0.4,
            size = 2.0, family = "Arial", color = TXT) +
  geom_text(data = lab, aes(category, y, label = txt), inherit.aes = FALSE,
            size = 2.2, family = "Arial", fontface = "bold", color = "#C0392B",
            lineheight = 0.9) +
  scale_fill_manual(values = c("High attention\n(top 5%)" = COL_HIGH,
                               "Other edges" = COL_OTHER), name = NULL) +
  scale_y_continuous(limits = c(0, 0.34), breaks = c(0,0.1,0.2,0.3),
                     labels = scales::percent_format(accuracy = 1), expand = c(0,0)) +
  labs(x = NULL, y = "Edges with the relationship",
       title = "Learned attention is enriched for\nfunctionally coherent site pairs") +
  theme_classic(base_size = 7, base_family = "Arial") +
  theme(plot.title = element_text(size = 7.2, color = TXT, margin = margin(0,0,3,0)),
        axis.title.y = element_text(size = 6.5, color = TXT),
        axis.text = element_text(size = 6.2, color = TXT),
        axis.line = element_line(linewidth = 0.3), axis.ticks = element_line(linewidth = 0.3),
        legend.text = element_text(size = 6, color = TXT, lineheight = 0.85),
        legend.key.size = unit(3, "mm"), legend.position = "bottom",
        legend.margin = margin(0,0,0,0),
        plot.margin = margin(6,8,4,8,"pt"))

# violin: per-edge attention by relationship
vio$value <- factor(vio$value, levels = c("yes","no"),
                    labels = c("with","without"))
vio$label <- factor(vio$label, levels = c("Same protein", "Same pathway"))
p_vio <- ggplot(vio, aes(value, attention, fill = value)) +
  geom_violin(scale = "width", width = 0.8, color = "black", linewidth = 0.2, alpha = 0.85) +
  geom_boxplot(width = 0.16, outlier.shape = NA, color = "black", linewidth = 0.2, fill = "white") +
  facet_wrap(~ label, nrow = 1) +
  scale_fill_manual(values = c("with" = COL_HIGH, "without" = COL_OTHER), guide = "none") +
  scale_y_continuous(limits = c(0, 0.10), breaks = c(0,0.02,0.04,0.06,0.08,0.10),
                     oob = scales::squish, expand = c(0,0)) +
  labs(x = NULL, y = "Per-edge attention",
       title = "Attention is higher on coherent edges") +
  theme_classic(base_size = 7, base_family = "Arial") +
  theme(plot.title = element_text(size = 7.2, color = TXT, margin = margin(0,0,3,0)),
        axis.title.y = element_text(size = 6.5, color = TXT),
        axis.text = element_text(size = 6.0, color = TXT),
        axis.line = element_line(linewidth = 0.3), axis.ticks = element_line(linewidth = 0.3),
        strip.background = element_rect(fill = "#F5F5F5", color = NA),
        strip.text = element_text(size = 6.5, color = TXT),
        plot.margin = margin(6,8,4,8,"pt"))

combined <- plot_grid(p_bar, p_vio, ncol = 2, rel_widths = c(1, 1.05),
                      labels = c("",""), align = "h")
cairo_pdf("fig2d_interpretability_proto.pdf", width = 170/25.4, height = 65/25.4, family = "Arial")
print(combined); invisible(dev.off())
cat("wrote fig2d_interpretability_proto.pdf\n")
