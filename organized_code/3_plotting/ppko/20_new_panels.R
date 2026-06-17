# New honest panels grounded in real P100 data:
#  (1) drug x cell-line cosine accuracy heatmap  (Squidiff Fig 3g analog)
#  (2) representative site-level predicted-vs-observed scatter + global Pearson dist
# Reuses sci-plot config (Arial, low-sat palette, theme_fig4).

script_dir <- "E:/data/gongke/TCGA-TCPA/02_results/figure_outputs/fig4_v3/scripts"
source(file.path(script_dir, "01_config.R"))
suppressPackageStartupMessages({ library(pheatmap); library(grid); library(ggplot2) })

MECH_SRC <- "E:/data/gongke/TCGA-TCPA/04_figure_source_data/fig4_ppko_v10b_mechanism"
cand <- readr::read_tsv(file.path(MECH_SRC, "representative_comparison_candidates.tsv"),
                        show_col_types = FALSE)
site <- readr::read_tsv(file.path(MECH_SRC, "p100_sitelevel_delta_long.tsv"),
                        show_col_types = FALSE)

# normalise drug class underscore -> slash; title-case drug names (merge vorinostat batches)
class_map <- c("ABL_SRC" = "ABL/SRC", "EGFR_HER2" = "EGFR/HER2", "HDAC" = "HDAC",
               "MEK" = "MEK", "mTOR" = "mTOR", "proteasome" = "Proteasome",
               "other" = "Other")
cand <- cand %>%
  mutate(drug = stringr::str_to_title(perturbation),
         drug_class_en = dplyr::recode(drug_class, !!!class_map))

# ============================================================
# Panel 1: drug x cell-line cosine accuracy heatmap
# ============================================================
mat_df <- cand %>%
  group_by(drug, cell_line) %>%
  summarise(cosine = mean(cosine, na.rm = TRUE), .groups = "drop")

# order drugs by drug_class then mean cosine; cell lines by mean cosine
drug_meta <- cand %>% group_by(drug) %>%
  summarise(drug_class_en = dplyr::first(drug_class_en),
            mean_cos = mean(cosine), .groups = "drop") %>%
  arrange(factor(drug_class_en, levels = DRUG_CLASS_ORDER_EN), desc(mean_cos))
cl_order <- mat_df %>% group_by(cell_line) %>%
  summarise(m = mean(cosine)) %>% arrange(desc(m)) %>% pull(cell_line)

M <- mat_df %>%
  tidyr::pivot_wider(names_from = cell_line, values_from = cosine) %>%
  as.data.frame()
rownames(M) <- M$drug; M$drug <- NULL
M <- as.matrix(M)[drug_meta$drug, cl_order]

num <- matrix(ifelse(is.na(M), "", sprintf("%.2f", M)),
              nrow = nrow(M), dimnames = dimnames(M))

ann_row <- data.frame(Class = drug_meta$drug_class_en); rownames(ann_row) <- drug_meta$drug
present <- intersect(DRUG_CLASS_ORDER_EN, unique(drug_meta$drug_class_en))
ann_row$Class <- factor(ann_row$Class, levels = present)
ann_colors <- list(Class = DRUG_CLASS_COLORS[present])

ph <- pheatmap(
  M, cluster_rows = FALSE, cluster_cols = FALSE,
  display_numbers = num, number_color = "grey15", fontsize_number = 6,
  annotation_row = ann_row, annotation_colors = ann_colors,
  color = RDBU_RAMP,
  breaks = seq(0.3, 1.0, length.out = 101),
  na_col = "grey90",
  fontsize = 7, fontsize_row = 7, fontsize_col = 7,
  border_color = "white", angle_col = 45,
  main = "PPKO accuracy across drug x cell line (cosine)",
  legend = TRUE, silent = TRUE, fontfamily = BASE_FONT
)
cairo_pdf(file.path(PANEL_DIR, "panel_drug_cellline_cosine_heatmap.pdf"),
          width = 4.6, height = 4.4, family = BASE_FONT)
grid::grid.newpage(); grid::grid.draw(ph$gtable); dev.off()
png(file.path(PANEL_DIR, "panel_drug_cellline_cosine_heatmap.png"),
    width = 4.6*300, height = 4.4*300, res = 300)
grid::grid.newpage(); grid::grid.draw(ph$gtable); dev.off()
png(file.path(PANEL_DIR, "thumbs/panel_drug_cellline_cosine_heatmap_thumb.png"),
    width = 4.6*130, height = 4.4*130, res = 130)
grid::grid.newpage(); grid::grid.draw(ph$gtable); dev.off()
cat("panel 1 (drug x cell-line heatmap) done; matrix", nrow(M), "x", ncol(M), "\n")

# ============================================================
# Panel 2: representative site-level scatter + global Pearson distribution
# ============================================================
rep_id <- "p100_delta_00073"   # Bosutinib / YAPC, r=0.88
rep_row <- cand %>% filter(comparison_id == rep_id) %>% slice(1)
rep <- site %>% filter(comparison_id == rep_id) %>%
  transmute(observed_delta, predicted_delta,
            responsive = ifelse(is_responsive20 %in% c(TRUE, "True", "TRUE"),
                                "Responsive top 20%", "Other site"))

ann <- sprintf("Pearson r = %.2f\nSpearman = %.2f\ncosine = %.2f\nn = %d sites",
               rep_row$pearson, rep_row$spearman, rep_row$cosine, nrow(rep))

p_sc <- ggplot(rep, aes(observed_delta, predicted_delta)) +
  geom_hline(yintercept = 0, color = "grey85", linewidth = 0.3) +
  geom_vline(xintercept = 0, color = "grey85", linewidth = 0.3) +
  geom_smooth(method = "lm", se = FALSE, color = "grey55",
              linewidth = 0.5, linetype = "dashed") +
  geom_point(aes(color = responsive), size = 1.6, alpha = 0.85, stroke = 0) +
  scale_color_manual(values = c("Responsive top 20%" = "#5A89B3",
                                "Other site" = "#C9C9C9"), name = NULL) +
  annotate("text", x = -Inf, y = Inf, hjust = -0.08, vjust = 1.15,
           label = ann, size = 2.4, family = BASE_FONT, color = "grey15",
           lineheight = 1.15) +
  labs(title = "Site-level vector consistency",
       subtitle = sprintf("Representative: %s, %s",
                          rep_row$perturbation, rep_row$cell_line),
       x = "Observed Δp", y = "Predicted Δp") +
  theme_fig4() +
  theme(legend.position = c(0.99, 0.02), legend.justification = c(1, 0),
        legend.background = element_rect(fill = "white", color = NA),
        legend.key.size = unit(0.28, "cm"))

# inset: global per-comparison Pearson distribution
p_inset <- ggplot(cand, aes(pearson)) +
  geom_histogram(bins = 22, fill = "#C1D8E9", color = "white", linewidth = 0.2) +
  geom_vline(xintercept = rep_row$pearson, color = "#5A89B3",
             linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = median(cand$pearson), color = "grey40",
             linetype = "dotted", linewidth = 0.4) +
  annotate("text", x = median(cand$pearson), y = Inf, vjust = 1.4, hjust = 1.1,
           label = sprintf("median %.2f", median(cand$pearson)),
           size = 1.9, family = BASE_FONT, color = "grey35") +
  scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
  labs(title = "Per-comparison Pearson (n = 125)", x = NULL, y = NULL) +
  theme_fig4(base_size = 6) +
  theme(plot.title = element_text(size = 6),
        panel.grid.major = element_blank(),
        axis.text.y = element_blank(), axis.ticks.y = element_blank())

p2 <- p_sc + patchwork::inset_element(p_inset, left = 0.52, bottom = 0.55,
                                      right = 1.0, top = 1.0)
ggsave(file.path(PANEL_DIR, "panel_representative_sitelevel_scatter.pdf"), p2,
       width = 3.6, height = 3.2, units = "in", device = cairo_pdf)
ggsave(file.path(PANEL_DIR, "panel_representative_sitelevel_scatter.png"), p2,
       width = 3.6, height = 3.2, units = "in", dpi = 300, bg = "white")
ggsave(file.path(PANEL_DIR, "thumbs/panel_representative_sitelevel_scatter_thumb.png"), p2,
       width = 3.6, height = 3.2, units = "in", dpi = 130, bg = "white")
cat("panel 2 (representative scatter + Pearson inset) done\n")
