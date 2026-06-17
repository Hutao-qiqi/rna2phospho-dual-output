# Extended Data figure (M2): PPKO vs published-method baselines on P100.
# 2x2 faceted bar: responsive-top20 cosine, responsive-top20 direction, AUROC, signed fraction.
script_dir <- "E:/data/gongke/TCGA-TCPA/02_results/figure_outputs/fig4_v3/scripts"
source(file.path(script_dir, "01_config.R"))

f <- "E:/data/gongke/TCGA-TCPA/02_results/single_cell/20260602_scp682_ppko_v10b_p100_published_baselines/tables/p100_v10b_published_baseline_comparison_summary.tsv"
s <- readr::read_tsv(f, show_col_types = FALSE)

lab <- c(ppko_v10b="PPKO", direct_kinase_substrate="KSEA-fwd",
         graph_heat_diffusion="Heat", rwr_signed_graph="RWR",
         ridge_signed_target="Ridge", retrieval_target="Retr-T",
         retrieval_target_baseline="Retr-TB")
ord <- c("PPKO","KSEA-fwd","Heat","RWR","Ridge","Retr-T","Retr-TB")

s <- as.data.frame(s[s$model %in% names(lab), ])   # drop zero_vector etc.

metrics <- list(
  c("responsive20_cosine",     "Responsive-top20 cosine"),
  c("responsive20_direction",  "Responsive-top20 direction"),
  c("response_auroc_abs_pred", "Within-comparison AUROC"),
  c("signed_fraction_all",     "Signed fraction (coverage)")
)
facet_levels <- sapply(metrics, `[`, 2)
facet_ref    <- c(0, 0.5, 0.5, NA)

build_one <- function(k, facet) data.frame(
  method = unname(lab[s$model]), facet = facet,
  mean = as.numeric(s[[paste0(k, "_mean")]]),
  lo   = as.numeric(s[[paste0(k, "_ci95_low")]]),
  hi   = as.numeric(s[[paste0(k, "_ci95_high")]]),
  stringsAsFactors = FALSE)
df <- do.call(rbind, lapply(metrics, function(m) build_one(m[1], m[2])))
df$method <- factor(df$method, levels = ord)
df$facet  <- factor(df$facet, levels = facet_levels)
df$is_ppko <- df$method == "PPKO"
refs <- data.frame(facet = factor(facet_levels, levels = facet_levels), ref = facet_ref)

p <- ggplot(df, aes(method, mean, fill = is_ppko)) +
  geom_hline(data = refs[!is.na(refs$ref),], aes(yintercept = ref),
             linetype = "dashed", color = "grey55", linewidth = 0.35) +
  geom_col(width = 0.72, color = "white", linewidth = 0.2) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.25, linewidth = 0.35,
                color = "grey30") +
  geom_text(aes(label = sprintf("%.2f", mean),
                y = pmax(hi, 0) + 0.04),
            size = 2.0, color = "grey20", family = BASE_FONT) +
  facet_wrap(~ facet, ncol = 2, scales = "free_y") +
  scale_fill_manual(values = c(`TRUE` = "#2166AC", `FALSE` = "#BDBDBD"),
                    guide = "none") +
  labs(x = NULL, y = "Per-comparison mean (± 95% CI)",
       title = "PPKO vs published-method baselines (LINCS P100, n = 125)",
       subtitle = "PPKO (blue) vs KSEA-forward, heat diffusion, RWR, ridge, retrieval (grey); dashed = chance") +
  theme_fig4() +
  theme(axis.text.x = element_text(angle = 35, hjust = 1, size = BASE_SIZE - 1),
        panel.grid.major.x = element_blank(),
        plot.subtitle = element_text(size = BASE_SIZE - 1, color = "grey40"),
        strip.text = element_text(size = BASE_SIZE, face = "bold"))

ED <- "E:/data/gongke/TCGA-TCPA/paper_final/fig4/extended_data"
dir.create(ED, showWarnings = FALSE, recursive = TRUE)
ggsave(file.path(ED, "ED_Fig_M2_published_baselines.pdf"), p, width = 6.6, height = 5.2,
       units = "in", device = cairo_pdf)
ggsave(file.path(ED, "ED_Fig_M2_published_baselines.png"), p, width = 6.6, height = 5.2,
       units = "in", dpi = 300, bg = "white")
cat("ED baseline figure written to", ED, "\n")
