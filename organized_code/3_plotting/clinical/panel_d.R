source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_common.R")

make_panel_d <- function() {
  d <- read.delim(file.path(.FIG5_DATA, "panel_d_cptac_ccrcc_rps6_bubble_partial_correlations.tsv"),
                  sep = "\t", stringsAsFactors = FALSE)
  keep <- c("MS p-S6 S235/S236", "MS p-S6K1 T421/S424", "MS p-4EBP1 S65",
            "MS p-4EBP1 T70", "MS p-mTOR S2481", "RPS6 mRNA adjusted out",
            "RPS6 protein adjusted out", "Proliferation score")
  d <- d[d$column_label %in% keep, ]
  pretty <- c(
    "MS p-S6 S235/S236" = "Target pS6\nmeasured",
    "MS p-S6K1 T421/S424" = "p-S6K1\nT421/S424",
    "MS p-4EBP1 S65" = "p-4EBP1\nS65",
    "MS p-4EBP1 T70" = "p-4EBP1\nT70",
    "MS p-mTOR S2481" = "p-mTOR\nS2481",
    "RPS6 mRNA adjusted out" = "RPS6 mRNA\nadjusted",
    "RPS6 protein adjusted out" = "RPS6 protein\nadjusted",
    "Proliferation score" = "Proliferation\nscore"
  )
  d$column_pretty <- pretty[d$column_label]
  d$column_pretty <- factor(d$column_pretty, levels = pretty[keep])
  d$row_label <- factor(d$row_label, levels = rev(c("Predicted pS6", "Measured pS6")))
  d$abs_r <- abs(d$r)
  d$label <- ifelse(d$mode == "adjusted_out", "0", sprintf("%.2f", d$r))

  ggplot(d, aes(column_pretty, row_label)) +
    geom_vline(xintercept = c(1.5, 5.5), color = "#D8D8D8", linewidth = 0.35) +
    geom_point(aes(size = abs_r, fill = r), shape = 21, color = "white", stroke = 0.35) +
    geom_text(aes(label = label), family = "Arial", size = 1.8, color = COL_TEXT) +
    geom_text(aes(label = paste0("n=", n)), nudge_y = -0.30, family = "Arial", size = 1.45, color = "#666666") +
    scale_fill_gradient2(low = COL_BLUE, mid = "white", high = COL_WARM, midpoint = 0,
                         limits = c(-0.75, 0.75), name = "partial r",
                         guide = guide_colorbar(barheight = unit(17, "mm"), barwidth = unit(2.6, "mm"))) +
    scale_size(range = c(1.2, 7.0), limits = c(0, 1), guide = "none") +
    labs(title = "Target-site and mTOR-axis covariance",
         subtitle = "partial r after RPS6 mRNA + total protein adjustment; TCGA mTOR-state excludes target pS6",
         x = NULL, y = NULL) +
    theme_fig5(7) +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 5.9, lineheight = 0.92),
      axis.text.y = element_text(size = 6.6),
      panel.grid.major.x = element_line(color = "#EFEFEF", linewidth = 0.25),
      panel.grid.major.y = element_line(color = "#EFEFEF", linewidth = 0.25),
      legend.position = "right",
      legend.key.size = unit(2.5, "mm"),
      legend.spacing.y = unit(0.2, "mm")
    )
}
