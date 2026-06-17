source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_common.R")

make_panel_a <- function() {
  d <- read.delim(file.path(.FIG5_DATA, "panel_a_tcga_predicted_pS6_mRNA_controls.tsv"),
                  sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
  d$`RPS6 mRNA log2(TPM+1)` <- log2(as.numeric(d$RPS6_mrna) + 1)

  p1 <- raincloud_plot(d, "OS status", "predicted_rps6_s235_s236",
                       c("Alive/censored", "Deceased"), c(COL_BLUE, COL_WARM),
                       "OS outcome", "Predicted RPS6 pS235/S236")
  p2 <- raincloud_plot(d[!is.na(d$Grade), ], "Grade", "predicted_rps6_s235_s236",
                       c("G1/G2", "G3/G4"), c(COL_BLUE, COL_WARM),
                       "Histologic grade", NULL)
  p3 <- raincloud_plot(d[!is.na(d$`predicted mTOR phospho-state`), ],
                       "predicted mTOR phospho-state", "predicted_rps6_s235_s236",
                       c("Low", "High"), c(COL_BLUE, COL_WARM),
                       "mTOR-axis state (target excluded)", NULL)
  p4 <- raincloud_plot(d, "OS status", "RPS6 mRNA log2(TPM+1)",
                       c("Alive/censored", "Deceased"), c(COL_BLUE, COL_WARM),
                       "RPS6 mRNA control", "RPS6 mRNA log2(TPM+1)")
  p5 <- raincloud_plot(d[!is.na(d$Grade), ], "Grade", "RPS6 mRNA log2(TPM+1)",
                       c("G1/G2", "G3/G4"), c(COL_BLUE, COL_WARM),
                       "RPS6 mRNA control", NULL)
  p6 <- raincloud_plot(d[!is.na(d$`predicted mTOR phospho-state`), ],
                       "predicted mTOR phospho-state", "RPS6 mRNA log2(TPM+1)",
                       c("Low", "High"), c(COL_BLUE, COL_WARM),
                       "RPS6 mRNA control", NULL)

  cowplot::plot_grid(p1, p2, p3, p4, p5, p6, nrow = 2, align = "hv")
}
