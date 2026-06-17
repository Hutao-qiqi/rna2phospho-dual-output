# Fig 4 v3 data loading
# Loads all panel source tables into a named list

load_fig4_data <- function() {
  d <- list()

  # ---- panel b: overall bars ----
  d$panel_b_bars <- read_tsv(
    file.path(TBL_DIR, "panel_b_p100_true_overall_bars_cosine_direction.tsv"),
    show_col_types = FALSE
  )

  # ---- panel c: distribution ----
  d$panel_c_cosine_long <- read_tsv(
    file.path(TBL_DIR, "panel_c_p100_true_comparison_distribution_long.tsv"),
    show_col_types = FALSE
  )
  d$panel_c_direction_long <- read_tsv(
    file.path(TBL_DIR, "panel_c_direction_accuracy_distribution_long.tsv"),
    show_col_types = FALSE
  )
  d$panel_c_direction_median <- read_tsv(
    file.path(TBL_DIR, "panel_c_direction_accuracy_median_iqr.tsv"),
    show_col_types = FALSE
  )
  d$panel_c_direction_wilcox <- read_tsv(
    file.path(TBL_DIR, "panel_c_direction_accuracy_wilcoxon.tsv"),
    show_col_types = FALSE
  )

  # ---- panel d: drug class ----
  d$panel_d_class <- read_tsv(
    file.path(TBL_DIR, "panel_d_p100_true_drug_class_cosine_direction.tsv"),
    show_col_types = FALSE
  )
  d$panel_d_class_direction_ci <- read_tsv(
    file.path(TBL_DIR, "panel_d_drug_class_direction_accuracy_mean_ci.tsv"),
    show_col_types = FALSE
  )
  d$panel_d_class_kw <- read_tsv(
    file.path(TBL_DIR, "panel_d_drug_class_direction_kruskal_wallis.tsv"),
    show_col_types = FALSE
  )

  # ---- panel e: drug heatmap ----
  d$panel_e_drug <- read_tsv(
    file.path(TBL_DIR, "panel_e_p100_true_drug_heatmap_multi_metrics.tsv"),
    show_col_types = FALSE
  )

  # ---- panel f: true vs zero ----
  d$panel_f_paired <- read_tsv(
    file.path(TBL_DIR, "panel_f_p100_true_vs_zero_paired.tsv"),
    show_col_types = FALSE
  )
  d$panel_f_summary <- read_tsv(
    file.path(TBL_DIR, "panel_f_p100_true_vs_zero_paired_summary.tsv"),
    show_col_types = FALSE
  )
  d$panel_f_wilcox <- read_tsv(
    file.path(TBL_DIR, "panel_f_true_vs_zero_wilcoxon_above_diagonal.tsv"),
    show_col_types = FALSE
  )

  # ---- panel g: TCGA ROC + boxplot ----
  d$panel_g_box <- read_tsv(
    file.path(TBL_DIR, "panel_g_tcga_v10b_patient_score_boxplot.tsv"),
    show_col_types = FALSE
  )
  d$panel_g_box_summary <- read_tsv(
    file.path(TBL_DIR, "panel_g_tcga_v10b_patient_score_boxplot_summary.tsv"),
    show_col_types = FALSE
  )
  d$panel_g_roc <- read_tsv(
    file.path(TBL_DIR, "panel_g_tcga_v10b_roc_curve.tsv"),
    show_col_types = FALSE
  )
  d$panel_g_auc <- read_tsv(
    file.path(TBL_DIR, "panel_g_tcga_v10b_auc_ci_permutation.tsv"),
    show_col_types = FALSE
  )

  # ---- panel h: model comparison ----
  d$panel_h_models <- read_tsv(
    file.path(TBL_DIR, "panel_h_tcga_model_auc_comparison.tsv"),
    show_col_types = FALSE
  )
  d$panel_h_primary <- read_tsv(
    file.path(TBL_DIR, "panel_h_tcga_primary_score_v10_vs_v10b.tsv"),
    show_col_types = FALSE
  )

  # ---- panel i: controls ----
  d$panel_i_general <- read_tsv(
    file.path(TBL_DIR, "panel_i_tcga_general_score_control_auc.tsv"),
    show_col_types = FALSE
  )
  d$panel_i_random  <- read_tsv(
    file.path(TBL_DIR, "panel_i_tcga_random_marker_control_auc.tsv"),
    show_col_types = FALSE
  )

  d
}

cat("data loader defined\n")
