source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_common.R")

.residualize <- function(y, covars) {
  d <- data.frame(y = y, covars)
  d <- d[complete.cases(d), ]
  out <- rep(NA_real_, length(y))
  if (nrow(d) < ncol(covars) + 4) return(out)
  fit <- stats::lm(y ~ ., data = d)
  out[as.integer(rownames(d))] <- stats::resid(fit)
  out
}

make_panel_c <- function() {
  d <- read.delim(file.path(.FIG5_DATA, "panel_c_cptac_ccrcc_rps6_prediction_measurement.tsv"),
                  sep = "\t", stringsAsFactors = FALSE)
  d$row_id <- seq_len(nrow(d))
  dd <- d[complete.cases(d[, c("measured_rps6_s235_s236", "predicted_rps6_s235_s236",
                               "RPS6_mrna", "RPS6_total_protein")]), ]
  rho <- suppressWarnings(stats::cor(dd$predicted_rps6_s235_s236, dd$measured_rps6_s235_s236,
                                     method = "spearman"))
  p_raw <- suppressWarnings(stats::cor.test(dd$predicted_rps6_s235_s236, dd$measured_rps6_s235_s236,
                                            method = "spearman")$p.value)

  raw <- ggplot(dd, aes(predicted_rps6_s235_s236, measured_rps6_s235_s236)) +
    geom_point(size = 1.45, color = COL_WARM, alpha = 0.82) +
    geom_smooth(method = "lm", se = FALSE, color = COL_TEXT, linewidth = 0.45) +
    labs(title = "Target-site match", subtitle = sprintf("n=%d; rho=%.2f; %s", nrow(dd), rho, p_text(p_raw)),
         x = "Predicted RPS6 pS235/S236", y = "Measured RPS6 pS235/S236") +
    theme_fig5(7)

  dd$pred_resid <- stats::resid(stats::lm(predicted_rps6_s235_s236 ~ RPS6_mrna + RPS6_total_protein, data = dd))
  dd$meas_resid <- stats::resid(stats::lm(measured_rps6_s235_s236 ~ RPS6_mrna + RPS6_total_protein, data = dd))
  rho2 <- suppressWarnings(stats::cor(dd$pred_resid, dd$meas_resid, method = "spearman"))
  p2 <- suppressWarnings(stats::cor.test(dd$pred_resid, dd$meas_resid, method = "spearman")$p.value)

  res <- ggplot(dd, aes(pred_resid, meas_resid)) +
    geom_hline(yintercept = 0, color = "#EEEEEE", linewidth = 0.3) +
    geom_vline(xintercept = 0, color = "#EEEEEE", linewidth = 0.3) +
    geom_point(size = 1.45, color = COL_BLUE, alpha = 0.82) +
    geom_smooth(method = "lm", se = FALSE, color = COL_TEXT, linewidth = 0.45) +
    labs(title = "Adjusted match", subtitle = sprintf("rho=%.2f; %s", rho2, p_text(p2)),
         x = "Predicted residual", y = "Measured residual") +
    theme_fig5(7)

  cowplot::plot_grid(raw, res, nrow = 1, align = "h")
}
