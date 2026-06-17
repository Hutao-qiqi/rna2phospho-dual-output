source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_common.R")

suppressMessages({
  library(survival)
})

.km_df <- function(fit) {
  s <- summary(fit)
  data.frame(
    time = s$time,
    surv = s$surv,
    strata = sub("^group=", "", s$strata)
  )
}

.risk_counts <- function(d, cuts) {
  do.call(rbind, lapply(cuts, function(t) {
    data.frame(
      time = t,
      group = c("Low predicted pS6", "High predicted pS6"),
      n_risk = c(
        sum(d$time >= t & d$group == "Low predicted pS6"),
        sum(d$time >= t & d$group == "High predicted pS6")
      )
    )
  }))
}

.load_panel_b_surv <- function() {
  d <- read.delim(file.path(.FIG5_DATA, "panel_b_tcga_kirc_rps6_survival_samples.tsv"),
                  sep = "\t", stringsAsFactors = FALSE)
  d <- d[is.finite(d$survival_time) & is.finite(d$survival_event) &
           d$survival_time > 0 & is.finite(d$predicted_rps6_s235_s236), ]
  med <- median(d$predicted_rps6_s235_s236, na.rm = TRUE)
  d$group <- ifelse(d$predicted_rps6_s235_s236 >= med, "High predicted pS6", "Low predicted pS6")
  d$group <- factor(d$group, levels = c("Low predicted pS6", "High predicted pS6"))
  d
}

# KM 曲线 + at-risk 表（独立子图）
make_panel_b_km <- function() {
  d <- .load_panel_b_surv()
  fit <- survival::survfit(survival::Surv(survival_time, survival_event) ~ group, data = d)
  km <- .km_df(fit)
  lr <- survival::survdiff(survival::Surv(survival_time, survival_event) ~ group, data = d)
  p_lr <- stats::pchisq(lr$chisq, df = 1, lower.tail = FALSE)

  km_plot <- ggplot(km, aes(time / 365.25, surv, color = strata)) +
    geom_step(linewidth = 0.55) +
    scale_color_manual(values = c("Low predicted pS6" = COL_BLUE, "High predicted pS6" = COL_WARM)) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
    coord_cartesian(xlim = c(0, 12)) +
    labs(title = "TCGA-KIRC overall survival", subtitle = sprintf("Median split; log-rank %s", p_text(p_lr)),
         x = "Years", y = "Overall survival") +
    theme_fig5(7) +
    theme(legend.position = c(0.70, 0.88), legend.title = element_blank(),
          legend.background = element_blank(), legend.key.size = unit(3, "mm"))

  cuts <- c(0, 3, 6, 9, 12) * 365.25
  risk <- .risk_counts(data.frame(time = d$survival_time, group = d$group), cuts)
  risk$years <- risk$time / 365.25
  risk_plot <- ggplot(risk, aes(years, group, label = n_risk, color = group)) +
    geom_text(size = 2.0, family = "Arial", show.legend = FALSE) +
    scale_color_manual(values = c("Low predicted pS6" = COL_BLUE, "High predicted pS6" = COL_WARM)) +
    scale_x_continuous(breaks = c(0, 3, 6, 9, 12), limits = c(0, 12)) +
    labs(x = NULL, y = "At risk") +
    theme_void(base_family = "Arial", base_size = 6) +
    theme(axis.text.x = element_text(color = COL_TEXT), axis.title.y = element_text(angle = 0, vjust = 0.5),
          plot.margin = margin(0, 4, 0, 28, "pt"))

  cowplot::plot_grid(km_plot, risk_plot, ncol = 1, rel_heights = c(1, 0.18))
}

# mRNA 校正 Cox forest（独立子图）
make_panel_b_forest <- function() {
  cox <- read.delim(file.path(.FIG5_DATA, "panel_b_tcga_kirc_rps6_cox_forest.tsv"),
                    sep = "\t", stringsAsFactors = FALSE)
  forest <- cox[cox$model %in% c("parent_mrna_only", "site_only", "site_plus_parent_mrna") &
                  (cox$model != "site_plus_parent_mrna" | cox$variable == "predicted_rps6_s235_s236"), ]
  forest$label <- c("RPS6 mRNA", "Predicted pS6", "Predicted pS6 | mRNA")[match(
    forest$model, c("parent_mrna_only", "site_only", "site_plus_parent_mrna"))]
  forest$label <- factor(forest$label, levels = rev(c("RPS6 mRNA", "Predicted pS6", "Predicted pS6 | mRNA")))
  forest$lo <- exp(forest$beta_per_sd - 1.96 * forest$se)
  forest$hi <- exp(forest$beta_per_sd + 1.96 * forest$se)
  forest$col <- ifelse(forest$hr_per_sd < 1, COL_BLUE, COL_WARM)

  ggplot(forest, aes(hr_per_sd, label)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "#888888", linewidth = 0.35) +
    geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.18, linewidth = 0.35, color = COL_TEXT) +
    geom_point(aes(fill = I(col)), shape = 21, size = 2.4, color = "white", stroke = 0.3) +
    scale_x_log10(limits = c(0.65, 1.9), breaks = c(0.7, 1, 1.5), labels = c("0.7", "1.0", "1.5")) +
    labs(title = "mRNA-adjusted Cox", subtitle = "LRT p=8.26e-7",
         x = "Hazard ratio per SD", y = NULL) +
    theme_fig5(7) +
    theme(panel.grid.major.y = element_blank())
}

# 合并版（向后兼容 render_fig5_panels.R）
make_panel_b <- function() {
  cowplot::plot_grid(
    make_panel_b_km(),
    make_panel_b_forest(),
    ncol = 2, rel_widths = c(1.25, 0.95), align = "h"
  )
}
