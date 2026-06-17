# fig21.R — 为什么模型在 PDO-CAF 失败？三联诊断
#   a: 各队列逐 readout 预测 SD（输出塌缩检查）。
#   b: 各队列逐 readout 平均 |pred−obs|。
#   c: 各队列 RNA-NMF 程序主导度分布（细胞类型 OOD 检查）。
# 数据：DT_EXT 各队列 predicted_observed + RNA_SD 各队列 program_scores.tsv.gz

make_fig21 <- function() {
  cohorts <- list(
    c("HeLa", "scp682_sc11_predicted_observed_signal_seq_gse256403_hela_2024.tsv", "signal_seq_gse256403_hela_2024"),
    c("GSE300551", "scp682_sc11_predicted_observed_gse300551_iccite_plex_kinase_2025.tsv", "gse300551_iccite_plex_kinase_2025"),
    c("Vivo-seq Th17", "scp682_sc11_predicted_observed_vivo_seq_th17_2025.tsv", "vivo_seq_th17_2025"),
    c("Blair", "scp682_sc11_predicted_observed_phospho_seq_blair_2025_phospho_multi.tsv", "phospho_seq_blair_2025_phospho_multi"),
    c("PDO-CAF", "scp682_sc11_predicted_observed_signal_seq_gse256404_pdo_caf_2024.tsv", "signal_seq_gse256404_pdo_caf_2024"))
  ccols <- c("HeLa" = "#6CBFB5", "GSE300551" = "#1F3A5F", "Vivo-seq Th17" = "#9C8FC4",
             "Blair" = "#D4A56B", "PDO-CAF" = "#C0392B")
  ord <- c("HeLa", "GSE300551", "Vivo-seq Th17", "Blair", "PDO-CAF")

  per <- list(); dom <- list()
  for (cc in cohorts) {
    disp <- cc[1]; f <- file.path(DT_EXT, cc[2])
    if (file.exists(f)) {
      d <- utils::read.delim(f, sep = "\t", stringsAsFactors = FALSE)
      d$predicted <- suppressWarnings(as.numeric(d$predicted))
      d$observed  <- suppressWarnings(as.numeric(d$observed))
      d$ae <- abs(d$predicted - d$observed)
      g <- split(d, d$target_id)
      per[[disp]] <- do.call(rbind, lapply(g, function(s) data.frame(
        cohort = disp, pred_std = stats::sd(s$predicted, na.rm = TRUE),
        mae = mean(s$ae, na.rm = TRUE))))
    }
    rf <- file.path(RNA_SD, paste0(cc[3], "_rna_nmf_program_scores.tsv.gz"))
    if (file.exists(rf)) {
      r <- utils::read.delim(gzfile(rf), sep = "\t", stringsAsFactors = FALSE)
      rc <- grep("^rna_nmf", names(r), value = TRUE)
      if (length(rc)) {
        X <- as.matrix(r[, rc]); s <- pmax(rowSums(X), 1e-9)
        dom[[disp]] <- data.frame(cohort = disp, dom = apply(X, 1, max) / s)
      }
    }
  }
  pv <- do.call(rbind, per); pv$cohort <- factor(pv$cohort, levels = ord)
  dv <- do.call(rbind, dom); dv$cohort <- factor(dv$cohort, levels = ord)

  vio <- function(yv, ylab, ttl) {
    ggplot2::ggplot(pv, ggplot2::aes(cohort, .data[[yv]], fill = cohort)) +
      ggplot2::geom_violin(scale = "width", color = "white", linewidth = 0.3, alpha = 0.85) +
      ggplot2::geom_jitter(width = 0.12, height = 0, size = 0.5, color = "#333333", alpha = 0.5) +
      ggplot2::stat_summary(fun = stats::median, geom = "crossbar", width = 0.5,
                            linewidth = 0.25, color = "#111111") +
      ggplot2::scale_fill_manual(values = ccols, guide = "none") +
      ggplot2::labs(x = NULL, y = ylab, title = ttl) +
      theme_fig3() +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 20, hjust = 1, size = 6))
  }
  p_a <- vio("pred_std", "Per-readout predicted SD", "Predicted SD (collapse check)")
  p_b <- vio("mae", "Mean |pred − obs|", "Prediction error per readout")

  p_c <- ggplot2::ggplot(dv, ggplot2::aes(dom, color = cohort)) +
    ggplot2::geom_density(linewidth = 0.6) +
    ggplot2::scale_color_manual(values = ccols, name = NULL) +
    ggplot2::labs(x = "Max-program dominance (top / total)", y = "Density",
                  title = "RNA-NMF program dominance") +
    theme_fig3() +
    ggplot2::theme(legend.position = "right", legend.key.size = ggplot2::unit(3, "mm"))

  body <- cowplot::plot_grid(p_a, p_b, p_c, nrow = 1, rel_widths = c(1, 1, 1.3))
  cowplot::ggdraw() +
    cowplot::draw_label("Diagnostic: why does the model fail on PDO-CAF? (red)",
                        x = 0.012, y = 0.99, hjust = 0, vjust = 1,
                        fontfamily = "Arial", size = 7, color = "#222222") +
    cowplot::draw_plot(body, x = 0, y = 0, width = 1, height = 0.96)
}
