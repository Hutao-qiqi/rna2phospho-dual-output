# fig10.R — HeLa NMF 成分得分 vs 逐细胞预测误差（2 误差口径 × 3 成分 hexbin）
# 数据：fig3/fig10_nmf_vs_error_per_cell.tsv + fig10_nmf_vs_error_correlations.tsv

make_fig10 <- function() {
  d  <- utils::read.delim(file.path(FIG3_DIR, "fig10_nmf_vs_error_per_cell.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  cr <- utils::read.delim(file.path(FIG3_DIR, "fig10_nmf_vs_error_correlations.tsv"),
                          sep = "\t", stringsAsFactors = FALSE)
  comps <- c("nmf1", "nmf2", "nmf3")
  errs  <- c("error_high_var", "error_all")
  ylabs <- c(error_high_var = "mean |pred − obs|\n(5 high-variance)",
             error_all = "mean |pred − obs|\n(all 13 readouts)")

  mk <- function(comp, err, show_title, show_ylab) {
    rr <- cr[cr$component == toupper(comp) & cr$error_metric == err, ]
    rho <- rr$spearman_r[1]; p <- rr$spearman_p[1]
    ptxt <- if (p < 1e-10) "P < 1e-10" else sprintf("P = %.1e", p)
    ggplot2::ggplot(d, ggplot2::aes(.data[[comp]], .data[[err]])) +
      ggplot2::geom_hex(bins = 26) +
      ggplot2::scale_fill_gradientn(colors = FIG3_SEQ_TEAL, trans = "log10", name = "cells") +
      ggplot2::annotate("text", x = -Inf, y = Inf, hjust = -0.08, vjust = 1.2,
                        label = sprintf("ρ = %+.3f\n%s", rho, ptxt),
                        family = "Arial", size = 5.4 / PT, color = "#222222") +
      ggplot2::labs(x = paste0(toupper(comp), " score"),
                    y = if (show_ylab) ylabs[[err]] else NULL,
                    title = if (show_title) toupper(comp) else NULL) +
      theme_fig3() +
      ggplot2::theme(legend.position = "none",
                     plot.margin = ggplot2::margin(4, 6, 2, 4, "pt"))
  }

  panels <- list()
  for (ei in seq_along(errs)) for (ci in seq_along(comps))
    panels[[length(panels) + 1]] <- mk(comps[ci], errs[ei],
                                       show_title = (ei == 1), show_ylab = (ci == 1))
  body <- cowplot::plot_grid(plotlist = panels, ncol = 3, align = "hv")
  cowplot::ggdraw() +
    cowplot::draw_label("HeLa NMF component score vs per-cell prediction error (Spearman)",
                        x = 0.012, y = 0.985, hjust = 0, vjust = 1,
                        fontfamily = "Arial", size = 7, color = "#222222") +
    cowplot::draw_plot(body, x = 0, y = 0, width = 1, height = 0.95)
}
