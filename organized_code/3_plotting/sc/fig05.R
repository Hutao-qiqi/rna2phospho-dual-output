# fig05.R — 扩展图残差贡献分解（matched ablation，逐 readout）
#   灰底条 = 无扩展图 backbone 的 Spearman；叠加的彩色段 = 扩展 GNN 的 Δ（增=金/减=蓝）。
# 数据：fig3/fig5_gnn_residual_contribution.tsv

make_fig05 <- function() {
  d <- utils::read.delim(file.path(FIG3_DIR, "fig5_gnn_residual_contribution.tsv"),
                         sep = "\t", stringsAsFactors = FALSE)
  d$base <- suppressWarnings(as.numeric(d$spearman_matched_no_site_graph))
  d$inc  <- suppressWarnings(as.numeric(d$gnn_increment))
  d <- d[is.finite(d$base) & is.finite(d$inc), ]
  d <- d[order(d$inc), ]
  d$ord  <- seq_len(nrow(d))
  d$xend <- d$base + d$inc
  d$tlab <- fig3_short(d$target_id, 20)
  d$sign <- factor(ifelse(d$inc >= 0, "expanded-GNN gain", "expanded-GNN decrease"),
                   levels = c("expanded-GNN gain", "expanded-GNN decrease"))

  ggplot2::ggplot(d) +
    # 灰底：0 → no-graph backbone
    ggplot2::geom_rect(ggplot2::aes(xmin = 0, xmax = base,
                                    ymin = ord - 0.36, ymax = ord + 0.36),
                       fill = "#DBDBDB") +
    # 彩色段：backbone → backbone+Δ
    ggplot2::geom_rect(ggplot2::aes(xmin = base, xmax = xend,
                                    ymin = ord - 0.36, ymax = ord + 0.36, fill = sign)) +
    ggplot2::geom_vline(xintercept = 0, color = "#555555", linewidth = 0.4) +
    ggplot2::scale_fill_manual(values = c("expanded-GNN gain" = "#D4A56B",
                                          "expanded-GNN decrease" = "#92B1D9"),
                               name = NULL) +
    ggplot2::scale_y_continuous(breaks = d$ord, labels = d$tlab, expand = c(0, 0.6)) +
    ggplot2::scale_x_continuous(expand = ggplot2::expansion(mult = c(0, 0.02))) +
    ggplot2::labs(x = "Spearman ρ", y = NULL,
                  title = "Expanded graph contribution (matched ablation)",
                  subtitle = "Grey = SC backbone without expanded graph; colour = expanded-GNN Δ stacked on top.") +
    theme_fig3() +
    ggplot2::theme(axis.text.y = ggplot2::element_text(size = 5.2),
                   axis.ticks.y = ggplot2::element_blank())
}
