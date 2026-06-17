# fig07.R — 扩展 phosphosite 图邻域子图（56 监督节点 + 候选一跳辅助节点）
#   节点：监督节点按内部 CV ρ 着色（shape21 fill）；辅助节点小灰点。
#   边：按来源上色（target_to_bulk_candidate / CoPheeMap / CoPheeKSA / KSTAR）。
#   布局：igraph Fruchterman–Reingold（seed 682）。
# 数据：fig3/fig7_gnn_network_nodes.tsv + fig7_gnn_network_edges.tsv

make_fig07 <- function() {
  nodes <- utils::read.delim(file.path(FIG3_DIR, "fig7_gnn_network_nodes.tsv"),
                             sep = "\t", stringsAsFactors = FALSE)
  edges <- utils::read.delim(file.path(FIG3_DIR, "fig7_gnn_network_edges.tsv"),
                             sep = "\t", stringsAsFactors = FALSE)

  # 内部 CV ρ（per target_id 中位）
  icv <- utils::read.delim(
    file.path(.SC11, "02_data_tables", "scp682_sc11_formal_internal_5fold_per_target.tsv"),
    sep = "\t", stringsAsFactors = FALSE)
  icv$spearman <- suppressWarnings(as.numeric(icv$spearman))
  icv <- icv[icv$evaluation == "internal_cv_reconstruction" & icv$test_dataset == "all", ]
  icv_map <- tapply(icv$spearman, icv$target_id, function(x) stats::median(x, na.rm = TRUE))

  g <- igraph::graph_from_data_frame(
    edges[, c("node_1", "node_2")], directed = FALSE,
    vertices = data.frame(name = nodes$node_index, stringsAsFactors = FALSE))
  set.seed(682)
  lay <- igraph::layout_with_fr(g, niter = 600)
  pos <- data.frame(node_index = as.integer(igraph::V(g)$name),
                    x = lay[, 1], y = lay[, 2], stringsAsFactors = FALSE)

  nd <- merge(nodes, pos, by = "node_index")
  nd$is_sc <- nd$node_type == "sc_target"
  nd$icv   <- ifelse(nd$is_sc, icv_map[nd$label], NA)

  ec <- edges
  ec$x    <- pos$x[match(ec$node_1, pos$node_index)]
  ec$y    <- pos$y[match(ec$node_1, pos$node_index)]
  ec$xend <- pos$x[match(ec$node_2, pos$node_index)]
  ec$yend <- pos$y[match(ec$node_2, pos$node_index)]
  ec <- ec[is.finite(ec$x) & is.finite(ec$xend), ]
  esrc <- c("target_to_bulk_candidate", "CoPheeMap_onehop",
            "CoPheeKSA_onehop", "KSTAR_onehop")
  ec$source <- factor(ec$source, levels = esrc)
  ec <- ec[!is.na(ec$source), ]
  # 候选边画在最上层
  ec <- ec[order(ec$source != "target_to_bulk_candidate"), ]

  edge_cols <- c("target_to_bulk_candidate" = "#1F3A5F", "CoPheeMap_onehop" = "#6CBFB5",
                 "CoPheeKSA_onehop" = "#D4A56B", "KSTAR_onehop" = "#9C8FC4")
  edge_alpha <- c("target_to_bulk_candidate" = 0.5, "CoPheeMap_onehop" = 0.16,
                  "CoPheeKSA_onehop" = 0.16, "KSTAR_onehop" = 0.16)
  teal <- c("#F5F7FA", "#BFD8D2", "#6CBFB5", "#1F3A5F")

  sc  <- nd[nd$is_sc, ]
  aux <- nd[!nd$is_sc, ]
  lab <- sc[order(-sc$icv), ][seq_len(min(10, nrow(sc))), ]
  lab$tl <- fig3_short(lab$label, 12)

  ggplot2::ggplot() +
    ggplot2::geom_segment(data = ec,
      ggplot2::aes(x = x, y = y, xend = xend, yend = yend,
                   color = source, alpha = source), linewidth = 0.3) +
    ggplot2::geom_point(data = aux, ggplot2::aes(x, y),
                        shape = 21, fill = "#D4D4D4", color = NA, size = 0.6) +
    ggplot2::geom_point(data = sc, ggplot2::aes(x, y, fill = icv),
                        shape = 21, color = "white", stroke = 0.3, size = 2.3) +
    ggrepel::geom_text_repel(data = lab, ggplot2::aes(x, y, label = tl),
                             size = 6 / PT, family = "Arial", color = "#222222",
                             min.segment.length = 0, segment.size = 0.2,
                             box.padding = 0.25, max.overlaps = Inf, seed = 682) +
    ggplot2::scale_color_manual(values = edge_cols, name = "edge source", drop = FALSE) +
    ggplot2::scale_alpha_manual(values = edge_alpha, guide = "none") +
    ggplot2::scale_fill_gradientn(colors = teal, limits = c(0, 0.65),
                                  oob = scales::squish, name = "internal CV ρ") +
    ggplot2::labs(title = "Expanded phosphosite graph neighbourhood",
                  subtitle = "56 supervised nodes (coloured by internal CV) + one-hop auxiliary nodes.") +
    ggplot2::coord_equal() +
    cowplot::theme_nothing() +
    ggplot2::theme(
      plot.title    = ggplot2::element_text(size = 7, family = "Arial", color = "#222222",
                                            hjust = 0, margin = ggplot2::margin(0, 0, 1, 0)),
      plot.subtitle = ggplot2::element_text(size = 5.8, family = "Arial", color = "#555555",
                                            face = "italic", hjust = 0,
                                            margin = ggplot2::margin(0, 0, 3, 0)),
      legend.position = "right",
      legend.title  = ggplot2::element_text(size = 6, family = "Arial"),
      legend.text   = ggplot2::element_text(size = 5.6, family = "Arial"),
      legend.key.size = ggplot2::unit(3, "mm"),
      plot.margin = ggplot2::margin(6, 6, 6, 6, "pt"))
}
