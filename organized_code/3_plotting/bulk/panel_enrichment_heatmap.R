# panel_enrichment_heatmap.R — Top 100 best-predicted vs Bottom 100 worst-predicted
# 位点的 Hallmark 富集热图（hypergeometric + BH-FDR）。
#
# 输入：_scripts/enrichment_top_bottom_hallmark.tsv（50 行）
# 阈值：min(top_q, bot_q) < 0.25 入选；按 (top_neg_log10_q - bot_neg_log10_q) 排序
# 输出：两列热图（Top 100 / Bottom 100），cell color = -log10 q，cell text = k 重叠

.ENR_ROOT <- "E:/data/gongke/TCGA-TCPA/paper_materials_SCP682"
.ENR_TSV  <- file.path(.ENR_ROOT, "04_figure_source_data", "fig2",
                       "_scripts", "enrichment_top_bottom_hallmark.tsv")

# 配色：sequential 白 → 深橙红（low q 高对比）
.ENR_FILL <- c("#FFFFFF", "#FFE5D2", "#F6C8B6", "#ED8D5A", "#C0392B")

# ---------------------------------------------------------------------------
.enr_load <- function() {
  d <- utils::read.delim(.ENR_TSV, sep = "\t", stringsAsFactors = FALSE)
  d$min_q <- pmin(d$top_q, d$bot_q, na.rm = TRUE)
  d
}

.enr_select <- function(d, q_thr = 0.25) {
  sel <- d[d$min_q < q_thr & !is.na(d$min_q), , drop = FALSE]
  sel$direction <- ifelse(sel$top_neg_log10_q >= sel$bot_neg_log10_q,
                          "Top", "Bottom")
  # 目标布局（自底向上）：
  #   bottom-dir 最强 → … → bottom-dir 最弱 → top-dir 最弱 → … → top-dir 最强
  # 等价于 factor levels 顺序：HEME, COAG, COMP, IL2, EST_EARLY, EST_LATE, G2M, E2F
  # ggplot y 轴 level 1 在底、level n 在顶 → 最强 bottom 在底，最强 top 在顶
  sb <- sel[sel$direction == "Bottom", , drop = FALSE]
  sb <- sb[order(-sb$bot_neg_log10_q), , drop = FALSE]   # 最强在前 → 占据底部
  st <- sel[sel$direction == "Top", , drop = FALSE]
  st <- st[order(st$top_neg_log10_q), , drop = FALSE]    # 最弱在前 → 最强在末（顶部）
  rbind(sb, st)
}

.enr_long <- function(sel) {
  out <- rbind(
    data.frame(
      gene_set       = sel$gene_set,
      col            = "Top 100\nbest-predicted",
      neg_log10_q    = sel$top_neg_log10_q,
      k              = sel$top_k,
      n_query        = sel$top_n,
      fold           = sel$top_fold,
      direction      = sel$direction,
      stringsAsFactors = FALSE),
    data.frame(
      gene_set       = sel$gene_set,
      col            = "Bottom 100\nworst-predicted",
      neg_log10_q    = sel$bot_neg_log10_q,
      k              = sel$bot_k,
      n_query        = sel$bot_n,
      fold           = sel$bot_fold,
      direction      = sel$direction,
      stringsAsFactors = FALSE)
  )
  row_order <- unique(sel$gene_set)            # sel 顺序就是想要的 level 顺序
  out$gene_set <- factor(out$gene_set, levels = row_order)
  out$col      <- factor(out$col,
                         levels = c("Top 100\nbest-predicted",
                                    "Bottom 100\nworst-predicted"))
  out
}

# ---------------------------------------------------------------------------
make_panel_enrichment_heatmap <- function(q_thr = 0.25) {
  d   <- .enr_load()
  sel <- .enr_select(d, q_thr = q_thr)
  if (nrow(sel) == 0) {
    stop(sprintf("没有 Hallmark 通路在 q < %.2f 显著", q_thr))
  }
  long <- .enr_long(sel)

  # 干净的 term 显示名
  long$term_label <- gsub("^HALLMARK_", "", as.character(long$gene_set))
  long$term_label <- gsub("_", " ", long$term_label)
  long$term_label <- factor(long$term_label,
                            levels = unique(long$term_label[
                              order(as.numeric(long$gene_set))]))

  ggplot2::ggplot(long,
    ggplot2::aes(x = col, y = term_label, fill = neg_log10_q)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.6) +
    ggplot2::geom_text(
      ggplot2::aes(label = ifelse(k > 0,
                                  sprintf("%d", k),
                                  "")),
      family = "Arial", size = 2.2,
      color = ifelse(long$neg_log10_q >= 3, "white", "#222222")) +
    ggplot2::scale_fill_gradientn(
      colors  = .ENR_FILL,
      values  = scales::rescale(c(0, 0.5, 1.3, 3, 7)),
      limits  = c(0, max(7, ceiling(max(long$neg_log10_q, na.rm = TRUE)))),
      breaks  = c(0, 1.3, 3, 5, 7),
      labels  = c("0", "1.3\n(q=0.05)", "3", "5", "7"),
      name    = expression(-log[10]~italic(q))) +
    ggplot2::scale_x_discrete(position = "top", expand = c(0, 0)) +
    ggplot2::scale_y_discrete(expand = c(0, 0)) +
    ggplot2::labs(
      x        = NULL,
      y        = NULL,
      title    = "Hallmark enrichment: top vs bottom 100 SCP682-predicted phosphosites",
      subtitle = "Hypergeometric over-representation vs the 4,427-gene SCP682 evaluable universe; BH-FDR.\nCell value = k overlap (out of 82 unique top / 78 unique bottom genes).") +
    ggplot2::theme_classic(base_size = 7, base_family = "Arial") +
    ggplot2::theme(
      panel.background    = ggplot2::element_rect(fill = "white", color = NA),
      plot.background     = ggplot2::element_rect(fill = "white", color = NA),
      plot.title          = ggplot2::element_text(
        size = 7, face = "plain", hjust = 0, color = "#222222",
        margin = ggplot2::margin(0, 0, 1, 0)),
      plot.subtitle       = ggplot2::element_text(
        size = 5.8, face = "italic", hjust = 0, color = "#555555",
        margin = ggplot2::margin(0, 0, 4, 0)),
      plot.title.position = "plot",
      axis.text.x.top     = ggplot2::element_text(size = 6.2, color = "#222222",
                                                  face = "bold",
                                                  margin = ggplot2::margin(0, 0, 2, 0)),
      axis.text.y         = ggplot2::element_text(size = 6.2, color = "#222222"),
      axis.line           = ggplot2::element_blank(),
      axis.ticks          = ggplot2::element_blank(),
      legend.position     = "right",
      legend.text         = ggplot2::element_text(size = 5.8, color = "#222222"),
      legend.title        = ggplot2::element_text(size = 6.0, color = "#222222"),
      legend.key.height   = ggplot2::unit(8, "mm"),
      legend.key.width    = ggplot2::unit(2.5, "mm"),
      legend.margin       = ggplot2::margin(0, 0, 0, 2),
      plot.margin         = ggplot2::margin(6, 6, 6, 6, "pt")
    )
}
