# panel_a.R — Fig 3 panel a：SCP682-SC 架构示意（横长 ~180 × 50 mm）
#
# 用途：source 后调用 make_panel_a() 返回 ggplot 对象。panel letter 由合成层注入。
#
# 设计：Fig 2 panel a 的姊妹图，沿用同一视觉语法（annotate 盒子 + 箭头）。
#   坐标系 xlim=c(0,17.4) ylim=c(0,6.1)
#   顶部：Bulk SCP682 site network 小盒 → (dashed transfer) → Expanded ScNET site graph 横条
#         横条 ↓ 点线箭头进入 ScNET graph residual 盒
#   中部：scRNA → scFoundation φ → Pathway attention → ScNET graph residual → Fusion→ŷ
#   底部：训练配置 caption 条
#
# 配色（与 Fig 2 panel a 同源，低饱和）：
#   金（图 / transfer）  fill #F5E8D0 / stroke #D4A56B / text #6B4A1F
#   蓝（神经管线）        fill #C1D8E9 / stroke #5680B0 / text #1F3A5F
#   深蓝（输出 ŷ）        fill #3D5A80 / text white
#
# 数字来源（verified hyperparameters，见 paper_final manifest.json）：
#   scFoundation 3,072-d / 9 pathway tokens / 56 readouts
#   expanded graph 7,369 nodes (56 supervised + 7,313 auxiliary) · 882,959 edges
#   training 121,847 cells · AdamW lr 1.8e-4 · batch 2,048 · seed 682522 · delta_weight=0

make_panel_a <- function() {

  # ---- 配色 ----
  GF <- "#F5E8D0"; GS <- "#D4A56B"; GT <- "#6B4A1F"   # gold (graph / transfer)
  BF <- "#C1D8E9"; BS <- "#5680B0"; BT <- "#1F3A5F"   # blue (neural pipeline)
  OF <- "#3D5A80"; OS <- "#1F3A5F"                     # output dark blue
  CF <- "#F2F2F2"; CS <- "#C9C9C9"; CTx <- "#555555"   # caption strip grey

  # ---- 盒子 helper：返回 annotate 图层 list（sub=NA 则单行居中粗体）----
  box <- function(xmin, xmax, ymin, ymax, title, sub, fill, stroke, txt,
                  ts, ss = 5.8, lw = 0.4) {
    cx <- (xmin + xmax) / 2
    L <- list(ggplot2::annotate("rect", xmin = xmin, xmax = xmax,
                                ymin = ymin, ymax = ymax,
                                fill = fill, color = stroke, linewidth = lw))
    if (is.na(sub)) {
      L <- c(L, list(ggplot2::annotate(
        "text", x = cx, y = (ymin + ymax) / 2, label = title,
        family = "Arial", size = ts / PT, fontface = "bold", color = txt)))
    } else {
      L <- c(L, list(
        ggplot2::annotate("text", x = cx, y = ymax - (ymax - ymin) * 0.30,
                          label = title, family = "Arial", size = ts / PT,
                          fontface = "bold", color = txt),
        ggplot2::annotate("text", x = cx, y = ymin + (ymax - ymin) * 0.27,
                          label = sub, family = "Arial", size = ss / PT,
                          color = txt)))
    }
    L
  }
  arrow_h <- function(x, xend, col)
    ggplot2::annotate("segment", x = x, xend = xend, y = 3.00, yend = 3.00,
                      color = col, linewidth = 0.45,
                      arrow = ggplot2::arrow(length = ggplot2::unit(1.1, "mm"),
                                             type = "closed"))

  ggplot2::ggplot() +
    ggplot2::coord_cartesian(xlim = c(0, 17.4), ylim = c(0, 6.1), expand = FALSE) +

    # ---------- 顶部：bulk 网络 + 扩展图 ----------
    box(0.30, 4.00, 4.95, 5.80, "Bulk SCP682 site network",
        "18,592-site transfer prior", GF, GS, GT, 6.5, 5.6) +
    box(4.80, 16.95, 5.05, 5.70,
        "Expanded ScNET site graph  ·  7,369 nodes  ·  882,959 edges",
        NA, GF, GS, GT, 6.8) +
    # transfer dashed arrow bulk → expanded
    ggplot2::annotate("segment", x = 4.00, xend = 4.78, y = 5.37, yend = 5.37,
                      color = GS, linewidth = 0.35, linetype = "dashed",
                      arrow = ggplot2::arrow(length = ggplot2::unit(0.9, "mm"))) +
    ggplot2::annotate("text", x = 4.39, y = 6.00, label = "transfer",
                      family = "Arial", size = 5.4 / PT, fontface = "italic",
                      color = GT) +

    # ---------- 中部：神经管线 5 盒 ----------
    box(0.30, 1.95, 2.05, 3.95, "scRNA", NA, BF, BS, BT, 7) +
    box(2.65, 5.75, 2.05, 3.95, "scFoundation φ", "frozen · 3,072-d",
        BF, BS, BT, 7) +
    box(6.45, 9.55, 2.05, 3.95, "Pathway attention",
        "9 tokens · 56 site queries", BF, BS, BT, 7) +
    box(10.25, 13.00, 2.05, 3.95, "ScNET graph residual",
        "site message passing", GF, GS, GT, 7) +
    box(13.70, 16.95, 2.05, 3.95, "Fusion → ŷ phospho",
        "concat(pathway, graph) · 56 sites", OF, OS, "white", 7) +
    # 水平箭头
    arrow_h(1.95, 2.63, BS) + arrow_h(5.75, 6.43, BS) +
    arrow_h(9.55, 10.23, BS) + arrow_h(13.00, 13.68, GS) +
    # 扩展图 ↓ 点线箭头进入 graph residual 盒中点
    ggplot2::annotate("segment", x = 11.625, xend = 11.625,
                      y = 5.03, yend = 3.98, color = GS, linewidth = 0.35,
                      linetype = "dotted",
                      arrow = ggplot2::arrow(length = ggplot2::unit(0.95, "mm"))) +

    # ---------- 底部：训练配置 caption 条 ----------
    ggplot2::annotate("rect", xmin = 0.30, xmax = 16.95,
                      ymin = 0.45, ymax = 1.20,
                      fill = CF, color = CS, linewidth = 0.3) +
    ggplot2::annotate(
      "text", x = 8.625, y = 0.825,
      label = paste0("Supervised on 121,847 single cells  ·  56 readouts  ·  ",
                     "AdamW (lr 1.8×10⁻⁴, batch 2,048, seed 682522)  ·  ",
                     "no explicit drug-delta term (delta_weight = 0)"),
      family = "Arial", size = 5.6 / PT, fontface = "italic", color = CTx) +

    ggplot2::labs(
      title = "scFoundation encoder + pathway attention + ScNET graph residual") +
    ggplot2::theme_void(base_family = "Arial") +
    ggplot2::theme(
      plot.title  = ggplot2::element_text(size = 7, hjust = 0, color = "#222222",
                                          margin = ggplot2::margin(0, 0, 2, 13)),
      plot.margin = ggplot2::margin(4, 4, 2, 4, "pt"))
}
