# Panel a: architecture schematic (BioRender placeholder) — clean sci-plot palette

make_panel_a_placeholder <- function() {
  rects <- tibble::tribble(
    ~xmin, ~xmax, ~ymin, ~ymax, ~label, ~fill, ~tcol,
    0.02, 0.21, 0.62, 0.84, "Baseline\nphospho state", "#C1D8E9", "grey15",
    0.02, 0.21, 0.16, 0.38, "Drug target\n(multi-hot)", "#F6C8B6", "grey15",
    0.34, 0.52, 0.55, 0.84, "Baseline\nencoder",        "#92B1D9", "grey10",
    0.34, 0.52, 0.16, 0.45, "Target\nencoder",          "#E0A48F", "grey10",
    0.37, 0.52, 0.02, 0.13, "Graph prior",              "#D4D4D4", "grey15",
    0.63, 0.81, 0.28, 0.72, "Decoder",                  "#DBDDEF", "grey10",
    0.90, 1.10, 0.30, 0.70, "Δp vector",                "#FFFFFF", "grey10"
  )
  arr <- arrow(length = unit(0.08, "cm"), type = "closed")

  ggplot() +
    geom_rect(data = rects, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                fill = fill), color = "grey40", linewidth = 0.35) +
    geom_text(data = rects, aes(x = (xmin+xmax)/2, y = (ymin+ymax)/2,
                                label = label, color = tcol),
              size = 2.4, family = BASE_FONT, lineheight = 0.9) +
    annotate("segment", x = 0.21, xend = 0.34, y = 0.73, yend = 0.70,
             linewidth = 0.35, arrow = arr) +
    annotate("segment", x = 0.21, xend = 0.34, y = 0.27, yend = 0.30,
             linewidth = 0.35, arrow = arr) +
    annotate("segment", x = 0.445, xend = 0.445, y = 0.13, yend = 0.16,
             linewidth = 0.3) +
    annotate("segment", x = 0.52, xend = 0.63, y = 0.68, yend = 0.58,
             linewidth = 0.4, arrow = arr) +
    annotate("segment", x = 0.52, xend = 0.63, y = 0.32, yend = 0.45,
             linewidth = 0.4, arrow = arr) +
    annotate("segment", x = 0.81, xend = 0.90, y = 0.50, yend = 0.50,
             linewidth = 0.5, arrow = arr) +
    scale_fill_identity() + scale_color_identity() +
    scale_x_continuous(limits = c(0, 1.14), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 0.90), expand = c(0, 0)) +
    annotate("text", x = 0.56, y = 0.89, label = "PPKO architecture",
             fontface = "bold", size = 3.0, family = BASE_FONT, color = "grey10") +
    annotate("text", x = 1.00, y = 0.20, label = "BioRender placeholder",
             size = 1.9, color = "grey60", fontface = "italic", family = BASE_FONT) +
    theme_void(base_family = BASE_FONT) +
    theme(plot.margin = margin(4, 4, 4, 4))
}
