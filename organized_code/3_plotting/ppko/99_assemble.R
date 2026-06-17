# Assemble clean 8-panel Fig 4 v1.0 (sci-plot style). Panel a-h.
# V10B/V10 panel removed; former panel i -> panel h.

script_dir <- "E:/data/gongke/TCGA-TCPA/02_results/figure_outputs/fig4_v3/scripts"
source(file.path(script_dir, "01_config.R"))
source(file.path(script_dir, "02_load_data.R"))
source(file.path(script_dir, "03_panel_b.R"))
source(file.path(script_dir, "04_panel_c.R"))
source(file.path(script_dir, "05_panel_d.R"))
source(file.path(script_dir, "06_panel_e.R"))
source(file.path(script_dir, "07_panel_f.R"))
source(file.path(script_dir, "08_panel_g.R"))
source(file.path(script_dir, "09_panel_h.R"))
source(file.path(script_dir, "11_panel_a_placeholder.R"))
suppressPackageStartupMessages({ library(cowplot); library(grid) })

d <- load_fig4_data()

p_a <- make_panel_a_placeholder()
p_b <- make_panel_b(d)
p_c <- make_panel_c(d)
p_d <- make_panel_d(d)
e_gt <- panel_e_gtable(d)
p_e <- cowplot::ggdraw() + cowplot::draw_grob(e_gt)
f_gt <- make_panel_f(d)
p_f <- cowplot::ggdraw() + cowplot::draw_grob(f_gt)
p_g <- make_panel_g(d)
p_h <- make_panel_h(d)

# save individual panels (vector PDF + 300 dpi PNG + thumb)
save_one <- function(p, name, w, h) {
  ggsave(file.path(PANEL_DIR, paste0(name, ".pdf")), p,
         width = w, height = h, units = "in", device = cairo_pdf)
  ggsave(file.path(PANEL_DIR, paste0(name, ".png")), p,
         width = w, height = h, units = "in", dpi = 300, bg = "white")
  ggsave(file.path(PANEL_DIR, "thumbs", paste0(name, "_thumb.png")), p,
         width = w, height = h, units = "in", dpi = 130, bg = "white")
}
save_one(p_a, "panel_a", 7.0, 1.6)
save_one(p_b, "panel_b", 3.4, 2.8)
save_one(p_c, "panel_c", 4.8, 2.8)
save_one(p_d, "panel_d", 3.6, 2.8)
draw_panel_e(d, file.path(PANEL_DIR, "panel_e.pdf"),
             file.path(PANEL_DIR, "panel_e.png"), width_in = 7.8, height_in = 2.8)
save_one(p_f, "panel_f", 3.8, 3.2)
save_one(p_g, "panel_g", 5.4, 2.9)
save_one(p_h, "panel_h", 6.8, 3.0)

LBL <- list(label_size = 13, label_fontfamily = BASE_FONT, label_fontface = "bold")

row1 <- plot_grid(p_a, labels = "a", label_size = 13,
                  label_fontfamily = BASE_FONT)
row2 <- plot_grid(p_b, p_c, labels = c("b", "c"), nrow = 1,
                  rel_widths = c(1, 1.35), label_size = 13,
                  label_fontfamily = BASE_FONT)
row3 <- plot_grid(p_d, p_e, p_f, labels = c("d", "e", "f"), nrow = 1,
                  rel_widths = c(1.05, 0.92, 1.05), label_size = 13,
                  label_fontfamily = BASE_FONT)
row4 <- plot_grid(p_g, p_h, labels = c("g", "h"), nrow = 1,
                  rel_widths = c(1, 1.28), label_size = 13,
                  label_fontfamily = BASE_FONT)

final <- plot_grid(row1, row2, row3, row4, ncol = 1,
                   rel_heights = c(0.46, 1.15, 1.5, 1.22))

final_pdf <- file.path(FINAL_DIR, "Fig4_v1.0.pdf")
final_png <- file.path(FINAL_DIR, "Fig4_v1.0.png")
ggsave(final_pdf, final, width = 9.6, height = 11.6, units = "in", device = cairo_pdf)
ggsave(final_png, final, width = 9.6, height = 11.6, units = "in", dpi = 240, bg = "white")
cat("Fig4 v1.0 assembled:\n  ", final_pdf, "\n  ", final_png, "\n")
