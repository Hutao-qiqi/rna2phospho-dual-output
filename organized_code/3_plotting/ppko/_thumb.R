# Generate thumbnail PNGs for inspection (smaller dpi)
script_dir <- "E:/data/gongke/TCGA-TCPA/02_results/figure_outputs/fig4_v3/scripts"
source(file.path(script_dir, "01_config.R"))
source(file.path(script_dir, "02_load_data.R"))
source(file.path(script_dir, "10_panel_i.R"))
source(file.path(script_dir, "05_panel_d.R"))
source(file.path(script_dir, "04_panel_c.R"))

d <- load_fig4_data()

thumb_dir <- file.path(PANEL_DIR, "thumbs")
dir.create(thumb_dir, showWarnings = FALSE, recursive = TRUE)

ggsave(file.path(thumb_dir, "panel_i_thumb.png"),
       make_panel_i(d), width = 7.5, height = 3.2, units = "in", dpi = 130, bg = "white")
ggsave(file.path(thumb_dir, "panel_d_thumb.png"),
       make_panel_d(d), width = 4.8, height = 5.6, units = "in", dpi = 130, bg = "white")
ggsave(file.path(thumb_dir, "panel_c_thumb.png"),
       make_panel_c(d), width = 6.4, height = 3.4, units = "in", dpi = 130, bg = "white")

cat("thumbs saved\n")
