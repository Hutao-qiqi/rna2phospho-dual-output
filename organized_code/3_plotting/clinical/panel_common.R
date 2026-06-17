suppressMessages({
  library(ggplot2)
  library(cowplot)
  library(grid)
  library(scales)
})

source("E:/data/gongke/TCGA-TCPA/paper_final/fig5/scripts/panels/panel_fig2_style.R")

.FIG5_ROOT <- "E:/data/gongke/TCGA-TCPA/paper_final/fig5"
.FIG5_DATA <- file.path(.FIG5_ROOT, "source_data", "tables")

COL_BLUE <- FIG2_BLUE
COL_WARM <- FIG2_ORANGE
COL_GREY <- FIG2_COL_GRID
COL_PURPLE <- "#DBCB92"
COL_TEXT <- FIG2_COL_TEXT

theme_fig5 <- function(base_size = 7) {
  fig2_panel_theme(base_size = base_size)
}

p_text <- function(p) {
  if (is.na(p)) return("p=NA")
  if (p < 1e-4) return(sprintf("p=%.1e", p))
  if (p < 1e-3) return(sprintf("p=%.2e", p))
  sprintf("p=%.3f", p)
}

zscore <- function(x) {
  x <- as.numeric(x)
  (x - mean(x, na.rm = TRUE)) / stats::sd(x, na.rm = TRUE)
}

read_matrix_tsv <- function(path) {
  d <- read.delim(path, sep = "\t", check.names = FALSE, stringsAsFactors = FALSE)
  rn <- d[[1]]
  d <- d[, -1, drop = FALSE]
  rownames(d) <- rn
  as.data.frame(lapply(d, as.numeric), check.names = FALSE, row.names = rownames(d))
}

matrix_to_long <- function(mat, row_name = "row_id", col_name = "column_id", value_name = "value") {
  out <- data.frame(
    row_id = rep(rownames(mat), times = ncol(mat)),
    column_id = rep(colnames(mat), each = nrow(mat)),
    value = as.vector(as.matrix(mat)),
    stringsAsFactors = FALSE
  )
  names(out) <- c(row_name, col_name, value_name)
  out
}

save_panel <- function(plot, stem, width, height, out_dir = file.path(.FIG5_ROOT, "figures")) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  pdf_file <- file.path(out_dir, paste0(stem, ".pdf"))
  png_file <- file.path(out_dir, paste0(stem, ".png"))
  svg_file <- file.path(out_dir, paste0(stem, ".svg"))
  cairo_pdf(pdf_file, width = width, height = height, family = "Arial")
  print(plot)
  invisible(dev.off())
  ggsave(png_file, plot, width = width, height = height, dpi = 450, bg = "white")
  svglite::svglite(svg_file, width = width, height = height, system_fonts = list(sans = "Arial"))
  print(plot)
  invisible(dev.off())
  message("wrote ", png_file)
  invisible(c(pdf = pdf_file, png = png_file, svg = svg_file))
}

half_violin_data <- function(values, xpos, width = 0.085, n = 180) {
  values <- values[is.finite(values)]
  if (length(values) < 5 || stats::sd(values) == 0) return(NULL)
  den <- stats::density(values, n = n, na.rm = TRUE)
  sx <- den$y / max(den$y) * width
  data.frame(
    x = c(rep(xpos, length(den$x)), xpos - rev(sx)),
    y = c(den$x, rev(den$x))
  )
}

raincloud_plot <- function(data, group_col, value_col, order, colors, title, ylab = NULL) {
  d <- data[, c(group_col, value_col)]
  colnames(d) <- c("group", "value")
  d <- d[is.finite(d$value) & !is.na(d$group), ]
  d$group <- factor(d$group, levels = order)
  d <- d[!is.na(d$group), ]
  xpos <- if (length(order) == 2) c(1.00, 1.42) else seq_along(order)
  names(xpos) <- order
  d$xpos <- unname(xpos[as.character(d$group)])

  dens <- do.call(
    rbind,
    lapply(seq_along(order), function(i) {
      out <- half_violin_data(d$value[d$group == order[i]], xpos[i], width = 0.075)
      if (is.null(out)) return(NULL)
      out$group <- order[i]
      out
    })
  )

  vals <- lapply(order, function(g) d$value[d$group == g])
  p <- if (length(vals) == 2) stats::wilcox.test(vals[[1]], vals[[2]])$p.value else NA_real_
  shift <- if (length(vals) == 2) median(vals[[2]]) - median(vals[[1]]) else NA_real_
  stat_label <- if (length(vals) == 2) sprintf("shift=%.2f; %s", shift, p_text(p)) else ""

  set.seed(20260531)
  jittered <- d
  jittered$x <- jittered$xpos + 0.044 + stats::rnorm(nrow(jittered), 0, 0.010)

  p0 <- ggplot() +
    geom_polygon(
      data = dens,
      aes(x = x, y = y, group = group, fill = group, color = group),
      alpha = 0.32, linewidth = 0.25, show.legend = FALSE
    ) +
    geom_boxplot(
      data = d,
      aes(x = xpos, y = value, group = group),
      width = 0.068, outlier.shape = NA, fill = "white", color = COL_TEXT,
      linewidth = 0.32
    ) +
    geom_point(
      data = jittered,
      aes(x = x, y = value, color = group),
      size = ifelse(nrow(d) > 300, 0.65, 0.85), alpha = 0.72,
      stroke = 0.08, show.legend = FALSE
    ) +
    annotate(
      "text", x = max(xpos) + 0.16, y = Inf, label = stat_label,
      hjust = 1, vjust = 1.35, family = "Arial", size = 2.05, color = COL_TEXT
    ) +
    scale_fill_manual(values = setNames(colors, order)) +
    scale_color_manual(values = setNames(colors, order)) +
    scale_x_continuous(
      breaks = xpos,
      labels = paste0(order, "\n", "n=", vapply(vals, length, integer(1))),
      limits = c(min(xpos) - 0.16, max(xpos) + 0.20),
      expand = expansion(mult = c(0, 0))
    ) +
    labs(title = title, x = NULL, y = ylab) +
    theme_fig5(base_size = 7) +
    theme(axis.text.x = element_text(size = 6.4, lineheight = 0.9))
  attr(p0, "stats") <- data.frame(panel = title, p = p, shift = shift)
  p0
}
