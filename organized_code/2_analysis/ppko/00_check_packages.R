# 检查 Fig 4 出图所需 R 包是否齐备
pkgs <- c(
  "ggplot2", "dplyr", "tidyr", "readr", "ggdist", "ggExtra",
  "patchwork", "RColorBrewer", "scales", "cowplot",
  "circlize", "ComplexHeatmap", "pROC", "ggridges",
  "stringr", "forcats", "grid", "gridExtra"
)
status <- sapply(pkgs, function(p) requireNamespace(p, quietly = TRUE))
cat("\n--- R 包状态 ---\n")
for (p in names(status)) {
  cat(sprintf("%-20s %s\n", p, ifelse(status[p], "OK", "MISSING")))
}
missing_pkgs <- names(status)[!status]
cat("\n缺失列表:\n")
cat(paste(missing_pkgs, collapse = ", "), "\n")
cat("\n缺失数量:", length(missing_pkgs), "\n")
