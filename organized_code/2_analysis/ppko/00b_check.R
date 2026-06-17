pkgs <- c("pheatmap", "ggbeeswarm", "ggrepel", "paletteer", "ggdist",
          "patchwork", "cowplot", "viridisLite", "pROC", "scales")
status <- sapply(pkgs, function(p) requireNamespace(p, quietly = TRUE))
for (p in names(status)) cat(sprintf("%-14s %s\n", p, ifelse(status[p], "OK", "MISSING")))
cat("\nMISSING:", paste(names(status)[!status], collapse = ", "), "\n")
