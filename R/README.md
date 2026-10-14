# R preprocessing

The R code prepares raw query spatial-transcriptomics data for BinVAST. It selects shared variable genes, aligns the query with a pre-annotated single-cell reference, and exports the expression, coordinate, and label tables consumed by the Python workflow.

Use `select_shared_vst_seurat.R` for the Seurat `FindVariableFeatures(..., selection.method = "vst")` workflow and complete input-data preparation.

Before running the script, edit `RSTUDIO_SETTINGS`, keep `RUN_R_STUDIO_WORKFLOW <- FALSE` while checking paths, then set it to `TRUE` and source the script.

Required R packages:

```r
install.packages(c("data.table", "Rcpp"))
install.packages("Seurat")
```

`Matrix` is also required and is normally installed as a dependency.

The Python workflow expects these four outputs:

```text
target_expression_log1p_cp10k.csv
target_coordinates.csv
reference_expression_log1p_cp10k.csv
reference_labels.csv
```

The expression tables must be cells by genes, must use `cell_id` as the first column, and must contain the same selected genes in the same order.
