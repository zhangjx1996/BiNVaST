# Upload-friendly toy data

These two files are compressed sparse R bundles derived from the public 10x
Genomics ovarian-cancer datasets used by the Xenium cell-type annotation guide.
They are small enough to commit directly to GitHub and retain raw integer counts,
cell identifiers, gene names, and the metadata required by the R preprocessing
workflow.

| File | Contents | Size |
| --- | --- | ---: |
| `toy_spatial.rds` | 5,101 genes × 5,000 spatial cells, plus x/y coordinates | 1.09 MiB |
| `toy_single_cell_reference.rds` | 18,082 genes × 1,200 cells, plus 16 balanced labels | 1.50 MiB |

The spatial cells are a seeded random sample of positive-library Xenium cells.
The reference contains up to 75 QC-passing cells from each cell type, using the
guide's thresholds: `200 < nCount < 10000` and mitochondrial percentage below
10%. These files are intended for examples and tests, not for reproducing the
full-data biological result.

## Inspect the files in R

```r
library(Matrix)

spatial <- readRDS("toy_spatial.rds")
dim(spatial$counts)
head(spatial$coordinates)

reference <- readRDS("toy_single_cell_reference.rds")
dim(reference$counts)
table(reference$labels$label)
```

Both `counts` objects are gene-by-cell `dgCMatrix` sparse matrices. The spatial
bundle has a `coordinates` data frame with `cell_id`, `x`, and `y`. The reference
bundle has a `labels` data frame with `cell_id` and `label`.

## Prepare BiNVaST input files

From the repository root, run:

```bash
Rscript examples/xenium_10x_guide/prepare_10x_guide_data.R \
  examples/xenium_10x_guide/toy_data \
  examples/xenium_10x_guide/toy_preprocessed \
  .
```

The R script detects the toy RDS format automatically, uses all 5,000 spatial
cells, selects 2,000 shared variable genes, and writes the four standardized
CSV inputs expected by the Python package. Generated CSV files are local
intermediate files and should not be committed.

`make_toy_10x_data.R` recreates the bundles from the original downloaded 10x
files. File dimensions, checksums, and byte sizes are recorded in
`toy_dataset_manifest.csv`.
