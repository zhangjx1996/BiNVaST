# Upload-friendly toy data

This directory contains a 10,000-cell spatial query and a balanced 1,200-cell
single-cell reference derived from the public 10x Genomics ovarian-cancer
datasets. Every source file is below 10 MB and can be uploaded directly to
GitHub.

The spatial data retain 5,101 raw genes. The reference retains 18,082 raw genes
and up to 75 QC-passing cells from each of 16 cell types. Sampling uses seed 123.
These files are intended for examples and software testing, not as a substitute
for the full datasets in a biological analysis.

## Saved data format

A dense expression CSV stores every zero and exceeds the upload limit. The toy
data therefore use a standard sparse text layout:

- Raw counts are stored in **Matrix Market text** (`.mtx.gz`). Each nonzero
  integer count is represented by its 1-based gene row, cell column, and value.
- Gene and cell identifiers are stored as compressed **TSV text** files.
- Spatial coordinates and reference labels are compressed **CSV text** files.
- Gzip compression (`.gz`) reduces storage size without changing the underlying
  Matrix Market, TSV, or CSV formats.

The complete 10,000-cell spatial count matrix fits in one 5.367 MiB file, so no
splitting or RDS fallback is required.

## File layout

| File | Format | Contents |
| --- | --- | --- |
| `toy_spatial_counts.mtx.gz` | Matrix Market + gzip | 5,101 genes × 10,000 cells sparse raw counts |
| `toy_spatial_barcodes.tsv.gz` | TSV + gzip | Spatial cell IDs in count-matrix column order |
| `toy_spatial_features.tsv.gz` | TSV + gzip | Spatial `gene_id` and `gene_name` rows |
| `toy_spatial_coordinates.csv.gz` | CSV + gzip | `cell_id`, `x`, and `y` for all 10,000 cells |
| `toy_reference_counts.mtx.gz` | Matrix Market + gzip | 18,082 genes × 1,200 cells sparse raw counts |
| `toy_reference_barcodes.tsv.gz` | TSV + gzip | Reference cell IDs in count-matrix column order |
| `toy_reference_features.tsv.gz` | TSV + gzip | Reference `gene_id` and `gene_name` rows |
| `toy_reference_labels.csv.gz` | CSV + gzip | Reference `cell_id` and `label` |
| `toy_dataset_summary.csv` | CSV | Dataset dimensions and sampling description |
| `toy_dataset_manifest.csv` | CSV | Per-file format, byte size, MiB size, and MD5 checksum |

## Prepare BiNVaST inputs

The repository's `prepare_10x_guide_data.R` script detects this layout, imports
both sparse matrices, selects 2,000 shared variable genes, and writes the
standardized BiNVaST inputs:

```bash
Rscript examples/xenium_10x_guide/prepare_10x_guide_data.R \
  examples/xenium_10x_guide/toy_data \
  examples/xenium_10x_guide/toy_preprocessed \
  .
```

The generated dense `target_expression_log1p_cp10k.csv` is approximately
49 MB for 10,000 cells. It is a local intermediate consumed by Python and must
not be committed to GitHub. Only the sparse source files in this directory are
designed to satisfy the per-file upload limit.

## Import manually in R

```r
library(Matrix)
library(data.table)

connection <- gzfile("toy_spatial_counts.mtx.gz", "rb")
spatial_counts <- readMM(connection)
close(connection)

spatial_features <- fread("toy_spatial_features.tsv.gz")
spatial_barcodes <- fread("toy_spatial_barcodes.tsv.gz")
rownames(spatial_counts) <- spatial_features$gene_name
colnames(spatial_counts) <- spatial_barcodes$cell_id
spatial_counts <- as(spatial_counts, "CsparseMatrix")

spatial_coordinates <- fread("toy_spatial_coordinates.csv.gz")
```

The reference follows the same pattern with its reference-named files.
`make_toy_10x_data.R` recreates the bundle from the original 10x downloads and
stops if any generated source file reaches 10 MiB.
