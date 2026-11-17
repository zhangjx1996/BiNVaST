# 10x Xenium ovarian-cancer example

The executed notebook `xenium_ovarian_10x_binvast.ipynb` downloads the public
Xenium Prime FFPE human ovarian-cancer query and matched 17k Flex scRNA-seq
reference used in the 10x Genomics Xenium cell-type annotation guide. It
reproduces the guide's 100,000-cell Seurat leverage-score sketch, prepares the
BiNVaST inputs, generates synthetic training cells, trains the model, predicts
query labels, and visualizes the predictions in tissue space.

The companion `prepare_10x_guide_data.R` script requires Seurat, Matrix, and
data.table. The notebook requires the local BiNVaST package plus Jupyter,
Matplotlib, and the Python dependencies declared by BiNVaST.

## Small data that can be uploaded to GitHub

The `toy_data/` directory contains two compressed sparse RDS bundles:

- `toy_spatial.rds`: 5,000 Xenium cells with raw counts and coordinates
  (1.09 MiB).
- `toy_single_cell_reference.rds`: 1,200 QC-passing single cells, balanced
  across all 16 reference labels (1.50 MiB).

Prepare the toy inputs from the repository root with:

```bash
Rscript examples/xenium_10x_guide/prepare_10x_guide_data.R \
  examples/xenium_10x_guide/toy_data \
  examples/xenium_10x_guide/toy_preprocessed \
  .
```

The script automatically imports the sparse RDS bundles and uses all toy
spatial cells. If the directory instead contains the original 10x H5/CSV files,
the same script retains the full 100,000-cell leverage-score sketch workflow.

The checked-in notebook is executed through a 50-epoch checkpoint on all 100,000
sketch cells. Add later values to `SNAPSHOT_EPOCHS` for a longer analysis,
preferably using a GPU, and validate biological plausibility before reporting
the resulting annotations.

Generated downloads, prepared matrices, synthetic data, model artifacts, and
figures are stored under this example's `data/`, `toy_preprocessed/`, and
`results/` directories and are ignored by Git. The two small source bundles in
`toy_data/` are intentionally tracked.

Official resources:

- [10x Xenium cell-type annotation guide](https://www.10xgenomics.com/analysis-guides/xenium-cell-type-annotation)
- [Xenium Prime FFPE human ovarian-cancer data](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer)
- [17k human ovarian-cancer scFFPE reference](https://www.10xgenomics.com/datasets/17k-human-ovarian-cancer-scFFPE)
