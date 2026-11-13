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

The checked-in notebook is executed through a 50-epoch checkpoint on all 100,000
sketch cells. Add later values to `SNAPSHOT_EPOCHS` for a longer analysis,
preferably using a GPU, and validate biological plausibility before reporting
the resulting annotations.

Generated downloads, prepared matrices, synthetic data, model artifacts, and
figures are stored under this example's `data/` and `results/` directories and
are ignored by Git.

Official resources:

- [10x Xenium cell-type annotation guide](https://www.10xgenomics.com/analysis-guides/xenium-cell-type-annotation)
- [Xenium Prime FFPE human ovarian-cancer data](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer)
- [17k human ovarian-cancer scFFPE reference](https://www.10xgenomics.com/datasets/17k-human-ovarian-cancer-scFFPE)
