# BinVAST: query-adaptive annotation for high-resolution spatial transcriptomics

BinVAST is a spatially informed deep-learning workflow for transferring cell-type information from a pre-annotated single-cell reference to high-resolution spatial transcriptomics data. The pipeline aligns reference and query expression distributions, constructs query-matched synthetic training cells, and combines expression and spatial-graph representations in a dual-branch variational autoencoder with supervised classification.

The current implementation is designed for high-resolution platforms such as Xenium, CosMx, Atera, and Visium HD. Data are intentionally not included in this repository.

## Model overview

![BinVAST model overview](images/Figure1.png)

The original vector-quality figure is available as [Figure1.pdf](images/Figure1.pdf).

The workflow has four major stages:

1. Select shared, highly variable genes and export aligned query/reference matrices.
2. Adjust the reference distribution to the query platform and generate query-matched synthetic cells.
3. Train expression and spatial-graph VAE branches with an internal spatial validation split.
4. Predict cell labels and probability scores for the query tissue.

## Repository structure

```text
BiNVaST_GitHub/
├── R/                  # Query/reference preprocessing and gene selection
├── python/             # Synthetic-data generation, model training, and prediction
├── images/             # Model overview in PNG and PDF formats
├── requirements.txt
└── README.md
```

## R preprocessing

The R code is used only for preprocessing: it selects shared variable genes from the raw query data and the pre-annotated single-cell reference, then writes the aligned expression matrices, spatial coordinates, and reference labels required by the Python pipeline.

Use `R/select_shared_vst_seurat.R` to apply Seurat VST feature selection and generate the complete input bundle for the Python pipeline.

Both scripts produce the following core inputs:

```text
target_expression_log1p_cp10k.csv
target_coordinates.csv
reference_expression_log1p_cp10k.csv
reference_labels.csv
```

Edit the `RSTUDIO_SETTINGS` block, set `RUN_R_STUDIO_WORKFLOW <- TRUE`, and source the selected script in RStudio. See [R/README.md](R/README.md) for the expected input and output formats.

## Python setup

### Dependencies

BinVAST is implemented in Python with PyTorch. Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build appropriate for the CUDA version on your compute system before installing the remaining requirements. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Python usage

### Step 1: generate query-matched synthetic cells

```bash
python python/generate_synthetic_csv.py \
  --input-data-dir /path/to/binvast_preprocessed_input \
  --output-dir /path/to/binvast_synthetic_input \
  --pca-base reference \
  --n-pcs 50 \
  --tau 1.0 \
  --seed 259
```

The default strategy fits PCA to the globally aligned reference. Use `--pca-base target` to fit PCA to the query and project the aligned reference into query PCA space. In both modes, the script uses nearest-reference-cell distances within each cell type, Gaussian-kernel label sampling, and dropout-only zero alignment.

Synthetic-data outputs:

```text
synthetic_expression.csv
synthetic_labels.csv
synthetic_spatial_coords.csv
synthetic_cell_type_counts.csv
```

### Step 2: train BinVAST and annotate query cells

```bash
python python/run_binvast_continuous_snapshots.py \
  --project-dir /path/to/binvast_project \
  --target-data-dir /path/to/binvast_preprocessed_input \
  --synthetic-data-dir /path/to/binvast_synthetic_input \
  --output-dir /path/to/binvast_output \
  --snapshot-epochs 50 100 150 200 \
  --batch-size 512 \
  --graph-k 30 \
  --expr-latent-dim 4 \
  --graph-latent-dim 4 \
  --seed 259
```

Training follows one continuous trajectory up to the largest requested epoch and saves exact model, latent, prediction, and summary snapshots at each requested epoch.
The command directly trains the model, predicts a cell type for every query cell, and exports the final label-annotation table. No cluster submission wrapper is required.

## Input data

All expression matrices are cells by genes with `cell_id` in the first column. Query and reference matrices must contain the same selected genes in the same order.

| File | Required columns/content |
| --- | --- |
| `target_expression_log1p_cp10k.csv` | `cell_id`, followed by selected genes |
| `target_coordinates.csv` | `cell_id`, `x`, `y` |
| `reference_expression_log1p_cp10k.csv` | `cell_id`, followed by the same selected genes |
| `reference_labels.csv` | `cell_id`, `label` |

## Main parameters

### Synthetic-data generation

| Parameter | Description | Default |
| --- | --- | --- |
| `--pca-base` | Dataset used to fit PCA: `reference` or `target` | `reference` |
| `--n-pcs` | Number of principal components | `50` |
| `--tau` | Gaussian-kernel temperature used for label sampling | `1.0` |
| `--seed` | Random seed | `259` |
| `--csv-chunk-size` | Rows processed per CSV chunk | `2000` |
| `--assignment-batch-size` | Query cells processed per assignment batch | `10000` |

### BinVAST training

| Parameter | Description | Default |
| --- | --- | --- |
| `--snapshot-epochs` | Epochs at which exact outputs are saved | required |
| `--batch-size` | Training batch size | required |
| `--graph-k` | Spatial nearest neighbors used to construct the graph | `30` |
| `--expr-latent-dim` | Expression VAE latent dimension | `4` |
| `--graph-latent-dim` | Graph VAE latent dimension | `4` |
| `--seed` | Random seed | `259` |

Additional architecture and loss settings are defined in `TrainConfig` in `python/binvast_model_core.py`.

## Output

Each epoch/batch-size snapshot is written to a directory such as:

```text
epochs_050_batch_00512/
epochs_100_batch_00512/
```

Outputs include trained checkpoints, preprocessing metadata, latent representations, per-cell prediction scores, predicted labels, cell-type counts, and percentages. The final per-cell annotation is written inside each snapshot directory as:

```text
target_predicted_labels.csv
```

The pipeline also writes an aggregate table:

```text
target_cell_type_proportions_by_epoch_and_batch.csv
```

## Citation

If you use BinVAST in your research, please cite the associated manuscript. Citation details will be added when the manuscript is available.

## License

No license has been assigned yet. Add a license before public release if reuse and redistribution should be permitted.
