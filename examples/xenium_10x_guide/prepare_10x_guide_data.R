#!/usr/bin/env Rscript

# Prepare the 10x ovarian-cancer example for BiNVaST. RAW_DIR may contain either
# the two small GitHub toy RDS bundles or the original full 10x files. In toy
# mode all 5,000 spatial cells are used; full mode reproduces the guide's 100K
# Seurat leverage-score sketch.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop(
    "Usage: Rscript prepare_10x_guide_data.R RAW_DIR OUTPUT_DIR REPO_ROOT"
  )
}

raw_dir <- normalizePath(path.expand(args[[1L]]), mustWork = TRUE)
output_dir <- path.expand(args[[2L]])
repo_root <- normalizePath(path.expand(args[[3L]]), mustWork = TRUE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

source(file.path(repo_root, "R", "select_shared_vst_seurat.R"))

toy_spatial_file <- file.path(raw_dir, "toy_spatial.rds")
toy_reference_file <- file.path(raw_dir, "toy_single_cell_reference.rds")
if (xor(file.exists(toy_spatial_file), file.exists(toy_reference_file))) {
  stop(
    "Toy mode requires both toy_spatial.rds and ",
    "toy_single_cell_reference.rds in RAW_DIR"
  )
}
toy_mode <- file.exists(toy_spatial_file) && file.exists(toy_reference_file)

target_h5 <- file.path(
  raw_dir,
  "Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_cell_feature_matrix.h5"
)
cells_csv <- file.path(raw_dir, "cells.csv.gz")
reference_h5 <- file.path(
  raw_dir,
  "17k_Ovarian_Cancer_scFFPE_count_filtered_feature_bc_matrix.h5"
)
annotation_csv <- file.path(
  raw_dir,
  "FLEX_Ovarian_Barcode_Cluster_Annotation.csv"
)

required_files <- if (toy_mode) {
  c(toy_spatial_file, toy_reference_file)
} else {
  c(target_h5, cells_csv, reference_h5, annotation_csv)
}
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files)) {
  stop("Missing downloaded files: ", paste(missing_files, collapse = ", "))
}

if (toy_mode) {
  message_now("Loading compressed toy spatial bundle")
  toy_spatial <- readRDS(toy_spatial_file)
  if (!is.list(toy_spatial) ||
      !all(c("counts", "coordinates") %in% names(toy_spatial))) {
    stop("toy_spatial.rds must contain counts and coordinates")
  }
  target_counts <- toy_spatial$counts
  toy_coordinates <- data.table::as.data.table(toy_spatial$coordinates)
} else {
  message_now("Loading full Xenium expression matrix")
  target_counts <- get_gene_expression_matrix(
    Seurat::Read10X_h5(target_h5, use.names = TRUE, unique.features = TRUE)
  )
}
target_counts <- methods::as(target_counts, "dgCMatrix")
positive_target <- Matrix::colSums(target_counts) > 0
target_counts <- target_counts[, positive_target, drop = FALSE]

sketch_file <- file.path(output_dir, "sketch_cell_ids.csv")
if (toy_mode) {
  sketch_cells <- colnames(target_counts)
  sketch_method <- "all_toy_cells"
  data.table::fwrite(
    data.table::data.table(cell_id = sketch_cells),
    sketch_file
  )
  message_now("Using all ", length(sketch_cells), " toy spatial cells")
} else if (file.exists(sketch_file)) {
  message_now("Reusing the existing 100,000-cell sketch IDs")
  sketch_cells <- data.table::fread(
    sketch_file,
    colClasses = "character"
  )$cell_id
  if (!all(sketch_cells %in% colnames(target_counts))) {
    stop("Existing sketch IDs are not all present in the target matrix")
  }
} else {
  message_now("Creating the 100,000-cell Seurat leverage-score sketch")
  target_object <- Seurat::CreateSeuratObject(
    counts = target_counts,
    assay = "Xenium",
    project = "Xenium_Prime_Ovarian_Cancer"
  )
  Seurat::DefaultAssay(target_object) <- "Xenium"
  target_object <- Seurat::NormalizeData(target_object, verbose = FALSE)
  target_object <- Seurat::FindVariableFeatures(target_object, verbose = FALSE)
  target_object <- Seurat::SketchData(
    object = target_object,
    assay = "Xenium",
    ncells = 100000L,
    method = "LeverageScore",
    sketched.assay = "sketch",
    seed = 123L,
    verbose = TRUE
  )
  sketch_cells <- colnames(target_object[["sketch"]])
  sketch_method <- "LeverageScore"
  data.table::fwrite(
    data.table::data.table(cell_id = sketch_cells),
    sketch_file
  )
  rm(target_object)
  invisible(gc())
}
if (!toy_mode && !exists("sketch_method")) sketch_method <- "LeverageScore"
if (!toy_mode && length(sketch_cells) != 100000L) {
  stop("Expected 100,000 sketch cells; observed ", length(sketch_cells))
}

if (toy_mode) {
  message_now("Loading compressed toy single-cell reference bundle")
  toy_reference <- readRDS(toy_reference_file)
  if (!is.list(toy_reference) ||
      !all(c("counts", "labels") %in% names(toy_reference))) {
    stop("toy_single_cell_reference.rds must contain counts and labels")
  }
  reference_counts <- methods::as(toy_reference$counts, "dgCMatrix")
  toy_labels <- data.table::as.data.table(toy_reference$labels)
  if (!all(c("cell_id", "label") %in% names(toy_labels))) {
    stop("Toy reference labels must contain cell_id and label")
  }
  label_lookup <- setNames(toy_labels$label, toy_labels$cell_id)
  reference_labels_all <- unname(label_lookup[colnames(reference_counts)])
} else {
  message_now("Loading the matched full Flex scRNA-seq reference")
  reference_counts <- get_gene_expression_matrix(
    Seurat::Read10X_h5(reference_h5, use.names = TRUE, unique.features = TRUE)
  )
  reference_counts <- methods::as(reference_counts, "dgCMatrix")
  annotations <- data.table::fread(annotation_csv, colClasses = "character")
  annotation_label_column <- intersect(
    c("Cell Annotation", "Cell.Annotation"),
    names(annotations)
  )
  if (!"Barcode" %in% names(annotations) || !length(annotation_label_column)) {
    stop("Reference annotation CSV lacks Barcode or Cell Annotation")
  }
  label_lookup <- setNames(
    annotations[[annotation_label_column[[1L]]]],
    annotations$Barcode
  )
  reference_labels_all <- unname(label_lookup[colnames(reference_counts)])
}

# Match the guide's reference QC: 200 < UMI counts < 10,000 and <10% MT.
reference_library_size <- Matrix::colSums(reference_counts)
mitochondrial_genes <- grepl("^MT-", rownames(reference_counts))
if (any(mitochondrial_genes)) {
  mitochondrial_counts <- Matrix::colSums(
    reference_counts[mitochondrial_genes, , drop = FALSE]
  )
} else {
  mitochondrial_counts <- rep(0, ncol(reference_counts))
}
percent_mt <- 100 * mitochondrial_counts / pmax(reference_library_size, 1)
keep_reference <-
  reference_library_size > 200 &
  reference_library_size < 10000 &
  percent_mt < 10 &
  !is.na(reference_labels_all) &
  nzchar(reference_labels_all)
reference_cells <- colnames(reference_counts)[keep_reference]
reference_labels <- reference_labels_all[keep_reference]
reference_counts <- reference_counts[, reference_cells, drop = FALSE]

shared_genes <- rownames(target_counts)[
  rownames(target_counts) %in% rownames(reference_counts)
]
message_now("Shared genes: ", length(shared_genes))
if (length(shared_genes) < 2000L) {
  stop("Fewer than 2,000 shared genes are available")
}

message_now("Selecting 2,000 shared query genes with Seurat VST")
sketch_shared_counts <- target_counts[
  shared_genes,
  sketch_cells,
  drop = FALSE
]
vst_object <- Seurat::CreateSeuratObject(
  counts = sketch_shared_counts,
  assay = "RNA",
  project = if (toy_mode) "Ovarian_Toy_Spatial" else "Xenium_Ovarian_100K_Sketch"
)
set.seed(123L)
vst_object <- Seurat::FindVariableFeatures(
  vst_object,
  assay = "RNA",
  selection.method = "vst",
  nfeatures = 2000L,
  verbose = FALSE
)
selected_genes <- Seurat::VariableFeatures(vst_object, assay = "RNA")
if (length(selected_genes) != 2000L) {
  stop("Expected 2,000 selected genes; observed ", length(selected_genes))
}
data.table::fwrite(
  data.table::data.table(gene = selected_genes),
  file.path(output_dir, "top2000_gene_names.csv")
)
rm(vst_object, sketch_shared_counts)
invisible(gc())

message_now("Aligning sketch coordinates")
if (toy_mode) {
  if (!all(c("cell_id", "x", "y") %in% names(toy_coordinates))) {
    stop("Toy spatial coordinates must contain cell_id, x, and y")
  }
  coordinate_index <- match(sketch_cells, toy_coordinates$cell_id)
  if (anyNA(coordinate_index)) stop("Toy coordinates are missing sketch cells")
  coordinates <- toy_coordinates[coordinate_index, .(cell_id, x, y)]
} else {
  coordinates <- read_target_coordinates(cells_csv, sketch_cells)
}
data.table::fwrite(
  coordinates[, .(cell_id, x, y)],
  file.path(output_dir, "target_coordinates.csv")
)
data.table::fwrite(
  data.table::data.table(
    cell_id = reference_cells,
    label = reference_labels
  ),
  file.path(output_dir, "reference_labels.csv")
)

message_now("Writing target sketch expression")
write_expression_matrix_csv(
  counts = target_counts,
  genes = selected_genes,
  cells = sketch_cells,
  output_file = file.path(
    output_dir,
    "target_expression_log1p_cp10k.csv"
  ),
  expression_scale = "log1p_cp10k",
  chunk_size = 500L
)

message_now("Writing filtered reference expression")
write_expression_matrix_csv(
  counts = reference_counts,
  genes = selected_genes,
  cells = reference_cells,
  output_file = file.path(
    output_dir,
    "reference_expression_log1p_cp10k.csv"
  ),
  expression_scale = "log1p_cp10k",
  chunk_size = 500L
)

label_counts <- data.table::data.table(label = reference_labels)[
  , .(cells = .N), by = label
][order(-cells, label)]
data.table::fwrite(
  label_counts,
  file.path(output_dir, "reference_label_counts.csv")
)

summary <- data.table::data.table(
  metric = c(
    "input_mode",
    "full_xenium_cells",
    "sketch_cells",
    "xenium_genes",
    "reference_cells_before_qc",
    "reference_cells_after_qc",
    "reference_labels",
    "shared_genes",
    "selected_genes",
    "sketch_method",
    "sketch_seed"
  ),
  value = c(
    if (toy_mode) "github_toy_rds" else "full_10x_h5",
    ncol(target_counts),
    length(sketch_cells),
    nrow(target_counts),
    length(reference_labels_all),
    length(reference_cells),
    length(unique(reference_labels)),
    length(shared_genes),
    length(selected_genes),
    sketch_method,
    123L
  )
)
data.table::fwrite(summary, file.path(output_dir, "preparation_summary.csv"))
writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))

message_now("BiNVaST input preparation completed")
print(summary)
