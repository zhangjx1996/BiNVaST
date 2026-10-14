#!/usr/bin/env Rscript

# Exact Seurat FindVariableFeatures workflow for selecting 2,000 Xenium genes
# and exporting the complete CSV input bundle for generate_synthetic_csv.py.
#
# RStudio usage:
#   1. Open this file in RStudio.
#   2. Edit SAMPLE_NAME and the paths in RSTUDIO_SETTINGS.
#   3. Set RUN_R_STUDIO_WORKFLOW <- TRUE.
#   4. Click Source.
#
# The order of operations follows the shared-gene VST strategy:
#   target raw counts -> positive-library cells -> target/reference shared genes
#   -> Seurat::FindVariableFeatures(selection.method = "vst", nfeatures = 2000)
#
# FindVariableFeatures(method = "vst") models raw counts directly. Do not call
# NormalizeData before this step; normalized data are not used by Seurat's VST
# feature-selection method.
#
# In addition to the gene lists and VST statistics, the default workflow writes:
#   target_expression_log1p_cp10k.csv
#   target_coordinates.csv
#   reference_expression_log1p_cp10k.csv
#   reference_labels.csv

# =============================================================================
# RSTUDIO USER SETTINGS
# =============================================================================
get_active_script_directory <- function() {
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    document_path <- rstudioapi::getActiveDocumentContext()$path
    if (nzchar(document_path)) {
      return(dirname(normalizePath(document_path, mustWork = TRUE)))
    }
  }
  normalizePath(getwd(), mustWork = TRUE)
}

MODEL_WRAPUP_DIR <- get_active_script_directory()
PROJECT_ROOT <- dirname(MODEL_WRAPUP_DIR)

# Keep FALSE while editing. Change to TRUE and click Source to run.
RUN_R_STUDIO_WORKFLOW <- FALSE

RSTUDIO_SETTINGS <- list(
  sample_name = "QUERY_SAMPLE",
  target_h5 = "/path/to/query/cell_feature_matrix.h5",
  # NULL uses cells.csv.gz in the same directory as target_h5.
  target_cells_file = NULL,
  output_dir = "/path/to/binvast_preprocessed_input",

  # Fast option: supply a one-column TXT/CSV file of reference gene names.
  # If NULL, genes are read from reference_rds/reference_assay.
  reference_genes_file = NULL,
  reference_rds = "/path/to/preannotated_single_cell_reference.rds",
  reference_assay = "RNA",
  reference_label_field = "merge_annotation",
  excluded_reference_labels = c(
    "Astrocytes", "cycling", "other_non_malignant"
  ),

  nfeatures = 2000L,
  loess_span = 0.3,
  random_seed = 1234L,
  verbose = TRUE,

  # Keep TRUE to write all CSVs required by generate_synthetic_csv.py.
  write_synthetic_inputs = TRUE,
  expression_scale = "log1p_cp10k",
  expression_chunk_size = 500L
)
# =============================================================================

required_packages <- c("Seurat", "Matrix", "data.table")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Install the following packages before running this script: ",
    paste(missing_packages, collapse = ", ")
  )
}

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(data.table)
})

message_now <- function(...) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - ", ...)
  flush.console()
}

read_reference_genes_file <- function(path) {
  path <- normalizePath(path.expand(path), mustWork = TRUE)
  extension <- tolower(tools::file_ext(path))

  if (extension %in% c("csv", "tsv")) {
    separator <- if (extension == "tsv") "\t" else ","
    table <- utils::read.table(
      path,
      header = TRUE,
      sep = separator,
      quote = "\"",
      comment.char = "",
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    if (ncol(table) < 1L) stop("Reference gene file has no columns: ", path)
    preferred <- c("gene", "genes", "gene_name", "feature", "symbol")
    column_index <- match(preferred, tolower(colnames(table)), nomatch = 0L)
    column_index <- column_index[column_index > 0L]
    genes <- table[[if (length(column_index)) column_index[[1L]] else 1L]]
  } else {
    genes <- readLines(path, warn = FALSE)
  }

  genes <- unique(trimws(as.character(genes)))
  genes[nzchar(genes) & !is.na(genes)]
}

get_reference_genes <- function(reference_genes_file = NULL,
                                reference_rds,
                                reference_assay = "RNA") {
  if (!is.null(reference_genes_file) && nzchar(reference_genes_file)) {
    message_now("Reading reference genes from ", reference_genes_file)
    return(read_reference_genes_file(reference_genes_file))
  }

  reference_rds <- normalizePath(path.expand(reference_rds), mustWork = TRUE)
  message_now("Reading reference genes from ", reference_rds)
  reference <- readRDS(reference_rds)

  if (inherits(reference, "Seurat")) {
    if (!reference_assay %in% Seurat::Assays(reference)) {
      stop("Reference assay not found: ", reference_assay)
    }
    genes <- rownames(reference[[reference_assay]])
  } else if (inherits(reference, "Matrix") || is.matrix(reference)) {
    genes <- rownames(reference)
  } else {
    stop("reference_rds must contain a Seurat object or gene-by-cell matrix.")
  }

  genes <- unique(as.character(genes))
  genes[nzchar(genes) & !is.na(genes)]
}

get_gene_expression_matrix <- function(read10x_result) {
  if (inherits(read10x_result, "Matrix") || is.matrix(read10x_result)) {
    return(read10x_result)
  }
  if (!is.list(read10x_result)) {
    stop("Read10X_h5 returned an unsupported object.")
  }

  if ("Gene Expression" %in% names(read10x_result)) {
    return(read10x_result[["Gene Expression"]])
  }
  if (length(read10x_result) == 1L) return(read10x_result[[1L]])

  stop(
    "The H5 file contains multiple feature types but no 'Gene Expression' ",
    "matrix. Available names: ", paste(names(read10x_result), collapse = ", ")
  )
}

get_rna_counts <- function(object, assay = "RNA") {
  if (inherits(object, "Seurat")) {
    if (!assay %in% Seurat::Assays(object)) {
      stop("Reference assay not found: ", assay)
    }
    assay_object <- object[[assay]]
    assay_layers <- tryCatch(
      SeuratObject::Layers(assay_object),
      error = function(error) character()
    )
    if ("counts" %in% assay_layers) {
      return(SeuratObject::LayerData(assay_object, layer = "counts"))
    }
    return(Seurat::GetAssayData(object, assay = assay, slot = "counts"))
  }
  if (inherits(object, "Matrix") || is.matrix(object)) return(object)
  stop("Reference RDS must contain a Seurat object or gene-by-cell matrix.")
}

write_expression_matrix_csv <- function(counts,
                                        genes,
                                        cells,
                                        output_file,
                                        expression_scale = "log1p_cp10k",
                                        chunk_size = 500L,
                                        scale_factor = 10000) {
  expression_scale <- match.arg(expression_scale, c("log1p_cp10k", "raw"))
  chunk_size <- as.integer(chunk_size)
  if (!is.finite(chunk_size) || chunk_size < 1L) {
    stop("expression_chunk_size must be positive.")
  }
  if (length(setdiff(genes, rownames(counts)))) {
    stop("Selected genes are missing from an expression matrix.")
  }
  if (length(setdiff(cells, colnames(counts)))) {
    stop("Selected cells are missing from an expression matrix.")
  }

  # CP10K uses each cell's full biological-gene library, not only the selected
  # 2,000-gene block. Only the selected output block is converted to dense form.
  library_size <- Matrix::colSums(counts[, cells, drop = FALSE])
  if (any(!is.finite(library_size) | library_size <= 0)) {
    stop("Expression export received a zero/non-finite library-size cell.")
  }

  partial_file <- paste0(output_file, ".partial")
  if (file.exists(partial_file)) file.remove(partial_file)
  starts <- seq.int(1L, length(cells), by = chunk_size)

  for (chunk_index in seq_along(starts)) {
    first <- starts[[chunk_index]]
    last <- min(first + chunk_size - 1L, length(cells))
    indexes <- first:last
    if (chunk_index == 1L || chunk_index %% 50L == 0L || last == length(cells)) {
      message_now(
        "Writing ", basename(output_file), ": chunk ", chunk_index,
        "/", length(starts), " (cells ", first, "-", last, ")"
      )
    }

    block <- counts[genes, cells[indexes], drop = FALSE]
    if (expression_scale == "log1p_cp10k") {
      block <- log1p(t(t(block) / library_size[indexes]) * scale_factor)
    }
    dense <- t(as.matrix(block))
    output <- data.table::as.data.table(dense)
    data.table::setnames(output, genes)
    output[, cell_id := cells[indexes]]
    data.table::setcolorder(output, c("cell_id", genes))
    data.table::fwrite(
      output,
      partial_file,
      append = chunk_index > 1L,
      col.names = chunk_index == 1L
    )
    rm(block, dense, output)
    invisible(gc())
  }

  if (file.exists(output_file) && !file.remove(output_file)) {
    stop("Could not replace existing expression file: ", output_file)
  }
  if (!file.rename(partial_file, output_file)) {
    stop("Could not finalize expression file: ", output_file)
  }
  normalizePath(output_file, mustWork = TRUE)
}

read_target_coordinates <- function(path, target_cells) {
  path <- normalizePath(path.expand(path), mustWork = TRUE)
  header <- names(data.table::fread(path, nrows = 0L))
  if (!"cell_id" %in% header) stop("Coordinate file lacks cell_id: ", path)

  if (all(c("x_centroid", "y_centroid") %in% header)) {
    coordinates <- data.table::fread(
      path,
      select = c("cell_id", "x_centroid", "y_centroid"),
      colClasses = list(character = "cell_id")
    )
    data.table::setnames(
      coordinates,
      c("x_centroid", "y_centroid"),
      c("x", "y")
    )
  } else if (all(c("x", "y") %in% header)) {
    coordinates <- data.table::fread(
      path,
      select = c("cell_id", "x", "y"),
      colClasses = list(character = "cell_id")
    )
  } else {
    stop("Coordinate file must contain cell_id plus x/y or x_centroid/y_centroid.")
  }

  if (anyDuplicated(coordinates$cell_id)) {
    stop("Target coordinate file contains duplicate cell IDs.")
  }
  coordinate_index <- match(target_cells, coordinates$cell_id)
  if (anyNA(coordinate_index)) {
    stop("Coordinates are missing for ", sum(is.na(coordinate_index)), " target cells.")
  }
  coordinates <- coordinates[coordinate_index]
  if (!identical(as.character(coordinates$cell_id), target_cells)) {
    stop("Failed to align target coordinates to target expression order.")
  }
  if (any(!is.finite(coordinates$x)) || any(!is.finite(coordinates$y))) {
    stop("Target coordinates contain non-finite values.")
  }
  coordinates
}

run_seurat_vst_selection <- function(settings = RSTUDIO_SETTINGS) {
  target_h5 <- normalizePath(path.expand(settings$target_h5), mustWork = TRUE)
  output_dir <- path.expand(settings$output_dir)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  target_cells_file <- settings$target_cells_file
  if (is.null(target_cells_file) || !nzchar(target_cells_file)) {
    target_cells_file <- file.path(dirname(target_h5), "cells.csv.gz")
  }
  target_cells_file <- path.expand(target_cells_file)
  write_synthetic_inputs <- isTRUE(settings$write_synthetic_inputs)
  expression_scale <- match.arg(
    settings$expression_scale,
    c("log1p_cp10k", "raw")
  )
  expression_chunk_size <- as.integer(settings$expression_chunk_size)

  nfeatures <- as.integer(settings$nfeatures)
  loess_span <- as.numeric(settings$loess_span)
  if (!is.finite(nfeatures) || nfeatures < 1L) stop("nfeatures must be positive.")
  if (!is.finite(loess_span) || loess_span <= 0 || loess_span > 1) {
    stop("loess_span must be in (0, 1].")
  }
  if (!is.finite(expression_chunk_size) || expression_chunk_size < 1L) {
    stop("expression_chunk_size must be positive.")
  }
  if (write_synthetic_inputs) {
    if (!file.exists(target_cells_file)) {
      stop("Target cell-coordinate file does not exist: ", target_cells_file)
    }
    if (is.null(settings$reference_rds) || !nzchar(settings$reference_rds)) {
      stop("reference_rds is required to export reference expression and labels.")
    }
    reference_rds_path <- path.expand(settings$reference_rds)
    if (!file.exists(reference_rds_path)) {
      stop("Reference RDS does not exist: ", reference_rds_path)
    }
  }

  message_now("Reading Xenium raw counts from ", target_h5)
  target_counts <- get_gene_expression_matrix(
    Seurat::Read10X_h5(
      filename = target_h5,
      use.names = TRUE,
      unique.features = TRUE
    )
  )
  target_counts <- methods::as(target_counts, "dgCMatrix")

  if (is.null(rownames(target_counts)) || is.null(colnames(target_counts))) {
    stop("Target count matrix must have gene and cell names.")
  }
  if (anyDuplicated(rownames(target_counts))) {
    stop("Target gene names are not unique after Read10X_h5.")
  }
  target_gene_count <- nrow(target_counts)
  target_cell_count <- ncol(target_counts)

  # Determine positive-library cells from the
  # complete Xenium raw-count matrix before restricting it to shared genes.
  target_library_size <- Matrix::colSums(target_counts)
  keep_cells <- is.finite(target_library_size) & target_library_size > 0
  if (sum(keep_cells) < 2L) stop("Fewer than two positive-library target cells.")
  target_counts <- target_counts[, keep_cells, drop = FALSE]
  target_cells <- colnames(target_counts)

  reference_genes <- get_reference_genes(
    reference_genes_file = settings$reference_genes_file,
    reference_rds = settings$reference_rds,
    reference_assay = settings$reference_assay
  )

  # Preserve Xenium feature order when constructing the shared-gene matrix.
  shared_genes <- rownames(target_counts)[rownames(target_counts) %in% reference_genes]
  if (length(shared_genes) < nfeatures) {
    stop(
      "Only ", length(shared_genes), " target/reference shared genes are ",
      "available; cannot select ", nfeatures, " genes."
    )
  }
  shared_counts <- target_counts[shared_genes, , drop = FALSE]

  message_now(
    "Creating Seurat object with ", format(nrow(shared_counts), big.mark = ","),
    " shared genes and ", format(ncol(shared_counts), big.mark = ","), " cells"
  )
  target_object <- Seurat::CreateSeuratObject(
    counts = shared_counts,
    assay = "RNA",
    project = settings$sample_name,
    min.cells = 0,
    min.features = 0
  )

  set.seed(as.integer(settings$random_seed))
  message_now("Running Seurat::FindVariableFeatures exactly")

  # This is the requested literal Seurat feature-selection call.
  target_object <- Seurat::FindVariableFeatures(
    object = target_object,
    assay = "RNA",
    selection.method = "vst",
    nfeatures = nfeatures,
    loess.span = loess_span,
    verbose = isTRUE(settings$verbose)
  )

  selected_genes <- Seurat::VariableFeatures(target_object, assay = "RNA")
  vst_statistics <- Seurat::HVFInfo(
    target_object,
    assay = "RNA",
    method = "vst",
    status = TRUE
  )
  vst_statistics$gene <- rownames(vst_statistics)
  vst_statistics$selected_rank <- match(vst_statistics$gene, selected_genes)
  vst_statistics <- vst_statistics[
    match(shared_genes, vst_statistics$gene),
    c("gene", setdiff(colnames(vst_statistics), "gene")),
    drop = FALSE
  ]
  rownames(vst_statistics) <- NULL

  selected_statistics <- vst_statistics[
    match(selected_genes, vst_statistics$gene), , drop = FALSE
  ]
  selected_statistics <- selected_statistics[
    order(selected_statistics$selected_rank), , drop = FALSE
  ]

  checks <- c(
    selected_count_is_exact = length(selected_genes) == nfeatures,
    selected_genes_are_unique = !anyDuplicated(selected_genes),
    selected_genes_are_shared = all(selected_genes %in% shared_genes),
    selected_order_matches_rank = identical(
      selected_genes,
      selected_statistics$gene
    )
  )
  if (!all(checks)) {
    stop(
      "Post-selection validation failed: ",
      paste(names(checks)[!checks], collapse = ", ")
    )
  }

  top_file <- file.path(
    output_dir,
    sprintf("top%d_shared_genes_seurat_vst.csv", nfeatures)
  )
  utils::write.csv(selected_statistics, top_file, row.names = FALSE, quote = FALSE)
  utils::write.csv(
    data.frame(gene = selected_genes, stringsAsFactors = FALSE),
    file.path(output_dir, sprintf("top%d_gene_names.csv", nfeatures)),
    row.names = FALSE,
    quote = FALSE
  )
  utils::write.csv(
    vst_statistics,
    file.path(output_dir, "all_shared_genes_seurat_vst_statistics.csv"),
    row.names = FALSE,
    quote = FALSE
  )
  writeLines(reference_genes, file.path(output_dir, "reference_genes_used.txt"))
  utils::write.csv(
    data.frame(gene = reference_genes, stringsAsFactors = FALSE),
    file.path(output_dir, "reference_genes_used.csv"),
    row.names = FALSE,
    quote = FALSE
  )

  # The Seurat object is no longer needed after extracting the exact VST
  # ranking. Releasing it before loading the reference reduces peak memory.
  rm(target_object, shared_counts)
  invisible(gc())

  input_bundle_written <- FALSE
  target_expression_file <- NA_character_
  reference_expression_file <- NA_character_
  target_coordinate_file <- NA_character_
  reference_label_file <- NA_character_
  reference_cells_after_filter <- NA_integer_
  reference_labels_after_filter <- NA_integer_

  if (write_synthetic_inputs) {
    message_now("Preparing target coordinates")
    coordinates <- read_target_coordinates(target_cells_file, target_cells)

    message_now("Loading reference expression and labels from ", reference_rds_path)
    reference_object <- readRDS(reference_rds_path)
    if (!inherits(reference_object, "Seurat")) {
      stop("A Seurat reference object is required to export reference labels.")
    }
    reference_metadata <- reference_object[[]]
    if (!settings$reference_label_field %in% colnames(reference_metadata)) {
      stop("Reference label field not found: ", settings$reference_label_field)
    }
    reference_counts <- get_rna_counts(
      reference_object,
      assay = settings$reference_assay
    )
    if (!inherits(reference_counts, "dgCMatrix")) {
      reference_counts <- methods::as(reference_counts, "dgCMatrix")
    }
    if (length(setdiff(selected_genes, rownames(reference_counts)))) {
      stop("Some selected genes are absent from the reference count matrix.")
    }

    reference_cells_all <- colnames(reference_counts)
    metadata_index <- match(reference_cells_all, rownames(reference_metadata))
    if (anyNA(metadata_index)) {
      stop("Reference metadata is missing ", sum(is.na(metadata_index)), " expression cells.")
    }
    reference_labels_all <- as.character(
      reference_metadata[[settings$reference_label_field]][metadata_index]
    )
    keep_reference <- !is.na(reference_labels_all) &
      nzchar(reference_labels_all) &
      !(reference_labels_all %in% settings$excluded_reference_labels)
    reference_cells <- reference_cells_all[keep_reference]
    reference_labels <- reference_labels_all[keep_reference]
    if (!length(reference_cells)) stop("No reference cells remain after label filtering.")

    reference_library_size <- Matrix::colSums(
      reference_counts[, reference_cells, drop = FALSE]
    )
    positive_reference <- is.finite(reference_library_size) &
      reference_library_size > 0
    reference_cells <- reference_cells[positive_reference]
    reference_labels <- reference_labels[positive_reference]
    if (!length(reference_cells)) {
      stop("No positive-library reference cells remain after filtering.")
    }
    reference_cells_after_filter <- length(reference_cells)
    reference_labels_after_filter <- length(unique(reference_labels))
    message_now(
      "Filtered reference: ", reference_cells_after_filter,
      " cells across ", reference_labels_after_filter, " labels"
    )

    scale_stem <- if (expression_scale == "log1p_cp10k") {
      "expression_log1p_cp10k"
    } else {
      "counts_raw"
    }
    target_expression_file <- file.path(
      output_dir,
      paste0("target_", scale_stem, ".csv")
    )
    reference_expression_file <- file.path(
      output_dir,
      paste0("reference_", scale_stem, ".csv")
    )
    target_coordinate_file <- file.path(output_dir, "target_coordinates.csv")
    reference_label_file <- file.path(output_dir, "reference_labels.csv")

    data.table::fwrite(
      coordinates[, .(cell_id, x, y)],
      target_coordinate_file
    )
    data.table::fwrite(
      data.table::data.table(
        cell_id = reference_cells,
        label = reference_labels
      ),
      reference_label_file
    )
    reference_label_counts <- data.table::data.table(label = reference_labels)[
      , .(N = .N), by = label
    ][order(-N, label)]
    data.table::fwrite(
      reference_label_counts,
      file.path(output_dir, "reference_label_counts.csv")
    )
    excluded_label_counts <- data.table::data.table(
      label = settings$excluded_reference_labels
    )
    observed_excluded <- data.table::data.table(label = reference_labels_all)[
      label %in% settings$excluded_reference_labels,
      .(N = .N),
      by = label
    ]
    excluded_label_counts <- merge(
      excluded_label_counts,
      observed_excluded,
      by = "label",
      all.x = TRUE,
      sort = FALSE
    )
    excluded_label_counts[is.na(N), N := 0L]
    data.table::fwrite(
      excluded_label_counts,
      file.path(output_dir, "reference_excluded_label_counts.csv")
    )

    message_now("Writing selected-gene target expression CSV")
    write_expression_matrix_csv(
      counts = target_counts,
      genes = selected_genes,
      cells = target_cells,
      output_file = target_expression_file,
      expression_scale = expression_scale,
      chunk_size = expression_chunk_size
    )
    message_now("Writing selected-gene filtered-reference expression CSV")
    write_expression_matrix_csv(
      counts = reference_counts,
      genes = selected_genes,
      cells = reference_cells,
      output_file = reference_expression_file,
      expression_scale = expression_scale,
      chunk_size = expression_chunk_size
    )

    # Fast structural verification without loading the large expression files.
    expected_expression_header <- c("cell_id", selected_genes)
    target_header <- names(data.table::fread(target_expression_file, nrows = 0L))
    reference_header <- names(
      data.table::fread(reference_expression_file, nrows = 0L)
    )
    if (!identical(target_header, expected_expression_header)) {
      stop("Target expression CSV gene header/order validation failed.")
    }
    if (!identical(reference_header, expected_expression_header)) {
      stop("Reference expression CSV gene header/order validation failed.")
    }
    coordinate_check <- data.table::fread(
      target_coordinate_file,
      select = "cell_id",
      colClasses = "character"
    )$cell_id
    label_check <- data.table::fread(
      reference_label_file,
      select = "cell_id",
      colClasses = "character"
    )$cell_id
    if (!identical(coordinate_check, target_cells)) {
      stop("Target coordinate cell order validation failed.")
    }
    if (!identical(label_check, reference_cells)) {
      stop("Reference label cell order validation failed.")
    }

    input_bundle_written <- TRUE
    input_bundle <- data.frame(
      role = c(
        "selected_genes",
        "target_expression",
        "target_coordinates",
        "reference_expression",
        "reference_labels"
      ),
      file = c(
        top_file,
        target_expression_file,
        target_coordinate_file,
        reference_expression_file,
        reference_label_file
      ),
      rows = c(
        nfeatures,
        length(target_cells),
        length(target_cells),
        reference_cells_after_filter,
        reference_cells_after_filter
      ),
      feature_columns = c(nfeatures, nfeatures, 2L, nfeatures, 1L),
      stringsAsFactors = FALSE
    )
    utils::write.csv(
      input_bundle,
      file.path(output_dir, "synthetic_input_files.csv"),
      row.names = FALSE,
      quote = FALSE
    )

    rm(
      coordinates,
      coordinate_check,
      excluded_label_counts,
      input_bundle,
      label_check,
      observed_excluded,
      reference_counts,
      reference_label_counts,
      reference_metadata,
      reference_object
    )
    invisible(gc())
  }

  summary_table <- data.frame(
    sample_name = settings$sample_name,
    target_h5 = target_h5,
    seurat_version = as.character(utils::packageVersion("Seurat")),
    selection_method = "vst",
    nfeatures_requested = nfeatures,
    loess_span = loess_span,
    target_genes = target_gene_count,
    target_cells = target_cell_count,
    positive_library_cells = ncol(target_counts),
    reference_genes = length(reference_genes),
    shared_genes = length(shared_genes),
    selected_genes = length(selected_genes),
    all_checks_passed = all(checks),
    synthetic_input_bundle_written = input_bundle_written,
    expression_scale = expression_scale,
    expression_chunk_size = expression_chunk_size,
    target_expression_file = target_expression_file,
    target_coordinate_file = target_coordinate_file,
    reference_expression_file = reference_expression_file,
    reference_label_file = reference_label_file,
    reference_cells_after_filter = reference_cells_after_filter,
    reference_labels_after_filter = reference_labels_after_filter,
    stringsAsFactors = FALSE
  )
  utils::write.csv(
    summary_table,
    file.path(output_dir, "vst_selection_summary.csv"),
    row.names = FALSE,
    quote = FALSE
  )
  writeLines(
    capture.output(sessionInfo()),
    file.path(output_dir, "sessionInfo.txt")
  )

  message_now("Finished. Selected genes: ", length(selected_genes))
  message_now("Primary output: ", top_file)
  if (input_bundle_written) {
    message_now("Synthetic-generation CSV input bundle written to ", output_dir)
  }

  invisible(list(
    selected_genes = selected_genes,
    selected_statistics = selected_statistics,
    all_statistics = vst_statistics,
    summary = summary_table,
    output_dir = normalizePath(output_dir, mustWork = TRUE)
  ))
}

if (isTRUE(RUN_R_STUDIO_WORKFLOW)) {
  rstudio_result <- run_seurat_vst_selection(RSTUDIO_SETTINGS)
} else if (interactive()) {
  message(
    "Script loaded without running. Edit RSTUDIO_SETTINGS, set ",
    "RUN_R_STUDIO_WORKFLOW <- TRUE, and click Source; or run:\n",
    "rstudio_result <- run_seurat_vst_selection()"
  )
}
