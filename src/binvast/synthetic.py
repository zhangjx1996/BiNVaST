#!/usr/bin/env python3
"""Generate PCA nearest-cell synthetic data from R-prepared CSV inputs.

Expected files in --input-data-dir (created by
select_shared_vst_seurat.R):

  target_expression_log1p_cp10k.csv
  target_coordinates.csv
  reference_expression_log1p_cp10k.csv
  reference_labels.csv

The target and reference matrices must contain cells in rows, genes in columns,
and a first ``cell_id`` column.  The output directory contains only four CSV
products used by the subsequent BinVAST training job:

  synthetic_expression.csv
  synthetic_labels.csv
  synthetic_spatial_coords.csv
  synthetic_cell_type_counts.csv

Strategy: global dropout-only mode/maximum alignment of reference to target;
PCA after target-derived gene scaling; per-cell-type distance to the nearest
reference cell; Gaussian-kernel label sampling; and cell-type-specific
mode/maximum plus dropout-only alignment of the selected nearest cell. PCA is
fit on the aligned reference by default. With ``--pca-base target``, PCA is fit
on the unchanged target and the aligned reference is projected into target PCA
space. No zero is backfilled when the reference is sparser than the target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


EPS = 1e-8
BIN_WIDTH = 0.05
MAX_BIN_VALUE = 12.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--seed", type=int, default=259)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument(
        "--pca-base",
        choices=("reference", "target"),
        default="reference",
        help=(
            "Dataset used to fit PCA. Default: reference. With target, fit on "
            "the unchanged target and project the globally aligned reference."
        ),
    )
    parser.add_argument("--csv-chunk-size", type=int, default=2000)
    parser.add_argument("--assignment-batch-size", type=int, default=10000)
    parser.add_argument("--keep-work", action="store_true")
    return parser.parse_args()


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle) - 1
    if rows < 1:
        raise ValueError(f"Expression CSV has no data rows: {path}")
    return rows


def csv_to_memmap(path: Path, destination: Path, chunk_size: int):
    header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    if len(header) < 2:
        raise ValueError(f"Expected cell_id plus genes in {path}")
    id_column = header[0]
    genes = np.asarray(header[1:], dtype=object)
    n_rows = count_csv_rows(path)
    matrix = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(n_rows, len(genes))
    )
    ids: list[str] = []
    offset = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size, dtype={id_column: str}):
        chunk_ids = chunk.iloc[:, 0].astype(str).tolist()
        values = chunk.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
        if values.shape[1] != len(genes):
            raise ValueError(f"Inconsistent column count while reading {path}")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Non-finite or negative expression values in {path}")
        stop = offset + len(values)
        matrix[offset:stop] = values
        ids.extend(chunk_ids)
        offset = stop
    if offset != n_rows:
        raise ValueError(f"Expected {n_rows} rows but read {offset} from {path}")
    matrix.flush()
    del matrix
    ids_array = np.asarray(ids, dtype=object)
    if len(pd.Index(ids_array).unique()) != len(ids_array):
        raise ValueError(f"Duplicate cell IDs in {path}")
    return np.load(destination, mmap_mode="r"), ids_array, genes


def load_aligned_labels(path: Path, cell_ids: np.ndarray) -> np.ndarray:
    table = pd.read_csv(path, dtype=str)
    if not {"cell_id", "label"}.issubset(table.columns):
        raise ValueError(f"{path} must contain cell_id and label columns")
    series = table.set_index("cell_id")["label"]
    missing = pd.Index(cell_ids).difference(series.index)
    if len(missing):
        raise ValueError(f"{len(missing)} reference cells are missing labels")
    labels = series.loc[cell_ids].astype(str).to_numpy()
    if np.any(pd.isna(labels)) or np.any(labels == ""):
        raise ValueError("Reference labels contain missing/empty values")
    return labels


def load_aligned_coordinates(path: Path, cell_ids: np.ndarray) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"cell_id": str})
    if not {"cell_id", "x", "y"}.issubset(table.columns):
        raise ValueError(f"{path} must contain cell_id, x, and y columns")
    table = table.set_index("cell_id")
    missing = pd.Index(cell_ids).difference(table.index)
    if len(missing):
        raise ValueError(f"{len(missing)} target cells are missing coordinates")
    result = table.loc[cell_ids, ["x", "y"]].astype(float).reset_index()
    if not np.isfinite(result[["x", "y"]].to_numpy()).all():
        raise ValueError("Target coordinates contain non-finite values")
    return result


def compute_group_stats(
    matrix: np.ndarray,
    labels: np.ndarray,
    levels: np.ndarray,
    chunk_size: int,
):
    n_levels = len(levels)
    n_genes = matrix.shape[1]
    n_bins = int(math.ceil(MAX_BIN_VALUE / BIN_WIDTH)) + 1
    n_total = np.zeros(n_levels, dtype=np.int64)
    n_positive = np.zeros((n_levels, n_genes), dtype=np.int64)
    maximum = np.zeros((n_levels, n_genes), dtype=np.float32)
    mode = np.zeros((n_levels, n_genes), dtype=np.float32)

    for level_index, level in enumerate(levels):
        rows = np.flatnonzero(labels == level)
        n_total[level_index] = len(rows)
        if not len(rows):
            continue
        histogram = np.zeros((n_bins, n_genes), dtype=np.uint32)
        print(f"Statistics: {level} ({len(rows):,} cells)", flush=True)
        for start in range(0, len(rows), chunk_size):
            block = np.asarray(matrix[rows[start : start + chunk_size]], dtype=np.float32)
            positive = block > EPS
            n_positive[level_index] += positive.sum(axis=0)
            maximum[level_index] = np.maximum(
                maximum[level_index], np.where(positive, block, 0).max(axis=0)
            )
            row_index, column_index = np.nonzero(positive)
            if len(row_index):
                bins = np.floor(
                    np.minimum(block[row_index, column_index], MAX_BIN_VALUE) / BIN_WIDTH
                ).astype(np.int32)
                np.add.at(histogram, (bins, column_index), 1)
        mode[level_index] = (histogram.argmax(axis=0) + 0.5) * BIN_WIDTH
        mode[level_index, n_positive[level_index] == 0] = 0

    zero_rate = 1.0 - n_positive / np.maximum(n_total[:, None], 1)
    return {
        "n_total": n_total,
        "n_positive": n_positive,
        "zero_rate": zero_rate.astype(np.float32),
        "mode": mode,
        "max": maximum,
    }


def scale_mode_max(values, source_mode, source_max, target_mode, target_max):
    values = np.asarray(values, dtype=np.float32).copy()
    positive = values > EPS
    denominator = source_max - source_mode
    numerator = target_max - target_mode
    valid = (denominator > EPS) & (numerator > EPS) & (target_max > EPS)
    if np.any(valid):
        values[:, valid] = target_mode[valid] + (
            values[:, valid] - source_mode[valid]
        ) * (numerator[valid] / denominator[valid])
    if np.any(~valid):
        fallback = np.where(
            target_mode[~valid] > EPS, target_mode[~valid], target_max[~valid]
        )
        values[:, ~valid] = np.where(positive[:, ~valid], fallback, 0)
    values = np.where(positive, values, 0)
    return np.minimum(np.maximum(values, 0), np.maximum(target_max, 0)[None, :])


def dropout_only(values, target_zero, source_zero, rng):
    probability = np.maximum(target_zero - source_zero, 0) / np.maximum(
        1 - source_zero, EPS
    )
    probability[~np.isfinite(probability)] = 0
    if np.any(probability > 0):
        drop = rng.random(values.shape, dtype=np.float32) < probability[None, :]
        values[(values > EPS) & drop] = 0
    return values


def target_mean_sd(matrix: np.ndarray, chunk_size: int):
    sums = np.zeros(matrix.shape[1], dtype=np.float64)
    sums_squared = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, matrix.shape[0], chunk_size):
        block = np.asarray(matrix[start : start + chunk_size], dtype=np.float64)
        sums += block.sum(axis=0)
        sums_squared += np.square(block).sum(axis=0)
    mean = sums / matrix.shape[0]
    variance = np.maximum(
        (sums_squared - matrix.shape[0] * np.square(mean))
        / max(matrix.shape[0] - 1, 1),
        EPS,
    )
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def align_reference_globally(target, reference, destination, chunk_size, rng):
    all_target = np.repeat("all", target.shape[0])
    all_reference = np.repeat("all", reference.shape[0])
    target_stats = compute_group_stats(
        target, all_target, np.array(["all"]), chunk_size
    )
    reference_stats = compute_group_stats(
        reference, all_reference, np.array(["all"]), chunk_size
    )
    output = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=reference.shape
    )
    for start in range(0, reference.shape[0], chunk_size):
        stop = min(start + chunk_size, reference.shape[0])
        values = scale_mode_max(
            reference[start:stop],
            reference_stats["mode"][0],
            reference_stats["max"][0],
            target_stats["mode"][0],
            target_stats["max"][0],
        )
        output[start:stop] = dropout_only(
            values,
            target_stats["zero_rate"][0],
            reference_stats["zero_rate"][0],
            rng,
        )
    output.flush()
    del output
    return np.load(destination, mmap_mode="r")


def fit_reference_pca(target, aligned_reference, work_dir, n_pcs, chunk_size, seed):
    n_components = min(n_pcs, aligned_reference.shape[0] - 1, target.shape[1])
    if n_components < 1:
        raise ValueError("Not enough cells/genes to fit PCA")
    target_mean, target_sd = target_mean_sd(target, chunk_size)
    reference_scaled = np.asarray(aligned_reference, dtype=np.float32).copy()
    reference_scaled -= target_mean
    reference_scaled /= target_sd
    model = PCA(
        n_components=n_components,
        svd_solver="randomized",
        iterated_power=4,
        random_state=seed,
    )
    print(f"Fitting reference PCA with {n_components} components", flush=True)
    reference_pcs = model.fit_transform(reference_scaled).astype(np.float32)
    del reference_scaled
    target_pc_path = work_dir / "target_pcs.npy"
    target_pcs = np.lib.format.open_memmap(
        target_pc_path,
        mode="w+",
        dtype=np.float32,
        shape=(target.shape[0], n_components),
    )
    for start in range(0, target.shape[0], chunk_size):
        stop = min(start + chunk_size, target.shape[0])
        block = np.asarray(target[start:stop], dtype=np.float32)
        block = (block - target_mean) / target_sd
        target_pcs[start:stop] = model.transform(block).astype(np.float32)
    target_pcs.flush()
    del target_pcs
    return reference_pcs, np.load(target_pc_path, mmap_mode="r")


def fit_target_pca(target, aligned_reference, work_dir, n_pcs, chunk_size, seed):
    """Fit PCA on unchanged target data and project the aligned reference."""
    n_components = min(n_pcs, target.shape[0] - 1, target.shape[1])
    if n_components < 1:
        raise ValueError("Not enough cells/genes to fit PCA")
    target_mean, target_sd = target_mean_sd(target, chunk_size)
    target_scaled = np.asarray(target, dtype=np.float32).copy()
    target_scaled -= target_mean
    target_scaled /= target_sd
    model = PCA(
        n_components=n_components,
        svd_solver="randomized",
        iterated_power=4,
        random_state=seed,
    )
    print(f"Fitting target PCA with {n_components} components", flush=True)
    target_pcs = model.fit_transform(target_scaled).astype(np.float32)
    del target_scaled

    reference_pc_path = work_dir / "reference_pcs.npy"
    reference_pcs = np.lib.format.open_memmap(
        reference_pc_path,
        mode="w+",
        dtype=np.float32,
        shape=(aligned_reference.shape[0], n_components),
    )
    for start in range(0, aligned_reference.shape[0], chunk_size):
        stop = min(start + chunk_size, aligned_reference.shape[0])
        block = np.asarray(aligned_reference[start:stop], dtype=np.float32)
        block = (block - target_mean) / target_sd
        reference_pcs[start:stop] = model.transform(block).astype(np.float32)
    reference_pcs.flush()
    del reference_pcs
    return np.load(reference_pc_path, mmap_mode="r"), target_pcs


def stable_softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def sample_rows(probabilities, rng):
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    draws = rng.random(probabilities.shape[0])
    return np.sum(draws[:, None] > cumulative, axis=1)


def assign_labels_nearest_cell(
    target_pcs,
    reference_pcs,
    reference_labels,
    tau,
    batch_size,
    rng,
):
    levels = np.asarray(sorted(pd.unique(reference_labels)), dtype=object)
    models = []
    reference_rows = []
    for level in levels:
        rows = np.flatnonzero(reference_labels == level)
        reference_rows.append(rows)
        models.append(NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(reference_pcs[rows]))

    assigned = np.empty(target_pcs.shape[0], dtype=object)
    selected_reference_rows = np.empty(target_pcs.shape[0], dtype=np.int64)
    for start in range(0, target_pcs.shape[0], batch_size):
        stop = min(start + batch_size, target_pcs.shape[0])
        print(f"Nearest-cell Gaussian assignment: rows {start + 1:,}-{stop:,}", flush=True)
        block = np.asarray(target_pcs[start:stop], dtype=np.float32)
        distances = np.empty((len(block), len(levels)), dtype=np.float64)
        nearest = np.empty((len(block), len(levels)), dtype=np.int64)
        for level_index, model in enumerate(models):
            distance, local_index = model.kneighbors(block, return_distance=True)
            distances[:, level_index] = distance[:, 0]
            nearest[:, level_index] = reference_rows[level_index][local_index[:, 0]]
        probabilities = stable_softmax(-np.square(distances) / (2.0 * tau**2))
        chosen = sample_rows(probabilities, rng)
        local_rows = np.arange(len(block))
        assigned[start:stop] = levels[chosen]
        selected_reference_rows[start:stop] = nearest[local_rows, chosen]
    return assigned, selected_reference_rows, levels


def write_expression_csv(
    destination,
    target,
    reference,
    target_ids,
    genes,
    assigned,
    selected_reference_rows,
    reference_labels,
    levels,
    chunk_size,
    rng,
):
    print("Computing cell-type-specific alignment statistics", flush=True)
    target_stats = compute_group_stats(target, assigned, levels, chunk_size)
    reference_stats = compute_group_stats(
        reference, reference_labels, levels, chunk_size
    )
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    level_to_index = {level: index for index, level in enumerate(levels)}
    first = True
    for start in range(0, len(target_ids), chunk_size):
        stop = min(start + chunk_size, len(target_ids))
        print(f"Writing synthetic expression: rows {start + 1:,}-{stop:,}", flush=True)
        chunk_labels = assigned[start:stop]
        block = np.zeros((stop - start, len(genes)), dtype=np.float32)
        for level in pd.unique(chunk_labels):
            level_index = level_to_index[level]
            local_rows = np.flatnonzero(chunk_labels == level)
            source_rows = selected_reference_rows[start:stop][local_rows]
            values = scale_mode_max(
                reference[source_rows],
                reference_stats["mode"][level_index],
                reference_stats["max"][level_index],
                target_stats["mode"][level_index],
                target_stats["max"][level_index],
            )
            block[local_rows] = dropout_only(
                values,
                target_stats["zero_rate"][level_index],
                reference_stats["zero_rate"][level_index],
                rng,
            )
        frame = pd.DataFrame(block, columns=genes)
        frame.insert(0, "cell_id", target_ids[start:stop])
        frame.to_csv(
            partial,
            mode="w" if first else "a",
            index=False,
            header=first,
            float_format="%.7g",
        )
        first = False
    os.replace(partial, destination)


def validate_outputs(output_dir, target_ids, genes, levels, chunk_size):
    expression_path = output_dir / "synthetic_expression.csv"
    labels = pd.read_csv(output_dir / "synthetic_labels.csv", dtype={"cell_id": str})
    coords = pd.read_csv(
        output_dir / "synthetic_spatial_coords.csv", dtype={"cell_id": str}
    )
    counts = pd.read_csv(output_dir / "synthetic_cell_type_counts.csv")
    header = pd.read_csv(expression_path, nrows=0).columns.astype(str).tolist()
    if header != ["cell_id", *genes.tolist()]:
        raise AssertionError("Synthetic expression gene header/order changed")
    if not labels["cell_id"].to_numpy().tolist() == target_ids.tolist():
        raise AssertionError("Synthetic label IDs do not match target order")
    if not coords["cell_id"].to_numpy().tolist() == target_ids.tolist():
        raise AssertionError("Synthetic coordinate IDs do not match target order")
    if set(labels["label"]) - set(levels):
        raise AssertionError("Synthetic output contains unknown labels")
    if int(counts["N"].sum()) != len(target_ids):
        raise AssertionError("Synthetic cell-type counts do not sum to target cells")

    observed_ids = []
    rows = 0
    minimum = np.inf
    maximum = -np.inf
    zero_count = 0
    total_values = 0
    for chunk in pd.read_csv(expression_path, chunksize=chunk_size):
        observed_ids.extend(chunk.iloc[:, 0].astype(str).tolist())
        values = chunk.iloc[:, 1:].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise AssertionError("Synthetic expression contains non-finite values")
        if len(values):
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
        zero_count += int(np.count_nonzero(values == 0))
        total_values += values.size
        rows += len(values)
    if observed_ids != target_ids.tolist() or rows != len(target_ids):
        raise AssertionError("Synthetic expression row IDs/count do not match target")
    if minimum < 0:
        raise AssertionError("Synthetic expression contains negative values")
    return {
        "rows": rows,
        "genes": len(genes),
        "cell_types": len(levels),
        "minimum": minimum,
        "maximum": maximum,
        "zero_fraction": zero_count / max(total_values, 1),
        "verified": True,
    }


def generate_synthetic_data(
    input_data_dir: str | Path,
    output_dir: str | Path,
    *,
    work_dir: str | Path | None = None,
    seed: int = 259,
    tau: float = 1.0,
    n_pcs: int = 50,
    pca_base: str = "reference",
    csv_chunk_size: int = 2000,
    assignment_batch_size: int = 10000,
    keep_work: bool = False,
) -> dict:
    """Generate query-matched synthetic cells from the R-prepared input bundle.

    Parameters mirror the ``binvast-simulate`` command. The returned dictionary
    contains the output paths and verification summary.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    if n_pcs < 1 or csv_chunk_size < 1 or assignment_batch_size < 1:
        raise ValueError("PCA components and batch sizes must be positive")
    if pca_base not in {"reference", "target"}:
        raise ValueError("pca_base must be 'reference' or 'target'")

    input_dir = Path(input_data_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    required = {
        "target": input_dir / "target_expression_log1p_cp10k.csv",
        "coordinates": input_dir / "target_coordinates.csv",
        "reference": input_dir / "reference_expression_log1p_cp10k.csv",
        "reference_labels": input_dir / "reference_labels.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    auto_work = work_dir is None
    if auto_work:
        scratch_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        resolved_work_dir = Path(
            tempfile.mkdtemp(prefix="binvast_synthetic_", dir=scratch_root)
        )
    else:
        resolved_work_dir = Path(work_dir).expanduser().resolve()
        resolved_work_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"Input data directory: {input_dir}", flush=True)
    print(f"Synthetic output directory: {output_dir}", flush=True)
    print(f"Temporary work directory: {resolved_work_dir}", flush=True)
    print(f"PCA base: {pca_base}", flush=True)
    try:
        target, target_ids, target_genes = csv_to_memmap(
            required["target"], resolved_work_dir / "target.npy", csv_chunk_size
        )
        reference, reference_ids, reference_genes = csv_to_memmap(
            required["reference"], resolved_work_dir / "reference.npy", csv_chunk_size
        )
        if not np.array_equal(target_genes, reference_genes):
            raise ValueError("Target/reference gene names or order differ")
        reference_labels = load_aligned_labels(
            required["reference_labels"], reference_ids
        )
        coordinates = load_aligned_coordinates(required["coordinates"], target_ids)

        rng_alignment = np.random.default_rng(seed)
        aligned_reference = align_reference_globally(
            target,
            reference,
            resolved_work_dir / "reference_globally_aligned.npy",
            csv_chunk_size,
            rng_alignment,
        )
        if pca_base == "reference":
            reference_pcs, target_pcs = fit_reference_pca(
                target,
                aligned_reference,
                resolved_work_dir,
                n_pcs,
                csv_chunk_size,
                seed,
            )
            pca_direction = (
                "PCA fit on globally dropout-aligned reference; unchanged "
                "target projected"
            )
        else:
            reference_pcs, target_pcs = fit_target_pca(
                target,
                aligned_reference,
                resolved_work_dir,
                n_pcs,
                csv_chunk_size,
                seed,
            )
            pca_direction = (
                "PCA fit on unchanged target; globally dropout-aligned "
                "reference projected"
            )
        assigned, selected_reference_rows, levels = assign_labels_nearest_cell(
            target_pcs,
            reference_pcs,
            reference_labels,
            tau,
            assignment_batch_size,
            np.random.default_rng(seed + 10),
        )

        pd.DataFrame({"cell_id": target_ids, "label": assigned}).to_csv(
            output_dir / "synthetic_labels.csv", index=False
        )
        coordinates.to_csv(output_dir / "synthetic_spatial_coords.csv", index=False)
        label_counts = (
            pd.Series(assigned, name="label")
            .value_counts()
            .reindex(levels, fill_value=0)
            .rename_axis("label")
            .reset_index(name="N")
        )
        label_counts["percentage"] = 100.0 * label_counts["N"] / len(assigned)
        label_counts.to_csv(
            output_dir / "synthetic_cell_type_counts.csv", index=False
        )
        write_expression_csv(
            output_dir / "synthetic_expression.csv",
            target,
            reference,
            target_ids,
            target_genes,
            assigned,
            selected_reference_rows,
            reference_labels,
            levels,
            csv_chunk_size,
            np.random.default_rng(seed + 101),
        )
        verification = validate_outputs(
            output_dir, target_ids, target_genes, levels, csv_chunk_size
        )
        verification.update(
            {
                "strategy": (
                    f"{pca_base}_pca_nearest_cell_gaussian_"
                    "dropout_only_no_backfill"
                ),
                "pca_base": pca_base,
                "pca_direction": pca_direction,
                "tau": tau,
                "requested_pcs": n_pcs,
                "seed": seed,
                "elapsed_minutes": (time.time() - started) / 60.0,
            }
        )
        print("Synthetic data verification passed:", flush=True)
        print(json.dumps(verification, indent=2), flush=True)
        print("Output files:", flush=True)
        for name in (
            "synthetic_expression.csv",
            "synthetic_labels.csv",
            "synthetic_spatial_coords.csv",
            "synthetic_cell_type_counts.csv",
        ):
            print(f"  {output_dir / name}", flush=True)
    finally:
        if auto_work and not keep_work:
            shutil.rmtree(resolved_work_dir, ignore_errors=True)

    output_files = {
        name: output_dir / name
        for name in (
            "synthetic_expression.csv",
            "synthetic_labels.csv",
            "synthetic_spatial_coords.csv",
            "synthetic_cell_type_counts.csv",
        )
    }
    return {
        "output_dir": output_dir,
        "files": output_files,
        "verification": verification,
    }


def main() -> None:
    args = parse_args()
    generate_synthetic_data(
        input_data_dir=args.input_data_dir,
        output_dir=args.output_dir,
        work_dir=args.work_dir,
        seed=args.seed,
        tau=args.tau,
        n_pcs=args.n_pcs,
        pca_base=args.pca_base,
        csv_chunk_size=args.csv_chunk_size,
        assignment_batch_size=args.assignment_batch_size,
        keep_work=args.keep_work,
    )


if __name__ == "__main__":
    main()
