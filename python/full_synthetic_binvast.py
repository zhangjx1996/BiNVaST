#!/usr/bin/env python3
"""Platform-neutral BinVAST training, evaluation, and prediction helper.

Keep this file with binvast_model_core.py and pass the prepared target and
synthetic-data folders:

  preprocessed_input/
  synthetic_input/
  binvast_model_core.py

In Python:

  from full_synthetic_binvast import run_full_synthetic_binvast
  result = run_full_synthetic_binvast(
      project_root="/path/to/binvast_project",
      target_dir_name="preprocessed_input",
      synthetic_dir_name="synthetic_input",
  )
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from binvast_model_core import (
    CellDataset,
    LabelEncoder,
    TrainConfig,
    asdict,
    build_knn_graph,
    device,
    fit_gene_minmax_scaler,
    get_log_expression,
    evaluate_model,
    load_coordinate_data,
    load_label_data,
    matrix_to_dataframe,
    log1p_cpm,
    make_internal_split,
    make_model,
    multimodal_loss,
    normalize_coordinates,
    safe_probability_column,
    save_latent_csv,
    torch,
    train_one_model,
    transform_gene_minmax_to_minus1_1,
    load_trained_model,
)


DEFAULT_SEED = 259


def _read_sidecar(stem: str | Path, suffix: str, column: str) -> np.ndarray:
    return pd.read_csv(Path(str(stem) + suffix))[column].astype(str).to_numpy()


def matrix_sidecars(matrix_path: str | Path) -> tuple[np.ndarray, list[str]]:
    matrix_path = Path(matrix_path)
    stem = str(matrix_path)[:-4]
    cell_ids = _read_sidecar(stem, "_cell_ids.csv", "cell_id")
    genes = _read_sidecar(stem, "_genes.csv", "gene").tolist()
    return cell_ids, genes


def resolve_first_existing(directory: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"None of the supported files exist in {directory}: {', '.join(candidates)}"
    )


def expression_matrix_metadata(matrix_path: str | Path):
    """Return cell IDs, genes, and shape for an NPY or cells-by-genes CSV."""
    matrix_path = Path(matrix_path)
    if matrix_path.suffix.lower() == ".npy":
        array = np.load(matrix_path, mmap_mode="r")
        cell_ids, genes = matrix_sidecars(matrix_path)
        return cell_ids, genes, tuple(array.shape)
    if matrix_path.suffix.lower() == ".csv":
        header = pd.read_csv(matrix_path, nrows=0).columns.astype(str).tolist()
        if len(header) < 2:
            raise ValueError(f"Expected cell_id plus genes in {matrix_path}")
        id_column = header[0]
        cell_ids = pd.read_csv(
            matrix_path, usecols=[id_column], dtype={id_column: str}
        )[id_column].astype(str).to_numpy()
        return cell_ids, header[1:], (len(cell_ids), len(header) - 1)
    raise ValueError(f"Unsupported expression matrix format: {matrix_path}")


def iter_expression_batches(matrix_path: str | Path, batch_size: int):
    """Yield (start, stop, IDs, float32 values) without loading a CSV at once."""
    matrix_path = Path(matrix_path)
    if matrix_path.suffix.lower() == ".npy":
        array = np.load(matrix_path, mmap_mode="r")
        cell_ids, _ = matrix_sidecars(matrix_path)
        for start in range(0, array.shape[0], batch_size):
            stop = min(start + batch_size, array.shape[0])
            yield start, stop, cell_ids[start:stop], np.asarray(
                array[start:stop], dtype=np.float32
            )
        return
    if matrix_path.suffix.lower() == ".csv":
        offset = 0
        for chunk in pd.read_csv(matrix_path, chunksize=batch_size):
            ids = chunk.iloc[:, 0].astype(str).to_numpy()
            values = chunk.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
            stop = offset + len(chunk)
            yield offset, stop, ids, values
            offset = stop
        return
    raise ValueError(f"Unsupported expression matrix format: {matrix_path}")


def resolve_input_dir(project_root: Path, path_or_name: str | Path) -> Path:
    path_or_name = Path(path_or_name)
    if path_or_name.is_absolute():
        return path_or_name
    return project_root / path_or_name


def compute_spatial_features_and_graph(coords_norm: np.ndarray, k: int = 30, eps: float = 1e-8):
    k_eff = min(k + 1, coords_norm.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(coords_norm)
    distances, indices = nbrs.kneighbors(coords_norm)
    nn_dist = distances[:, 1:]
    neighbor_indices = indices[:, 1:].astype(np.int64, copy=False)
    mean_dist = nn_dist.mean(axis=1, keepdims=True)
    density = (1.0 / (nn_dist + eps)).mean(axis=1, keepdims=True)
    H = np.concatenate([coords_norm, density, mean_dist], axis=1)
    H_mean = H.mean(axis=0, keepdims=True)
    H_std = np.maximum(H.std(axis=0, keepdims=True), eps)
    H = ((H - H_mean) / H_std).astype(np.float32)

    n = coords_norm.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int64), neighbor_indices.shape[1])
    cols = neighbor_indices.reshape(-1)
    pos_edges = np.column_stack([rows, cols]).astype(np.int64, copy=False)
    adj_rows = np.concatenate([rows, cols, np.arange(n, dtype=np.int64)])
    adj_cols = np.concatenate([cols, rows, np.arange(n, dtype=np.int64)])
    degree = np.bincount(adj_rows, minlength=n).astype(np.float32)
    degree = np.maximum(degree, 1.0)
    values = (1.0 / np.sqrt(degree[adj_rows] * degree[adj_cols])).astype(np.float32)
    idx = torch.tensor(np.vstack([adj_rows, adj_cols]), dtype=torch.long)
    val = torch.tensor(values, dtype=torch.float32)
    adj_norm = torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()
    return H, adj_norm, pos_edges, neighbor_indices


def sample_negative_edges_fast(n_nodes: int, pos_edges: np.ndarray, n_samples: int, seed: int = DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    pos_set = set(map(tuple, pos_edges[: min(len(pos_edges), 2_000_000)]))
    neg_edges = []
    max_trials = max(10000, n_samples * 30)
    trials = 0
    while len(neg_edges) < n_samples and trials < max_trials:
        a = int(rng.integers(0, n_nodes))
        b = int(rng.integers(0, n_nodes))
        trials += 1
        if a == b or (a, b) in pos_set:
            continue
        neg_edges.append((a, b))
    return np.array(neg_edges, dtype=np.int64)


def load_full_synthetic_data(
    expression_matrix: str | Path,
    label_csv: str | Path,
    coord_csv: str | Path,
    cell_id_col: str = "cell_id",
    label_col: str = "label",
    x_col: str = "x",
    y_col: str = "y",
):
    expression_matrix = Path(expression_matrix)
    X_df = matrix_to_dataframe(expression_matrix)
    X_df.index = X_df.index.astype(str)
    cell_ids = X_df.index.to_numpy()

    labels = load_label_data(label_csv, cell_id_col=cell_id_col, label_col=label_col)
    coords = load_coordinate_data(coord_csv, cell_id_col=cell_id_col, x_col=x_col, y_col=y_col)
    labels.index = labels.index.astype(str)
    coords.index = coords.index.astype(str)
    common = X_df.index.intersection(labels.index).intersection(coords.index)
    X_df = X_df.loc[common].copy()
    y_raw = labels.loc[common].astype(str).values
    coord_df = coords.loc[common].copy()

    training_cells = pd.DataFrame({
        "sample_order": np.arange(len(common), dtype=np.int64),
        "cell_id": common.astype(str),
        "original_row": pd.Series(np.arange(len(cell_ids), dtype=np.int64), index=cell_ids).loc[common].values,
        "label": y_raw,
    })
    cell_type_counts = (
        pd.Series(y_raw, name="label")
        .value_counts()
        .rename_axis("label")
        .reset_index(name="synthetic_cells")
        .sort_values("label")
    )
    cell_type_counts["sampled_cells"] = cell_type_counts["synthetic_cells"]
    cell_type_counts["sample_fraction"] = 1.0
    return X_df, y_raw, coord_df, training_cells, cell_type_counts


def train_full_synthetic(
    synthetic_count_npy: str | Path,
    synthetic_label_csv: str | Path,
    synthetic_coord_csv: str | Path,
    cfg: TrainConfig,
    output_dir: str | Path,
    input_normalized: str = "log",
):
    X_df, y_raw, coord_df, training_cells, cell_type_counts = load_full_synthetic_data(
        synthetic_count_npy,
        synthetic_label_csv,
        synthetic_coord_csv,
    )
    genes = X_df.columns.tolist()
    X_log = get_log_expression(X_df, input_normalized=input_normalized)
    gene_min, gene_max, denom = fit_gene_minmax_scaler(X_log)
    X_scaled = transform_gene_minmax_to_minus1_1(X_log, gene_min, denom)
    coords_norm, coord_center, coord_scale = normalize_coordinates(coord_df)
    H, adj_norm, pos_edges, _ = compute_spatial_features_and_graph(coords_norm, k=cfg.graph_k_neighbors)
    neg_edges = sample_negative_edges_fast(
        len(coords_norm),
        pos_edges,
        min(cfg.edge_sample_size, len(pos_edges)),
        seed=cfg.seed,
    )

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    train_idx, val_idx = make_internal_split(
        coords_norm,
        val_fraction=cfg.val_fraction,
        method=cfg.split_method,
        seed=cfg.seed,
    )
    print(f"Full synthetic cells: {len(y)} | Train: {len(train_idx)} | Validation: {len(val_idx)}", flush=True)
    print(f"Cell types in training pool: {len(le.classes_)}", flush=True)
    print(f"Graph k: {cfg.graph_k_neighbors} | Positive edges: {len(pos_edges)} | Negative edges: {len(neg_edges)}", flush=True)

    model, metrics, history = train_one_model(
        X_scaled, y, H, adj_norm, pos_edges, neg_edges,
        train_idx, val_idx, len(le.classes_), cfg,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_cells.to_csv(output_dir / "synthetic_full_training_cells.csv", index=False)
    cell_type_counts.to_csv(output_dir / "synthetic_full_training_cell_type_counts.csv", index=False)
    pd.DataFrame([{"epochs": cfg.epochs, **metrics}]).to_csv(output_dir / "internal_validation_results.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)

    ckpt_path = output_dir / "binvast_best_model.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": {
            "expr_dim": X_scaled.shape[1],
            "graph_feat_dim": H.shape[1],
            "num_classes": len(le.classes_),
            "expr_latent_dim": cfg.expr_latent_dim,
            "graph_latent_dim": cfg.graph_latent_dim,
            "expr_hidden_dims": cfg.expr_hidden_dims,
            "graph_hidden_dims": cfg.graph_hidden_dims,
            "classifier_hidden_dim": cfg.classifier_hidden_dim,
            "graph_k_neighbors": cfg.graph_k_neighbors,
            "graph_weight": cfg.graph_weight,
        },
        "train_config": asdict(cfg),
        "preprocess": {
            "genes": genes,
            "gene_min": gene_min.tolist(),
            "gene_max": gene_max.tolist(),
            "denom": denom.tolist(),
            "coord_center": coord_center.tolist(),
            "coord_scale": coord_scale.tolist(),
            "input_normalized": input_normalized,
            "target_input_normalized": input_normalized,
        },
        "label_classes": le.classes_.tolist(),
        "internal_validation": {
            "sample_fraction_per_label": 1.0,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "best_metrics": metrics,
        },
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Best model saved to: {ckpt_path}", flush=True)

    split = np.array(["train"] * len(y_raw), dtype=object)
    split[val_idx] = "validation"
    expr_latent, graph_latent = model.encode_all(X_scaled, H, adj_norm, batch_size=cfg.batch_size)
    save_latent_csv(
        coord_df.index,
        y_raw,
        expr_latent,
        graph_latent,
        output_dir / "synthetic_full_training_pool_latents.csv",
        split=split,
        coords=coord_df,
    )
    return ckpt_path, checkpoint


def stream_synthetic_expr_latents(
    count_npy: str | Path,
    label_csv: str | Path,
    coord_csv: str | Path,
    ckpt_path: str | Path,
    output_csv: str | Path,
    input_normalized: str = "log",
    batch_size: int = 8192,
):
    model, checkpoint = load_trained_model(ckpt_path)
    arr = np.load(count_npy, mmap_mode="r")
    cell_ids, available_genes = matrix_sidecars(count_npy)
    labels = load_label_data(label_csv)
    coords = load_coordinate_data(coord_csv)
    labels.index = labels.index.astype(str)
    coords.index = coords.index.astype(str)

    ref_genes = checkpoint["preprocess"]["genes"]
    gene_to_idx = {g: i for i, g in enumerate(available_genes)}
    gene_idx = np.array([gene_to_idx[g] for g in ref_genes], dtype=np.int64)
    same_gene_order = np.array_equal(gene_idx, np.arange(len(ref_genes), dtype=np.int64))
    gene_min = np.array(checkpoint["preprocess"]["gene_min"], dtype=np.float32)
    denom = np.array(checkpoint["preprocess"]["denom"], dtype=np.float32)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()
    first = True
    with torch.no_grad():
        for start in range(0, arr.shape[0], batch_size):
            stop = min(start + batch_size, arr.shape[0])
            if start == 0 or stop == arr.shape[0] or (start // batch_size) % 25 == 0:
                print(f"Encoding all synthetic VAE latents: rows {start + 1}-{stop} of {arr.shape[0]}", flush=True)
            X = np.asarray(arr[start:stop, :], dtype=np.float32)
            if not same_gene_order:
                X = X[:, gene_idx]
            X[~np.isfinite(X)] = 0.0
            X[X < 0] = 0.0
            if input_normalized != "log":
                X = log1p_cpm(X)
            X_scaled = transform_gene_minmax_to_minus1_1(X, gene_min, denom)
            xb = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            mu_expr, _ = model.expr_vae.encode(xb)
            out = pd.DataFrame({"cell_id": cell_ids[start:stop]})
            out["label"] = labels.loc[out["cell_id"]].values
            out["x"] = coords.loc[out["cell_id"], "x"].values
            out["y"] = coords.loc[out["cell_id"], "y"].values
            z = mu_expr.detach().cpu().numpy()
            for j in range(z.shape[1]):
                out[f"expr_latent_{j + 1}"] = z[:, j]
            out.to_csv(output_csv, mode="w" if first else "a", index=False, header=first)
            first = False
    print(f"Saved all synthetic VAE latents: {output_csv}", flush=True)


def stream_target_binvast_predictions(
    expression_matrix: str | Path,
    coord_csv: str | Path,
    ckpt_path: str | Path,
    output_csv: str | Path,
    input_normalized: str = "log",
    batch_size: int = 8192,
):
    model, checkpoint = load_trained_model(ckpt_path)
    expression_matrix = Path(expression_matrix)
    cell_ids, available_genes, matrix_shape = expression_matrix_metadata(
        expression_matrix
    )
    coords = load_coordinate_data(coord_csv)
    coords.index = coords.index.astype(str)
    coords = coords.loc[cell_ids]
    coord_center = np.array(checkpoint["preprocess"]["coord_center"], dtype=np.float32)
    coord_scale = np.array(checkpoint["preprocess"]["coord_scale"], dtype=np.float32)
    coords_norm, _, _ = normalize_coordinates(coords, coord_center=coord_center, coord_scale=coord_scale)
    k_neighbors = checkpoint["config"].get("graph_k_neighbors", 30)
    print(f"Building target graph for {matrix_shape[0]} cells with k={k_neighbors}", flush=True)
    H, adj_norm, _, _ = compute_spatial_features_and_graph(coords_norm, k=k_neighbors)

    h_all = torch.tensor(H, dtype=torch.float32, device=device)
    adj_norm = adj_norm.to(device)
    with torch.no_grad():
        z_graph_all, _, _ = model.graph_vae(h_all, adj_norm)

    ref_genes = checkpoint["preprocess"]["genes"]
    gene_to_idx = {g: i for i, g in enumerate(available_genes)}
    gene_idx = np.array([gene_to_idx[g] for g in ref_genes], dtype=np.int64)
    same_gene_order = np.array_equal(gene_idx, np.arange(len(ref_genes), dtype=np.int64))
    gene_min = np.array(checkpoint["preprocess"]["gene_min"], dtype=np.float32)
    denom = np.array(checkpoint["preprocess"]["denom"], dtype=np.float32)
    classes = checkpoint["label_classes"]
    graph_weight = checkpoint["config"].get("graph_weight", 0.1)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()
    first = True
    with torch.no_grad():
        for start, stop, batch_ids, X in iter_expression_batches(
            expression_matrix, batch_size
        ):
            if start == 0 or stop == matrix_shape[0] or (start // batch_size) % 25 == 0:
                print(f"Predicting target BinVAST scores: rows {start + 1}-{stop} of {matrix_shape[0]}", flush=True)
            if not np.array_equal(batch_ids, cell_ids[start:stop]):
                raise ValueError("Target expression cell order changed while streaming")
            if not same_gene_order:
                X = X[:, gene_idx]
            X[~np.isfinite(X)] = 0.0
            X[X < 0] = 0.0
            if input_normalized != "log":
                X = log1p_cpm(X)
            X_scaled = transform_gene_minmax_to_minus1_1(X, gene_min, denom)
            xb = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            idxb = torch.arange(start, stop, dtype=torch.long, device=device)
            mu_expr, _ = model.expr_vae.encode(xb)
            z_graph = z_graph_all[idxb]
            graph_logits = model.graph_cls_head(z_graph)
            fused_logits = model.fused_cls_head(torch.cat([mu_expr, z_graph], dim=1))
            logits = fused_logits + graph_weight * graph_logits
            prob = torch.softmax(logits, dim=1)
            pmax, pidx = prob.max(dim=1)

            out = pd.DataFrame({
                "cell_id": cell_ids[start:stop],
                "x": coords.iloc[start:stop]["x"].values,
                "y": coords.iloc[start:stop]["y"].values,
                "predicted_label": [classes[j] for j in pidx.detach().cpu().numpy()],
                "prediction_confidence": pmax.detach().cpu().numpy(),
            })
            expr_np = mu_expr.detach().cpu().numpy()
            graph_np = z_graph.detach().cpu().numpy()
            for j in range(expr_np.shape[1]):
                out[f"expr_latent_{j + 1}"] = expr_np[:, j]
            for j in range(graph_np.shape[1]):
                out[f"graph_latent_{j + 1}"] = graph_np[:, j]
            prob_np = prob.detach().cpu().numpy()
            for j, class_name in enumerate(classes):
                out[safe_probability_column(class_name)] = prob_np[:, j]
            out.to_csv(output_csv, mode="w" if first else "a", index=False, header=first)
            first = False
    print(f"Saved target BinVAST predictions/latents/scores: {output_csv}", flush=True)


def save_prediction_tables(target_scores_csv: str | Path, output_dir: str | Path, classes: list[str]):
    output_dir = Path(output_dir)
    df = pd.read_csv(target_scores_csv)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    pred_cols = ["cell_id", "x", "y", "predicted_label", "prediction_confidence"] + prob_cols
    pred_csv = output_dir / "target_predicted_labels.csv"
    counts_csv = output_dir / "target_predicted_label_counts.csv"
    df[pred_cols].to_csv(pred_csv, index=False)
    counts = df["predicted_label"].value_counts().rename_axis("predicted_label").reset_index(name="N")
    counts.to_csv(counts_csv, index=False)

    total = int(counts["N"].sum())
    complete = pd.DataFrame({"predicted_label": list(classes)})
    complete = complete.merge(counts, on="predicted_label", how="left")
    complete["N"] = complete["N"].fillna(0).astype(int)
    complete["percentage"] = np.where(total > 0, complete["N"] / total * 100.0, 0.0)
    complete["legend_label"] = complete.apply(
        lambda r: f"{r['predicted_label']} ({r['percentage']:.2f}%)",
        axis=1,
    )
    complete.to_csv(output_dir / "target_predicted_label_counts_with_legend_percent.csv", index=False)
    complete.to_csv(output_dir / "target_predicted_label_counts_all_38_labels.csv", index=False)
    print(f"Saved target predicted labels: {pred_csv}", flush=True)
    print(f"Saved target predicted label counts: {counts_csv}", flush=True)
    return counts, complete


def make_config(
    output_dir: str | Path,
    epochs: int = 100,
    batch_size: int = 8192,
    graph_k: int = 30,
    seed: int = DEFAULT_SEED,
    expr_latent_dim: int = 4,
    graph_latent_dim: int = 4,
):
    return TrainConfig(
        batch_size=int(batch_size),
        lr=1e-3,
        epochs=int(epochs),
        expr_latent_dim=int(expr_latent_dim),
        graph_latent_dim=int(graph_latent_dim),
        expr_hidden_dims=(256, 64, 16),
        graph_hidden_dims=(64, 16),
        classifier_hidden_dim=32,
        graph_k_neighbors=int(graph_k),
        split_method="random",
        graph_reconstruction_weight=0.02,
        classifier_weight=5.0,
        expr_classifier_weight=2.0,
        graph_classifier_weight=1.0,
        graph_weight=0.1,
        save_dir=str(output_dir),
        seed=int(seed),
    )


def run_full_synthetic_binvast(
    project_root: str | Path | None = None,
    target_dir_name: str | Path = "preprocessed_input",
    synthetic_dir_name: str | Path = "synthetic_input",
    output_name: str = "binvast_output",
    epochs: int | None = None,
    batch_size: int | None = None,
    graph_k: int | None = None,
    expr_latent_dim: int = 4,
    graph_latent_dim: int = 4,
    seed: int = DEFAULT_SEED,
):
    project_root = Path(project_root) if project_root is not None else Path.cwd()
    data_dir = resolve_input_dir(project_root, target_dir_name)
    synthetic_dir = resolve_input_dir(project_root, synthetic_dir_name)
    output_dir = project_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_config(
        output_dir=output_dir,
        epochs=int(os.environ.get("BINVAST_EPOCHS", epochs or 100)),
        batch_size=int(os.environ.get("BINVAST_BATCH_SIZE", batch_size or 8192)),
        graph_k=int(os.environ.get("BINVAST_GRAPH_K", graph_k or 30)),
        seed=seed,
        expr_latent_dim=expr_latent_dim,
        graph_latent_dim=graph_latent_dim,
    )

    synthetic_count_npy = synthetic_dir / "synthetic_expression.npy"
    synthetic_label_csv = synthetic_dir / "synthetic_labels.csv"
    synthetic_coord_csv = synthetic_dir / "synthetic_spatial_coords.csv"
    target_count_npy = data_dir / "target_counts.npy"
    target_coord_csv = data_dir / "target_coords.csv"

    for path in [synthetic_count_npy, synthetic_label_csv, synthetic_coord_csv, target_count_npy, target_coord_csv]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Project root:", project_root, flush=True)
    print("Output:", output_dir, flush=True)
    print("Device:", device, flush=True)

    ckpt_path, checkpoint = train_full_synthetic(
        synthetic_count_npy,
        synthetic_label_csv,
        synthetic_coord_csv,
        cfg,
        output_dir,
        input_normalized="log",
    )

    synthetic_all_latents = output_dir / "synthetic_all_expr_vae_latents.csv"
    stream_synthetic_expr_latents(
        synthetic_count_npy,
        synthetic_label_csv,
        synthetic_coord_csv,
        ckpt_path,
        synthetic_all_latents,
        input_normalized="log",
        batch_size=cfg.batch_size,
    )

    target_scores_csv = output_dir / "target_binvast_latents_and_scores.csv"
    stream_target_binvast_predictions(
        target_count_npy,
        target_coord_csv,
        ckpt_path,
        target_scores_csv,
        input_normalized="log",
        batch_size=cfg.batch_size,
    )
    counts, complete_counts = save_prediction_tables(
        target_scores_csv,
        output_dir,
        checkpoint["label_classes"],
    )

    manifest = {
        "dataset": "high_resolution_spatial_transcriptomics",
        "project_root": str(project_root),
        "target_dir": str(data_dir),
        "synthetic_dir": str(synthetic_dir),
        "training_synthetic_fraction_per_cell_type": 1.0,
        "training_synthetic_cells": int(np.load(synthetic_count_npy, mmap_mode="r").shape[0]),
        "target_cells": int(np.load(target_count_npy, mmap_mode="r").shape[0]),
        "seed": seed,
        "graph_k_neighbors": cfg.graph_k_neighbors,
        "expr_latent_dim": cfg.expr_latent_dim,
        "graph_latent_dim": cfg.graph_latent_dim,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "checkpoint": str(ckpt_path),
        "synthetic_full_training_pool_latents": str(output_dir / "synthetic_full_training_pool_latents.csv"),
        "synthetic_all_expr_vae_latents": str(synthetic_all_latents),
        "target_binvast_latents_and_scores": str(target_scores_csv),
        "target_predicted_labels": str(output_dir / "target_predicted_labels.csv"),
        "target_predicted_label_counts": str(output_dir / "target_predicted_label_counts.csv"),
        "target_predicted_label_counts_with_legend_percent": str(
            output_dir / "target_predicted_label_counts_with_legend_percent.csv"
        ),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print(complete_counts.sort_values("N", ascending=False).to_string(index=False), flush=True)
    gc.collect()
    return {
        "output_dir": output_dir,
        "checkpoint": ckpt_path,
        "counts": counts,
        "complete_counts": complete_counts,
        "manifest": manifest,
    }


def train_one_model_with_epoch_snapshots(
    X_scaled,
    y,
    H,
    adj_norm,
    pos_edges,
    neg_edges,
    train_idx,
    val_idx,
    num_classes,
    cfg,
    snapshot_epochs,
):
    """Train once and retain exact model states at requested epochs."""
    snapshot_epochs = sorted({int(epoch) for epoch in snapshot_epochs})
    if not snapshot_epochs or snapshot_epochs[0] < 1:
        raise ValueError("snapshot_epochs must contain positive integers")
    if snapshot_epochs[-1] != int(cfg.epochs):
        raise ValueError("cfg.epochs must equal the largest snapshot epoch")

    ds = CellDataset(X_scaled, y, train_idx)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model = make_model(X_scaled.shape[1], H.shape[1], num_classes, cfg)
    h_all = torch.tensor(H, dtype=torch.float32, device=device)
    adj_on_device = adj_norm.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history = []
    snapshots = {}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {
            "total": 0.0,
            "recon": 0.0,
            "expr_kl": 0.0,
            "graph_rec": 0.0,
            "graph_kl": 0.0,
            "fused_cls": 0.0,
            "expr_cls": 0.0,
            "graph_cls": 0.0,
        }
        n_batches = 0

        for xb, yb, idxb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            idxb = idxb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                out = model(xb, idxb, h_all, adj_on_device)
                total, parts = multimodal_loss(
                    model,
                    xb,
                    yb,
                    idxb,
                    out,
                    pos_edges,
                    neg_edges,
                    cfg,
                    epoch,
                )
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

            for key in running:
                running[key] += float(parts[key].detach().cpu())
            n_batches += 1

        row = {key: value / max(n_batches, 1) for key, value in running.items()}
        row["epoch"] = epoch
        val_metrics = None
        if epoch == 1 or epoch % 10 == 0 or epoch in snapshot_epochs or epoch == cfg.epochs:
            val_metrics = evaluate_model(
                model,
                X_scaled,
                y,
                H,
                adj_on_device,
                val_idx,
                cfg,
                cfg.batch_size,
            )
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | Total: {row['total']:.4f} | "
                f"Recon: {row['recon']:.4f} | FusedCls: {row['fused_cls']:.4f} | "
                f"ExprCls: {row['expr_cls']:.4f} | GraphCls: {row['graph_cls']:.4f} | "
                f"ValF1: {val_metrics['macro_f1']:.4f} | "
                f"ValAcc: {val_metrics['accuracy']:.4f} | "
                f"Score: {val_metrics['selection_score']:.4f}",
                flush=True,
            )
        history.append(row)

        if epoch in snapshot_epochs:
            if val_metrics is None:
                val_metrics = evaluate_model(
                    model,
                    X_scaled,
                    y,
                    H,
                    adj_on_device,
                    val_idx,
                    cfg,
                    cfg.batch_size,
                )
            snapshots[epoch] = {
                "model_state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "metrics": dict(val_metrics),
            }

    return model, snapshots, pd.DataFrame(history)


def save_synthetic_predictions_and_latents(
    model,
    expr_latent,
    graph_latent,
    cell_ids,
    labels,
    coords,
    classes,
    graph_weight,
    output_csv,
    batch_size,
):
    """Save synthetic labels, predictions, probabilities, and both latent spaces."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()
    model.eval()
    first = True
    with torch.no_grad():
        for start in range(0, len(cell_ids), batch_size):
            stop = min(start + batch_size, len(cell_ids))
            expr_batch = torch.tensor(expr_latent[start:stop], dtype=torch.float32, device=device)
            graph_batch = torch.tensor(graph_latent[start:stop], dtype=torch.float32, device=device)
            graph_logits = model.graph_cls_head(graph_batch)
            fused_logits = model.fused_cls_head(torch.cat([expr_batch, graph_batch], dim=1))
            probability = torch.softmax(fused_logits + graph_weight * graph_logits, dim=1)
            confidence, predicted_index = probability.max(dim=1)

            out = pd.DataFrame({
                "cell_id": cell_ids[start:stop],
                "label": labels[start:stop],
                "x": coords.iloc[start:stop]["x"].values,
                "y": coords.iloc[start:stop]["y"].values,
                "predicted_label": [classes[index] for index in predicted_index.cpu().numpy()],
                "prediction_confidence": confidence.cpu().numpy(),
            })
            for column in range(expr_latent.shape[1]):
                out[f"expr_latent_{column + 1}"] = expr_latent[start:stop, column]
            for column in range(graph_latent.shape[1]):
                out[f"graph_latent_{column + 1}"] = graph_latent[start:stop, column]
            probability_np = probability.cpu().numpy()
            for column, class_name in enumerate(classes):
                out[safe_probability_column(class_name)] = probability_np[:, column]
            out.to_csv(output_csv, mode="w" if first else "a", index=False, header=first)
            first = False
    print(f"Saved synthetic BinVAST predictions/latents/scores: {output_csv}", flush=True)


def run_full_synthetic_binvast_epoch_grid(
    project_root: str | Path | None = None,
    target_dir_name: str | Path = "preprocessed_input",
    synthetic_dir_name: str | Path = "synthetic_input",
    output_name: str = "binvast_result_full_synthetic_epoch_grid",
    snapshot_epochs=(50, 60, 70, 80, 90, 100),
    batch_size: int = 8192,
    graph_k: int = 30,
    expr_latent_dim: int = 4,
    graph_latent_dim: int = 4,
    seed: int = DEFAULT_SEED,
):
    """Train one uninterrupted trajectory and export each requested epoch snapshot."""
    snapshot_epochs = sorted({int(epoch) for epoch in snapshot_epochs})
    if not snapshot_epochs:
        raise ValueError("snapshot_epochs cannot be empty")

    project_root = Path(project_root) if project_root is not None else Path.cwd()
    data_dir = resolve_input_dir(project_root, target_dir_name)
    synthetic_dir = resolve_input_dir(project_root, synthetic_dir_name)
    grid_output_dir = project_root / output_name
    grid_output_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_config(
        output_dir=grid_output_dir,
        epochs=max(snapshot_epochs),
        batch_size=int(batch_size),
        graph_k=int(graph_k),
        seed=seed,
        expr_latent_dim=expr_latent_dim,
        graph_latent_dim=graph_latent_dim,
    )

    synthetic_expression = resolve_first_existing(
        synthetic_dir,
        ("synthetic_expression.csv", "synthetic_expression.npy"),
    )
    synthetic_label_csv = synthetic_dir / "synthetic_labels.csv"
    synthetic_coord_csv = synthetic_dir / "synthetic_spatial_coords.csv"
    target_expression = resolve_first_existing(
        data_dir,
        ("target_expression_log1p_cp10k.csv", "target_counts.csv", "target_counts.npy"),
    )
    target_coord_csv = resolve_first_existing(
        data_dir,
        ("target_coordinates.csv", "target_coords.csv"),
    )
    for path in [
        synthetic_expression,
        synthetic_label_csv,
        synthetic_coord_csv,
        target_expression,
        target_coord_csv,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    X_df, y_raw, coord_df, training_cells, cell_type_counts = load_full_synthetic_data(
        synthetic_expression,
        synthetic_label_csv,
        synthetic_coord_csv,
    )
    genes = X_df.columns.tolist()
    X_log = get_log_expression(X_df, input_normalized="log")
    gene_min, gene_max, denom = fit_gene_minmax_scaler(X_log)
    X_scaled = transform_gene_minmax_to_minus1_1(X_log, gene_min, denom)
    coords_norm, coord_center, coord_scale = normalize_coordinates(coord_df)
    H, adj_norm, pos_edges, _ = compute_spatial_features_and_graph(
        coords_norm,
        k=cfg.graph_k_neighbors,
    )
    neg_edges = sample_negative_edges_fast(
        len(coords_norm),
        pos_edges,
        min(cfg.edge_sample_size, len(pos_edges)),
        seed=cfg.seed,
    )

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    train_idx, val_idx = make_internal_split(
        coords_norm,
        val_fraction=cfg.val_fraction,
        method=cfg.split_method,
        seed=cfg.seed,
    )
    print(
        f"Continuous training | batch={cfg.batch_size} | max_epoch={cfg.epochs} | "
        f"snapshots={snapshot_epochs}",
        flush=True,
    )
    print(
        f"Full synthetic cells: {len(y)} | Train: {len(train_idx)} | "
        f"Validation: {len(val_idx)} | Cell types: {len(le.classes_)}",
        flush=True,
    )

    model, snapshots, history = train_one_model_with_epoch_snapshots(
        X_scaled,
        y,
        H,
        adj_norm,
        pos_edges,
        neg_edges,
        train_idx,
        val_idx,
        len(le.classes_),
        cfg,
        snapshot_epochs,
    )

    split = np.array(["train"] * len(y_raw), dtype=object)
    split[val_idx] = "validation"
    snapshot_results = {}
    for epoch in snapshot_epochs:
        run_name = f"epochs_{epoch:03d}_batch_{cfg.batch_size:05d}"
        output_dir = grid_output_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshots[epoch]
        model.load_state_dict(snapshot["model_state_dict"])

        training_cells.to_csv(output_dir / "synthetic_full_training_cells.csv", index=False)
        cell_type_counts.to_csv(
            output_dir / "synthetic_full_training_cell_type_counts.csv",
            index=False,
        )
        pd.DataFrame([{"epochs": epoch, **snapshot["metrics"]}]).to_csv(
            output_dir / "internal_validation_results.csv",
            index=False,
        )
        history.loc[history["epoch"] <= epoch].to_csv(
            output_dir / "training_history.csv",
            index=False,
        )

        snapshot_train_config = asdict(cfg)
        snapshot_train_config["epochs"] = epoch
        checkpoint = {
            "model_state_dict": snapshot["model_state_dict"],
            "config": {
                "expr_dim": X_scaled.shape[1],
                "graph_feat_dim": H.shape[1],
                "num_classes": len(le.classes_),
                "expr_latent_dim": cfg.expr_latent_dim,
                "graph_latent_dim": cfg.graph_latent_dim,
                "expr_hidden_dims": cfg.expr_hidden_dims,
                "graph_hidden_dims": cfg.graph_hidden_dims,
                "classifier_hidden_dim": cfg.classifier_hidden_dim,
                "graph_k_neighbors": cfg.graph_k_neighbors,
                "graph_weight": cfg.graph_weight,
            },
            "train_config": snapshot_train_config,
            "preprocess": {
                "genes": genes,
                "gene_min": gene_min.tolist(),
                "gene_max": gene_max.tolist(),
                "denom": denom.tolist(),
                "coord_center": coord_center.tolist(),
                "coord_scale": coord_scale.tolist(),
                "input_normalized": "log",
                "target_input_normalized": "log",
            },
            "label_classes": le.classes_.tolist(),
            "internal_validation": {
                "sample_fraction_per_label": 1.0,
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "snapshot_epoch": epoch,
                "metrics": snapshot["metrics"],
            },
            "snapshot": {
                "epoch": epoch,
                "continuous_training_max_epoch": cfg.epochs,
                "batch_size": cfg.batch_size,
            },
        }
        ckpt_path = output_dir / "binvast_best_model.pt"
        torch.save(checkpoint, ckpt_path)

        expr_latent, graph_latent = model.encode_all(
            X_scaled,
            H,
            adj_norm,
            batch_size=cfg.batch_size,
        )
        synthetic_pool_latents = output_dir / "synthetic_full_training_pool_latents.csv"
        save_latent_csv(
            coord_df.index,
            y_raw,
            expr_latent,
            graph_latent,
            synthetic_pool_latents,
            split=split,
            coords=coord_df,
        )

        synthetic_expr_latents = output_dir / "synthetic_all_expr_vae_latents.csv"
        pd.DataFrame({
            "cell_id": coord_df.index.astype(str),
            "label": y_raw,
            "x": coord_df["x"].values,
            "y": coord_df["y"].values,
            **{
                f"expr_latent_{column + 1}": expr_latent[:, column]
                for column in range(expr_latent.shape[1])
            },
        }).to_csv(synthetic_expr_latents, index=False)

        synthetic_scores_csv = output_dir / "synthetic_binvast_latents_and_scores.csv"
        save_synthetic_predictions_and_latents(
            model,
            expr_latent,
            graph_latent,
            coord_df.index.astype(str).to_numpy(),
            np.asarray(y_raw),
            coord_df,
            le.classes_.tolist(),
            cfg.graph_weight,
            synthetic_scores_csv,
            cfg.batch_size,
        )

        target_scores_csv = output_dir / "target_binvast_latents_and_scores.csv"
        stream_target_binvast_predictions(
            target_expression,
            target_coord_csv,
            ckpt_path,
            target_scores_csv,
            input_normalized="log",
            batch_size=cfg.batch_size,
        )
        counts, complete_counts = save_prediction_tables(
            target_scores_csv,
            output_dir,
            le.classes_.tolist(),
        )

        manifest = {
            "dataset": "high_resolution_spatial_transcriptomics",
            "training_mode": "continuous_epoch_snapshots",
            "project_root": str(project_root),
            "target_dir": str(data_dir),
            "synthetic_dir": str(synthetic_dir),
            "snapshot_epoch": epoch,
            "continuous_training_max_epoch": cfg.epochs,
            "batch_size": cfg.batch_size,
            "seed": seed,
            "graph_k_neighbors": cfg.graph_k_neighbors,
            "expr_latent_dim": cfg.expr_latent_dim,
            "graph_latent_dim": cfg.graph_latent_dim,
            "training_synthetic_cells": int(X_scaled.shape[0]),
            "target_cells": int(expression_matrix_metadata(target_expression)[2][0]),
            "checkpoint": str(ckpt_path),
            "synthetic_full_training_pool_latents": str(synthetic_pool_latents),
            "synthetic_all_expr_vae_latents": str(synthetic_expr_latents),
            "synthetic_binvast_latents_and_scores": str(synthetic_scores_csv),
            "target_binvast_latents_and_scores": str(target_scores_csv),
            "target_predicted_labels": str(output_dir / "target_predicted_labels.csv"),
            "target_predicted_label_counts_with_legend_percent": str(
                output_dir / "target_predicted_label_counts_with_legend_percent.csv"
            ),
        }
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
        snapshot_results[epoch] = {
            "output_dir": output_dir,
            "checkpoint": ckpt_path,
            "counts": counts,
            "complete_counts": complete_counts,
            "manifest": manifest,
        }
        print(f"Saved epoch {epoch} snapshot outputs to: {output_dir}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "grid_output_dir": grid_output_dir,
        "batch_size": cfg.batch_size,
        "snapshot_epochs": snapshot_epochs,
        "snapshots": snapshot_results,
        "training_history": history,
    }


if __name__ == "__main__":
    run_full_synthetic_binvast()
