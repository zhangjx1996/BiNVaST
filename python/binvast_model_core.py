# Core BinVAST model components for training and prediction.
# Synthetic data generation is handled by generate_synthetic_csv.py.

# %% Cell 1
# Cell 1: Import packages + config
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import itertools
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder


# Reproducibility
torch.manual_seed(259)
np.random.seed(259)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Cell 2
# Cell 2: I/O + preprocessing helpers
def read_matrix(path):
    """
    Load an expression matrix with rows=cells and columns=genes.

    CSV/TSV files use the first column as cell IDs. NPY files use sidecars:
    <stem>_cell_ids.csv and <stem>_genes.csv.
    """
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        X = np.load(path, mmap_mode=None).astype(np.float32, copy=False)
        stem = str(path)[:-4]
        cell_path = Path(stem + "_cell_ids.csv")
        gene_path = Path(stem + "_genes.csv")
        cell_ids = (
            pd.read_csv(cell_path).iloc[:, 0].astype(str).tolist()
            if cell_path.exists()
            else [f"cell_{i + 1}" for i in range(X.shape[0])]
        )
        genes = (
            pd.read_csv(gene_path).iloc[:, 0].astype(str).tolist()
            if gene_path.exists()
            else [f"gene_{j + 1}" for j in range(X.shape[1])]
        )
        if X.shape != (len(cell_ids), len(genes)):
            raise ValueError(f"{path} shape {X.shape} does not match its sidecar IDs/genes.")
        return X, cell_ids, genes
    if suffix == ".npz":
        loaded = np.load(path)
        key = "X" if "X" in loaded.files else loaded.files[0]
        X = loaded[key].astype(np.float32, copy=False)
        return X, [f"cell_{i + 1}" for i in range(X.shape[0])], [f"gene_{j + 1}" for j in range(X.shape[1])]
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix in {".tsv", ".txt"} else ","
        df = pd.read_csv(path, sep=sep, index_col=0)
        return df.to_numpy(dtype=np.float32), df.index.astype(str).tolist(), df.columns.astype(str).tolist()
    raise ValueError(f"Unsupported matrix format: {path}")


def matrix_to_dataframe(path):
    X, cell_ids, genes = read_matrix(path)
    return pd.DataFrame(X, index=cell_ids, columns=genes)


def choose_column(df, preferred=None, candidates=("gene", "genes", "Gene", "symbol")):
    if preferred:
        if preferred not in df.columns:
            raise ValueError(f"Column '{preferred}' was not found.")
        return preferred
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return df.columns[0]


def read_gene_pairs(gene_csv=None, gene_column=None, query_gene_column=None, ref_gene_column=None):
    """
    Optional gene selector. If query/ref columns differ, this can also map query
    genes to reference genes while preserving an output gene name.
    """
    if gene_csv is None:
        return None
    df = pd.read_csv(Path(gene_csv).expanduser())
    candidates = ("gene", "genes", "Gene", "symbol", "query_gene", "target_gene", "visium_gene", "ref_gene")
    query_col = choose_column(df, query_gene_column or gene_column, candidates)
    ref_col = choose_column(df, ref_gene_column or gene_column, candidates)
    out_col = choose_column(df, gene_column or query_col, candidates)
    pairs = df[[query_col, ref_col, out_col]].dropna().astype(str)
    pairs = pairs[(pairs[query_col] != "") & (pairs[ref_col] != "")]
    if pairs.empty:
        raise ValueError(f"No usable genes found in {gene_csv}.")
    return pairs[query_col].tolist(), pairs[ref_col].tolist(), pairs[out_col].tolist()


def subset_matrix_by_genes(X, available_genes, selected_genes, dataset_name, fill_missing=False):
    gene_to_idx = {gene: i for i, gene in enumerate(available_genes)}
    missing = [gene for gene in selected_genes if gene not in gene_to_idx]
    if missing and not fill_missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{dataset_name} is missing {len(missing)} selected genes, e.g. {preview}. "
            "Set fill_missing_genes=True to fill absent genes with zeros."
        )
    cols = [
        X[:, gene_to_idx[gene]] if gene in gene_to_idx else np.zeros(X.shape[0], dtype=np.float32)
        for gene in selected_genes
    ]
    return np.column_stack(cols).astype(np.float32, copy=False)


def align_target_reference_genes(target, target_genes, ref, ref_genes, gene_pairs=None, fill_missing=False):
    if gene_pairs is None:
        ref_set = set(ref_genes)
        shared = [gene for gene in target_genes if gene in ref_set]
        if not shared:
            raise ValueError("No shared genes found between target and reference matrices.")
        return (
            subset_matrix_by_genes(target, target_genes, shared, "target", fill_missing),
            subset_matrix_by_genes(ref, ref_genes, shared, "reference", fill_missing),
            shared,
        )

    target_selected, ref_selected, output_genes = gene_pairs
    return (
        subset_matrix_by_genes(target, target_genes, target_selected, "target", fill_missing),
        subset_matrix_by_genes(ref, ref_genes, ref_selected, "reference", fill_missing),
        output_genes,
    )


def read_vector_aligned(path, expected_ids, column=None, id_column="cell_id", name="vector"):
    path = Path(path).expanduser()
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=True)
        if len(values) != len(expected_ids):
            raise ValueError(f"{name} length must equal matrix rows.")
        return values

    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    value_col = column
    if value_col is None:
        non_id_cols = [c for c in df.columns if c != id_column]
        if len(non_id_cols) != 1:
            raise ValueError(f"{path} has multiple value columns; pass a column name.")
        value_col = non_id_cols[0]
    if id_column in df.columns:
        values = df.set_index(id_column).reindex(expected_ids)[value_col]
        if values.isna().any():
            missing = values[values.isna()].index.astype(str).tolist()[:10]
            raise ValueError(f"{name} is missing IDs, e.g. {', '.join(missing)}.")
        return values.to_numpy()
    if len(df) != len(expected_ids):
        raise ValueError(f"{name} length must equal matrix rows when no '{id_column}' column is present.")
    return df[value_col].to_numpy()


def read_coords_aligned(path, expected_ids, id_column="cell_id", x_col="x", y_col="y"):
    coords = pd.read_csv(Path(path).expanduser())
    if id_column not in coords.columns:
        raise ValueError(f"{path} must contain '{id_column}' for coordinate alignment.")
    coords = coords.set_index(id_column).reindex(expected_ids)
    if coords[[x_col, y_col]].isna().any(axis=1).any():
        missing = coords.index[coords[[x_col, y_col]].isna().any(axis=1)].astype(str).tolist()[:10]
        raise ValueError(f"Coordinate file is missing IDs, e.g. {', '.join(missing)}.")
    coords = coords.rename(columns={x_col: "x", y_col: "y"})
    coords = coords[["x", "y"]].astype(np.float32)
    coords.insert(0, id_column, coords.index.astype(str))
    return coords.reset_index(drop=True)


def load_label_data(label_csv, cell_id_col="cell_id", label_col="label"):
    y_df = pd.read_csv(Path(label_csv).expanduser())
    y_df = y_df[[cell_id_col, label_col]].dropna()
    y_df = y_df.set_index(cell_id_col)
    return y_df[label_col].astype(str)


def load_coordinate_data(coord_csv, cell_id_col="cell_id", x_col="x", y_col="y"):
    coord_df = pd.read_csv(Path(coord_csv).expanduser())
    coord_df = coord_df[[cell_id_col, x_col, y_col]].dropna()
    coord_df = coord_df.set_index(cell_id_col)
    coord_df = coord_df.rename(columns={x_col: "x", y_col: "y"})
    return coord_df.astype(np.float32)


def load_synthetic_data(count_csv, label_csv, coord_csv,
                        cell_id_col="cell_id", label_col="label",
                        x_col="x", y_col="y"):
    X_df = matrix_to_dataframe(count_csv)
    y = load_label_data(label_csv, cell_id_col=cell_id_col, label_col=label_col)
    coord_df = load_coordinate_data(coord_csv, cell_id_col=cell_id_col, x_col=x_col, y_col=y_col)
    common_cells = X_df.index.intersection(y.index).intersection(coord_df.index)
    return X_df.loc[common_cells].copy(), y.loc[common_cells].values, coord_df.loc[common_cells].copy()


def load_target_data(count_csv, coord_csv, cell_id_col="cell_id", x_col="x", y_col="y"):
    X_df = matrix_to_dataframe(count_csv)
    coord_df = load_coordinate_data(coord_csv, cell_id_col=cell_id_col, x_col=x_col, y_col=y_col)
    common_cells = X_df.index.intersection(coord_df.index)
    return X_df.loc[common_cells].copy(), coord_df.loc[common_cells].copy()


def log1p_cpm(X, scale_factor=1e4, eps=1e-12):
    row_sums = np.maximum(np.nansum(X, axis=1, keepdims=True), eps)
    return np.log1p(X / row_sums * scale_factor).astype(np.float32, copy=False)


def get_log_expression(X_df, input_normalized="counts", scale_factor=1e4):
    X = X_df.values.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    X[X < 0] = 0.0
    if input_normalized == "log":
        return X
    if input_normalized == "counts":
        return log1p_cpm(X, scale_factor=scale_factor)
    raise ValueError("input_normalized must be either 'counts' or 'log'.")


def fit_gene_minmax_scaler(X_log, eps=1e-8):
    gene_min = X_log.min(axis=0)
    gene_max = X_log.max(axis=0)
    denom = np.maximum(gene_max - gene_min, eps)
    return gene_min, gene_max, denom


def transform_gene_minmax_to_minus1_1(X_log, gene_min, denom):
    X_scaled = ((X_log - gene_min) / denom) * 2.0 - 1.0
    return np.clip(X_scaled, -1.0, 1.0).astype(np.float32)


def align_genes_to_reference(X_df, ref_genes):
    X = X_df.copy()
    missing = [g for g in ref_genes if g not in X.columns]
    for gene in missing:
        X[gene] = 0.0
    return X.loc[:, ref_genes]


def preprocess_with_saved_params(X_df, preprocess_dict, input_normalized_override=None):
    ref_genes = preprocess_dict["genes"]
    gene_min = np.array(preprocess_dict["gene_min"], dtype=np.float32)
    denom = np.array(preprocess_dict["denom"], dtype=np.float32)
    input_normalized = input_normalized_override or preprocess_dict.get(
        "target_input_normalized",
        preprocess_dict.get("input_normalized", "counts")
    )
    X_aligned = align_genes_to_reference(X_df, ref_genes)
    X_log = get_log_expression(X_aligned, input_normalized=input_normalized)
    return transform_gene_minmax_to_minus1_1(X_log, gene_min, denom)


def normalize_coordinates(coord_df, coord_center=None, coord_scale=None, eps=1e-8):
    coords = coord_df[["x", "y"]].values.astype(np.float32)
    if coord_center is None:
        coord_center = coords.mean(axis=0)
    if coord_scale is None:
        coord_scale = coords.std(axis=0)
    coord_scale = np.maximum(coord_scale, eps)
    coords_norm = (coords - coord_center) / coord_scale
    return coords_norm.astype(np.float32), coord_center.astype(np.float32), coord_scale.astype(np.float32)

# %% Cell 3
# Cell 4: Spatial graph utilities
# ---------------------------------------------------------------------
# These helpers build spatial node features and a kNN graph for the graph VAE.
# This version uses spatial graph features without neighborhood label correction.

def compute_spatial_node_features(coords_norm, k=100, eps=1e-8):
    n = coords_norm.shape[0]
    k_eff = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(coords_norm)
    distances, indices = nbrs.kneighbors(coords_norm)
    nn_dist = distances[:, 1:]
    mean_dist = nn_dist.mean(axis=1, keepdims=True)
    density = (1.0 / (nn_dist + eps)).mean(axis=1, keepdims=True)
    H = np.concatenate([coords_norm, density, mean_dist], axis=1)
    H_mean = H.mean(axis=0, keepdims=True)
    H_std = np.maximum(H.std(axis=0, keepdims=True), eps)
    return ((H - H_mean) / H_std).astype(np.float32), indices[:, 1:]


def edges_from_neighbor_index(neighbor_indices, k=None, symmetrize=True):
    if k is None:
        k = neighbor_indices.shape[1]
    k = min(k, neighbor_indices.shape[1])
    edge_set = set()
    for i in range(neighbor_indices.shape[0]):
        for j in neighbor_indices[i, :k]:
            if i == j:
                continue
            edge = tuple(sorted((int(i), int(j)))) if symmetrize else (int(i), int(j))
            edge_set.add(edge)
    if not edge_set:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(edge_set), dtype=np.int64)


def build_knn_graph(coords_norm, k=100, symmetrize=True, add_self_loops=True):
    n = coords_norm.shape[0]
    k_eff = min(k + 1, n)
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(coords_norm)
    _, indices = nbrs.kneighbors(coords_norm)
    neighbor_indices = indices[:, 1:].astype(np.int64, copy=False)
    pos_edges = edges_from_neighbor_index(neighbor_indices, k=k, symmetrize=symmetrize)

    rows, cols = [], []
    if len(pos_edges) > 0:
        rows.extend(pos_edges[:, 0].tolist())
        cols.extend(pos_edges[:, 1].tolist())
        if symmetrize:
            rows.extend(pos_edges[:, 1].tolist())
            cols.extend(pos_edges[:, 0].tolist())
    if add_self_loops:
        rows.extend(range(n))
        cols.extend(range(n))

    values = np.ones(len(rows), dtype=np.float32)
    degree = np.zeros(n, dtype=np.float32)
    for r in rows:
        degree[r] += 1.0
    degree = np.maximum(degree, 1.0)
    norm_values = [v / np.sqrt(degree[r] * degree[c]) for r, c, v in zip(rows, cols, values)]
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    val = torch.tensor(norm_values, dtype=torch.float32)
    adj_norm_sparse = torch.sparse_coo_tensor(idx, val, size=(n, n)).coalesce()
    return adj_norm_sparse, pos_edges, neighbor_indices


def sample_negative_edges(n_nodes, pos_edges, n_samples, seed=259):
    rng = np.random.default_rng(seed)
    pos_set = {tuple(sorted((int(a), int(b)))) for a, b in pos_edges}
    neg_edges = []
    max_trials = max(10000, n_samples * 20)
    trials = 0
    while len(neg_edges) < n_samples and trials < max_trials:
        a = int(rng.integers(0, n_nodes))
        b = int(rng.integers(0, n_nodes))
        trials += 1
        if a == b:
            continue
        edge = tuple(sorted((a, b)))
        if edge in pos_set:
            continue
        neg_edges.append(edge)
    if len(neg_edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(neg_edges, dtype=np.int64)


def make_internal_split(coords_norm, val_fraction=0.2, method="spatial_crop", seed=259):
    rng = np.random.default_rng(seed)
    n = coords_norm.shape[0]
    n_val = max(1, min(n - 1, int(round(val_fraction * n))))
    if method == "random":
        val_idx = rng.choice(n, size=n_val, replace=False)
    else:
        center_idx = int(rng.integers(0, n))
        center = coords_norm[center_idx]
        dist = np.linalg.norm(coords_norm - center, axis=1)
        val_idx = np.argsort(dist)[:n_val]
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    train_idx = np.where(~val_mask)[0]
    return train_idx.astype(np.int64), np.asarray(val_idx, dtype=np.int64)


def vae_kl_loss(mu, logvar):
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def graph_reconstruction_loss(z_graph_all, pos_edges, neg_edges, max_edges=20000, seed=259):
    rng = np.random.default_rng(seed)
    if len(pos_edges) > max_edges:
        pos_edges = pos_edges[rng.choice(len(pos_edges), size=max_edges, replace=False)]
    if len(neg_edges) > max_edges:
        neg_edges = neg_edges[rng.choice(len(neg_edges), size=max_edges, replace=False)]
    if len(pos_edges) == 0 or len(neg_edges) == 0:
        return torch.tensor(0.0, device=z_graph_all.device)

    pos_t = torch.tensor(pos_edges, dtype=torch.long, device=z_graph_all.device)
    neg_t = torch.tensor(neg_edges, dtype=torch.long, device=z_graph_all.device)
    logits = torch.cat([
        GraphVAE.decode_edges(z_graph_all, pos_t),
        GraphVAE.decode_edges(z_graph_all, neg_t),
    ])
    labels = torch.cat([
        torch.ones(len(pos_t), device=z_graph_all.device),
        torch.zeros(len(neg_t), device=z_graph_all.device),
    ])
    return F.binary_cross_entropy_with_logits(logits, labels)

# %% Cell 4
# Cell 5: Class objects
class CellDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = torch.tensor(X[indices], dtype=torch.float32)
        self.y = torch.tensor(y[indices], dtype=torch.long)
        self.indices = torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.indices[idx]


class ExpressionVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4, hidden_dims=(256, 64, 16)):
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.enc1 = nn.Linear(input_dim, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.enc3 = nn.Linear(h2, h3)
        self.mu_layer = nn.Linear(h3, latent_dim)
        self.logvar_layer = nn.Linear(h3, latent_dim)
        self.dec1 = nn.Linear(latent_dim, h3)
        self.dec2 = nn.Linear(h3, h2)
        self.dec3 = nn.Linear(h2, h1)
        self.out_layer = nn.Linear(h1, input_dim)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        return self.mu_layer(x), self.logvar_layer(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.dec1(z))
        z = F.relu(self.dec2(z))
        z = F.relu(self.dec3(z))
        return torch.tanh(self.out_layer(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, h, adj_norm):
        h = h.float()
        adj_norm = adj_norm.float()
        return self.linear(torch.sparse.mm(adj_norm, h))


class GraphVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4, hidden_dims=(64, 16)):
        super().__init__()
        h1, h2 = hidden_dims
        self.gcn1 = GraphConvolution(input_dim, h1)
        self.gcn2 = GraphConvolution(h1, h2)
        self.mu_layer = GraphConvolution(h2, latent_dim)
        self.logvar_layer = GraphConvolution(h2, latent_dim)

    def encode(self, h, adj_norm):
        h = F.relu(self.gcn1(h, adj_norm))
        h = F.relu(self.gcn2(h, adj_norm))
        return self.mu_layer(h, adj_norm), self.logvar_layer(h, adj_norm)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h, adj_norm):
        mu, logvar = self.encode(h, adj_norm)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    @staticmethod
    def decode_edges(z, edge_index):
        zi = z[edge_index[:, 0]]
        zj = z[edge_index[:, 1]]
        return torch.sum(zi * zj, dim=1)


class BinVAST(nn.Module):
    def __init__(self, expr_dim, graph_feat_dim, num_classes,
                 expr_latent_dim=4, graph_latent_dim=4,
                 expr_hidden_dims=(256, 64, 16),
                 graph_hidden_dims=(64, 16),
                 classifier_hidden_dim=32,
                 graph_weight=0.1):
        super().__init__()
        self.graph_weight = graph_weight
        self.expr_vae = ExpressionVAE(expr_dim, expr_latent_dim, expr_hidden_dims)
        self.graph_vae = GraphVAE(graph_feat_dim, graph_latent_dim, graph_hidden_dims)
        self.expr_cls_head = nn.Linear(expr_latent_dim, num_classes)
        self.graph_cls_head = nn.Linear(graph_latent_dim, num_classes)
        self.fused_cls_head = nn.Sequential(
            nn.Linear(expr_latent_dim + graph_latent_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x, idx, h_all, adj_norm):
        x_hat, z_expr, mu_expr, logvar_expr = self.expr_vae(x)
        expr_logits = self.expr_cls_head(mu_expr)
        z_graph_all, mu_graph_all, logvar_graph_all = self.graph_vae(h_all, adj_norm)
        z_graph = z_graph_all[idx]
        graph_logits = self.graph_cls_head(z_graph)
        fused_logits = self.fused_cls_head(torch.cat([mu_expr, z_graph], dim=1))
        logits = fused_logits + self.graph_weight * graph_logits
        return {
            "x_hat": x_hat,
            "mu_expr": mu_expr,
            "logvar_expr": logvar_expr,
            "z_graph_all": z_graph_all,
            "z_graph": z_graph,
            "mu_graph_all": mu_graph_all,
            "logvar_graph_all": logvar_graph_all,
            "expr_logits": expr_logits,
            "graph_logits": graph_logits,
            "fused_logits": fused_logits,
            "logits": logits,
        }

    @torch.no_grad()
    def encode_all(self, X_scaled, H, adj_norm, batch_size=8192):
        self.eval()
        h_all = torch.tensor(H, dtype=torch.float32, device=device)
        adj_norm = adj_norm.to(device)
        z_graph_all, _, _ = self.graph_vae(h_all, adj_norm)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        expr_latents = []
        graph_latents = []
        for start in range(0, len(X_tensor), batch_size):
            stop = min(start + batch_size, len(X_tensor))
            xb = X_tensor[start:stop].to(device)
            idxb = torch.arange(start, stop, dtype=torch.long, device=device)
            mu_expr, _ = self.expr_vae.encode(xb)
            expr_latents.append(mu_expr.detach().cpu().numpy())
            graph_latents.append(z_graph_all[idxb].detach().cpu().numpy())
        return np.vstack(expr_latents), np.vstack(graph_latents)

# %% Cell 5
# Cell 6: Training, internal validation, checkpointing, prediction, and visualization helpers
@dataclass
class TrainConfig:
    batch_size: int = 8192
    lr: float = 1e-3
    epochs: int = 100
    expr_latent_dim: int = 4
    graph_latent_dim: int = 4
    expr_hidden_dims: tuple = (256, 64, 16)
    graph_hidden_dims: tuple = (64, 16)
    classifier_hidden_dim: int = 32
    graph_k_neighbors: int = 100
    val_fraction: float = 0.2
    split_method: str = "spatial_crop"
    expr_kl_weight: float = 1e-3
    graph_kl_weight: float = 1e-3
    graph_reconstruction_weight: float = 0.02
    classifier_weight: float = 5.0
    expr_classifier_weight: float = 2.0
    graph_classifier_weight: float = 1.0
    graph_weight: float = 0.1
    edge_sample_size: int = 20000
    seed: int = 259
    save_dir: str = "./binvast_output"


def make_model(expr_dim, graph_feat_dim, num_classes, cfg):
    return BinVAST(
        expr_dim=expr_dim,
        graph_feat_dim=graph_feat_dim,
        num_classes=num_classes,
        expr_latent_dim=cfg.expr_latent_dim,
        graph_latent_dim=cfg.graph_latent_dim,
        expr_hidden_dims=cfg.expr_hidden_dims,
        graph_hidden_dims=cfg.graph_hidden_dims,
        classifier_hidden_dim=cfg.classifier_hidden_dim,
        graph_weight=cfg.graph_weight,
    ).to(device)


def multimodal_loss(model, x, y, idxb, out, pos_edges, neg_edges, cfg, epoch):
    recon_loss = F.mse_loss(out["x_hat"], x, reduction="mean")
    expr_kl = vae_kl_loss(out["mu_expr"], out["logvar_expr"])
    graph_kl = vae_kl_loss(out["mu_graph_all"], out["logvar_graph_all"])
    fused_cls = F.cross_entropy(out["logits"], y)
    expr_cls = F.cross_entropy(out["expr_logits"], y)
    graph_cls = F.cross_entropy(out["graph_logits"], y)
    graph_rec = graph_reconstruction_loss(
        out["z_graph_all"],
        pos_edges,
        neg_edges,
        max_edges=cfg.edge_sample_size,
        seed=cfg.seed + epoch,
    )

    total = (
        recon_loss
        + cfg.expr_kl_weight * expr_kl
        + cfg.graph_weight * cfg.graph_reconstruction_weight * (graph_rec + cfg.graph_kl_weight * graph_kl)
        + cfg.classifier_weight * fused_cls
        + cfg.expr_classifier_weight * expr_cls
        + cfg.graph_weight * cfg.graph_classifier_weight * graph_cls
    )
    parts = {
        "total": total,
        "recon": recon_loss,
        "expr_kl": expr_kl,
        "graph_rec": graph_rec,
        "graph_kl": graph_kl,
        "fused_cls": fused_cls,
        "expr_cls": expr_cls,
        "graph_cls": graph_cls,
    }
    return total, parts


@torch.no_grad()
def evaluate_model(model, X_scaled, y, H, adj_norm, eval_idx, cfg, batch_size=8192):
    model.eval()
    h_all = torch.tensor(H, dtype=torch.float32, device=device)
    adj_norm = adj_norm.to(device)
    z_graph_all, _, _ = model.graph_vae(h_all, adj_norm)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_true, y_pred = [], []

    for start in range(0, len(eval_idx), batch_size):
        idx_np = eval_idx[start:start + batch_size]
        xb = X_tensor[idx_np].to(device)
        idxb = torch.tensor(idx_np, dtype=torch.long, device=device)
        mu_expr, _ = model.expr_vae.encode(xb)
        graph_logits = model.graph_cls_head(z_graph_all[idxb])
        fused_logits = model.fused_cls_head(torch.cat([mu_expr, z_graph_all[idxb]], dim=1))
        logits = fused_logits + cfg.graph_weight * graph_logits
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y[idx_np].tolist())

    model.train()
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "selection_score": 0.7 * accuracy + 0.3 * macro_f1,
    }


def train_one_model(X_scaled, y, H, adj_norm, pos_edges, neg_edges, train_idx, val_idx, num_classes, cfg):
    ds = CellDataset(X_scaled, y, train_idx)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model = make_model(X_scaled.shape[1], H.shape[1], num_classes, cfg)
    h_all = torch.tensor(H, dtype=torch.float32, device=device)
    adj_norm = adj_norm.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # Keep training in float32 because CUDA sparse matrix multiplication does not support float16.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    history = []

    best = {"selection_score": -np.inf, "state": None, "metrics": None, "epoch": 0}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {
            "total": 0.0, "recon": 0.0, "expr_kl": 0.0, "graph_rec": 0.0,
            "graph_kl": 0.0, "fused_cls": 0.0, "expr_cls": 0.0, "graph_cls": 0.0,
        }
        n_batches = 0

        for xb, yb, idxb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            idxb = idxb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):
                out = model(xb, idxb, h_all, adj_norm)
                total, parts = multimodal_loss(model, xb, yb, idxb, out, pos_edges, neg_edges, cfg, epoch)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()

            for key in running:
                running[key] += float(parts[key].detach().cpu())
            n_batches += 1

        row = {key: value / max(n_batches, 1) for key, value in running.items()}
        row["epoch"] = epoch
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            val_metrics = evaluate_model(model, X_scaled, y, H, adj_norm, val_idx, cfg, cfg.batch_size)
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            if val_metrics["selection_score"] > best["selection_score"]:
                best = {
                    "selection_score": val_metrics["selection_score"],
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "metrics": val_metrics,
                    "epoch": epoch,
                }
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | Total: {row['total']:.4f} | "
                f"Recon: {row['recon']:.4f} | FusedCls: {row['fused_cls']:.4f} | "
                f"ExprCls: {row['expr_cls']:.4f} | GraphCls: {row['graph_cls']:.4f} | "
                f"ValF1: {val_metrics['macro_f1']:.4f} | ValAcc: {val_metrics['accuracy']:.4f} | "
                f"Score: {val_metrics['selection_score']:.4f}"
            )
        history.append(row)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    metrics = evaluate_model(model, X_scaled, y, H, adj_norm, val_idx, cfg, cfg.batch_size)
    return model, metrics, pd.DataFrame(history)


def train_with_internal_validation(synthetic_count_csv, synthetic_label_csv, synthetic_coord_csv,
                                   cfg=TrainConfig(),
                                   cell_id_col="cell_id", label_col="label", x_col="x", y_col="y",
                                   input_normalized="counts", target_input_normalized="counts",
                                   epoch_grid=None):
    X_df, y_raw, coord_df = load_synthetic_data(
        synthetic_count_csv,
        synthetic_label_csv,
        synthetic_coord_csv,
        cell_id_col=cell_id_col,
        label_col=label_col,
        x_col=x_col,
        y_col=y_col,
    )
    genes = X_df.columns.tolist()
    X_log = get_log_expression(X_df, input_normalized=input_normalized)
    gene_min, gene_max, denom = fit_gene_minmax_scaler(X_log)
    X_scaled = transform_gene_minmax_to_minus1_1(X_log, gene_min, denom)
    coords_norm, coord_center, coord_scale = normalize_coordinates(coord_df)
    H, _ = compute_spatial_node_features(coords_norm, k=cfg.graph_k_neighbors)
    adj_norm, pos_edges, neighbor_indices = build_knn_graph(coords_norm, k=cfg.graph_k_neighbors)
    neg_edges = sample_negative_edges(len(coords_norm), pos_edges, len(pos_edges), seed=cfg.seed)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    train_idx, val_idx = make_internal_split(coords_norm, val_fraction=cfg.val_fraction, method=cfg.split_method, seed=cfg.seed)
    print(f"Synthetic cells: {len(y)} | Train: {len(train_idx)} | Validation: {len(val_idx)}")

    epoch_grid = epoch_grid if epoch_grid is not None else [cfg.epochs]
    results = []
    best = {"selection_score": -np.inf, "model": None, "cfg": None, "metrics": None, "history": None}

    for epochs in epoch_grid:
        trial_cfg = TrainConfig(**asdict(cfg))
        trial_cfg.epochs = int(epochs)
        print()
        print(f"Training candidate | epochs={trial_cfg.epochs}")
        model, metrics, history = train_one_model(
            X_scaled, y, H, adj_norm, pos_edges, neg_edges,
            train_idx, val_idx, len(le.classes_), trial_cfg,
        )
        row = {"epochs": trial_cfg.epochs, **metrics}
        results.append(row)
        if metrics["selection_score"] > best["selection_score"]:
            best = {
                "selection_score": metrics["selection_score"],
                "model": model,
                "cfg": trial_cfg,
                "metrics": metrics,
                "history": history,
            }

    save_dir = Path(cfg.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    validation_results = pd.DataFrame(results).sort_values("selection_score", ascending=False)
    validation_results.to_csv(save_dir / "internal_validation_results.csv", index=False)
    best["history"].to_csv(save_dir / "training_history.csv", index=False)

    ckpt_path = save_dir / "binvast_best_model.pt"
    checkpoint = {
        "model_state_dict": best["model"].state_dict(),
        "config": {
            "expr_dim": X_scaled.shape[1],
            "graph_feat_dim": H.shape[1],
            "num_classes": len(le.classes_),
            "expr_latent_dim": best["cfg"].expr_latent_dim,
            "graph_latent_dim": best["cfg"].graph_latent_dim,
            "expr_hidden_dims": best["cfg"].expr_hidden_dims,
            "graph_hidden_dims": best["cfg"].graph_hidden_dims,
            "classifier_hidden_dim": best["cfg"].classifier_hidden_dim,
            "graph_k_neighbors": best["cfg"].graph_k_neighbors,
            "graph_weight": best["cfg"].graph_weight,
        },
        "train_config": asdict(best["cfg"]),
        "preprocess": {
            "genes": genes,
            "gene_min": gene_min.tolist(),
            "gene_max": gene_max.tolist(),
            "denom": denom.tolist(),
            "coord_center": coord_center.tolist(),
            "coord_scale": coord_scale.tolist(),
            "input_normalized": input_normalized,
            "target_input_normalized": target_input_normalized,
        },
        "label_classes": le.classes_.tolist(),
        "internal_validation": {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "best_metrics": best["metrics"],
        },
    }
    torch.save(checkpoint, ckpt_path)
    print(f"Best model saved to: {ckpt_path}")

    return {
        "model": best["model"],
        "checkpoint": checkpoint,
        "ckpt_path": str(ckpt_path),
        "validation_results": validation_results,
        "X_scaled": X_scaled,
        "y": y,
        "y_raw": y_raw,
        "coord_df": coord_df,
        "coords_norm": coords_norm,
        "H": H,
        "adj_norm": adj_norm,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def load_trained_model(ckpt_path):
    checkpoint = torch.load(
        Path(ckpt_path).expanduser(),
        map_location=device,
        weights_only=False,
    )
    cfg = checkpoint["config"]
    model = BinVAST(
        expr_dim=cfg["expr_dim"],
        graph_feat_dim=cfg["graph_feat_dim"],
        num_classes=cfg["num_classes"],
        expr_latent_dim=cfg["expr_latent_dim"],
        graph_latent_dim=cfg["graph_latent_dim"],
        expr_hidden_dims=tuple(cfg["expr_hidden_dims"]),
        graph_hidden_dims=tuple(cfg["graph_hidden_dims"]),
        classifier_hidden_dim=cfg["classifier_hidden_dim"],
        graph_weight=cfg.get("graph_weight", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def safe_probability_column(label):
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_")
    return f"prob_{safe}"


def prepare_target_inputs(target_count_csv, target_coord_csv, checkpoint,
                          cell_id_col="cell_id", x_col="x", y_col="y", target_input_normalized=None):
    X_df, coord_df = load_target_data(target_count_csv, target_coord_csv, cell_id_col=cell_id_col, x_col=x_col, y_col=y_col)
    X_scaled = preprocess_with_saved_params(X_df, checkpoint["preprocess"], input_normalized_override=target_input_normalized)
    coord_center = np.array(checkpoint["preprocess"]["coord_center"], dtype=np.float32)
    coord_scale = np.array(checkpoint["preprocess"]["coord_scale"], dtype=np.float32)
    coords_norm, _, _ = normalize_coordinates(coord_df, coord_center=coord_center, coord_scale=coord_scale)
    k_neighbors = checkpoint["config"].get("graph_k_neighbors", 100)
    H, _ = compute_spatial_node_features(coords_norm, k=k_neighbors)
    adj_norm, _, _ = build_knn_graph(coords_norm, k=k_neighbors)
    return X_df, coord_df, X_scaled, coords_norm, H, adj_norm


@torch.no_grad()
def predict_labels(target_count_csv, target_coord_csv, ckpt_path,
                   output_csv="target_predicted_labels.csv", batch_size=8192,
                   cell_id_col="cell_id", x_col="x", y_col="y", target_input_normalized=None):
    model, checkpoint = load_trained_model(ckpt_path)
    X_df, coord_df, X_scaled, coords_norm, H, adj_norm = prepare_target_inputs(
        target_count_csv,
        target_coord_csv,
        checkpoint,
        cell_id_col=cell_id_col,
        x_col=x_col,
        y_col=y_col,
        target_input_normalized=target_input_normalized,
    )
    classes = checkpoint["label_classes"]
    graph_weight = checkpoint["config"].get("graph_weight", 0.1)
    expr_latent, graph_latent = model.encode_all(X_scaled, H, adj_norm, batch_size=batch_size)

    preds, probs_max, probs_all = [], [], []
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    h_all = torch.tensor(H, dtype=torch.float32, device=device)
    adj_norm = adj_norm.to(device)
    z_graph_all, _, _ = model.graph_vae(h_all, adj_norm)

    for start in range(0, len(X_tensor), batch_size):
        stop = min(start + batch_size, len(X_tensor))
        xb = X_tensor[start:stop].to(device)
        idxb = torch.arange(start, stop, dtype=torch.long, device=device)
        mu_expr, _ = model.expr_vae.encode(xb)
        graph_logits = model.graph_cls_head(z_graph_all[idxb])
        fused_logits = model.fused_cls_head(torch.cat([mu_expr, z_graph_all[idxb]], dim=1))
        logits = fused_logits + graph_weight * graph_logits
        prob = torch.softmax(logits, dim=1)
        pmax, pidx = prob.max(dim=1)
        preds.extend([classes[j] for j in pidx.cpu().numpy()])
        probs_max.extend(pmax.cpu().numpy().tolist())
        probs_all.append(prob.cpu().numpy())

    out = pd.DataFrame({
        "cell_id": X_df.index,
        "x": coord_df["x"].values,
        "y": coord_df["y"].values,
        "predicted_label": preds,
        "prediction_confidence": probs_max,
    })
    probs_all = np.vstack(probs_all)
    for j, class_name in enumerate(classes):
        out[safe_probability_column(class_name)] = probs_all[:, j]
    output_csv = Path(output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Predictions saved to: {output_csv}")
    return out, expr_latent, graph_latent, X_df, coord_df, X_scaled, H, adj_norm


def save_latent_csv(cell_ids, labels, expr_latent, graph_latent, output_csv, split=None, coords=None):
    out = pd.DataFrame({"cell_id": cell_ids, "label": labels})
    if split is not None:
        out["split"] = split
    if coords is not None:
        out["x"] = coords["x"].values
        out["y"] = coords["y"].values
    for j in range(expr_latent.shape[1]):
        out[f"expr_latent_{j + 1}"] = expr_latent[:, j]
    for j in range(graph_latent.shape[1]):
        out[f"graph_latent_{j + 1}"] = graph_latent[:, j]
    output_csv = Path(output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Latents saved to: {output_csv}")


def visualize_latent_pca(expr_latent, graph_latent, labels, output_dir, prefix, random_state=259):
    print(f"Skipping {prefix} latent plots; latent CSV files are saved instead.")

# %% Cell 6
