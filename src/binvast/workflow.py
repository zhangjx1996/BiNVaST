"""High-level training and query-annotation workflow."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from .pipeline import run_full_synthetic_binvast_epoch_grid


REQUIRED_TARGET_INPUTS = (
    "target_expression_log1p_cp10k.csv",
    "target_coordinates.csv",
)

REQUIRED_SYNTHETIC_INPUTS = (
    "synthetic_expression.csv",
    "synthetic_labels.csv",
    "synthetic_spatial_coords.csv",
)


def validate_inputs(input_dir: str | Path, required_names: tuple[str, ...]) -> Path:
    """Validate that a workflow input directory contains the required files."""
    input_dir = Path(input_dir).expanduser().resolve()
    missing = [str(input_dir / name) for name in required_names if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))
    return input_dir


def aggregate_target_proportions(output_dir: str | Path) -> Path | None:
    """Combine cell-type percentages across saved epoch snapshots."""
    output_dir = Path(output_dir).expanduser().resolve()
    columns = []
    pattern = "epochs_*_batch_*/target_predicted_label_counts_with_legend_percent.csv"
    for csv_path in sorted(output_dir.glob(pattern)):
        table = pd.read_csv(csv_path)
        required = {"predicted_label", "percentage"}
        if not required.issubset(table.columns):
            continue
        columns.append(
            table.set_index("predicted_label")["percentage"].rename(csv_path.parent.name)
        )
    if not columns:
        return None

    summary = pd.concat(columns, axis=1).fillna(0.0).sort_index()
    summary.index.name = "cell_type"
    destination = output_dir / "target_cell_type_proportions_by_epoch_and_batch.csv"
    temporary = output_dir / f".{destination.name}.{os.getpid()}.tmp"
    summary.to_csv(temporary)
    os.replace(temporary, destination)
    return destination


def train_and_predict(
    project_dir: str | Path,
    target_data_dir: str | Path,
    synthetic_data_dir: str | Path,
    output_dir: str | Path,
    *,
    snapshot_epochs: tuple[int, ...] | list[int] = (50, 100, 150, 200),
    batch_size: int = 512,
    graph_k: int = 30,
    expr_latent_dim: int = 4,
    graph_latent_dim: int = 4,
    seed: int = 259,
) -> dict:
    """Train BinVAST and annotate the real query data at each requested epoch.

    The largest requested epoch is reported as the final snapshot. The returned
    dictionary includes the final per-cell label file and every snapshot output.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    target_data_dir = validate_inputs(target_data_dir, REQUIRED_TARGET_INPUTS)
    synthetic_data_dir = validate_inputs(synthetic_data_dir, REQUIRED_SYNTHETIC_INPUTS)
    output_dir = Path(output_dir).expanduser().resolve()
    epochs = sorted({int(epoch) for epoch in snapshot_epochs})
    if not epochs or epochs[0] < 1:
        raise ValueError("snapshot_epochs must contain positive integers")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = run_full_synthetic_binvast_epoch_grid(
        project_root=project_dir,
        target_dir_name=target_data_dir,
        synthetic_dir_name=synthetic_data_dir,
        output_name=output_dir,
        snapshot_epochs=epochs,
        batch_size=batch_size,
        graph_k=graph_k,
        expr_latent_dim=expr_latent_dim,
        graph_latent_dim=graph_latent_dim,
        seed=seed,
    )

    proportion_csv = aggregate_target_proportions(output_dir)
    final_epoch = max(epochs)
    final_output_dir = Path(result["snapshots"][final_epoch]["output_dir"])
    final_labels = final_output_dir / "target_predicted_labels.csv"
    manifest = {
        "status": "completed",
        "project_dir": str(project_dir),
        "target_data_dir": str(target_data_dir),
        "synthetic_data_dir": str(synthetic_data_dir),
        "output_dir": str(output_dir),
        "batch_size": int(batch_size),
        "snapshot_epochs": epochs,
        "continuous_training_max_epoch": final_epoch,
        "graph_k": int(graph_k),
        "expr_latent_dim": int(expr_latent_dim),
        "graph_latent_dim": int(graph_latent_dim),
        "seed": int(seed),
        "elapsed_minutes": (time.time() - started) / 60.0,
        "proportion_csv": str(proportion_csv) if proportion_csv else None,
        "final_predicted_labels": str(final_labels),
        "final_checkpoint": str(final_output_dir / "binvast_best_model.pt"),
        "snapshot_output_dirs": {
            str(epoch): str(snapshot["output_dir"])
            for epoch, snapshot in result["snapshots"].items()
        },
    }
    manifest_path = output_dir / "binvast_workflow_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": manifest, "manifest_path": manifest_path, "result": result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BinVAST and predict labels for real query spatial data."
    )
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--target-data-dir", type=Path, required=True)
    parser.add_argument("--synthetic-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-epochs", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--graph-k", type=int, default=30)
    parser.add_argument("--expr-latent-dim", type=int, default=4)
    parser.add_argument("--graph-latent-dim", type=int, default=4)
    parser.add_argument("--seed", type=int, default=259)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completed = train_and_predict(
        project_dir=args.project_dir,
        target_data_dir=args.target_data_dir,
        synthetic_data_dir=args.synthetic_data_dir,
        output_dir=args.output_dir,
        snapshot_epochs=args.snapshot_epochs,
        batch_size=args.batch_size,
        graph_k=args.graph_k,
        expr_latent_dim=args.expr_latent_dim,
        graph_latent_dim=args.graph_latent_dim,
        seed=args.seed,
    )
    print(json.dumps(completed["manifest"], indent=2), flush=True)


if __name__ == "__main__":
    main()
